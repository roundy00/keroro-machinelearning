# backend/model_runner.py
import os
import json
import yaml
import numpy as np
import torch

from imdiffusion.main_model import CSDI_Physio

class IMDiffusionRunner:
    """
    실시간 입력 window (L,K) -> score_now, residual_series(L), optional uncertainty
    dataset.py/TestData + utils.window_trick_evaluate_middle의 핵심만 온라인으로 재구현
    """
    def __init__(
        self,
        config_yaml_path: str,
        ckpt_path: str,
        device: str = "cuda:0",
        feature_dim: int = 38,      # SMD=38
        window_length: int = 100,   # dataset.py 기본
        split: int = 4,
        multiply_20: bool = True,   # dataset.py에서 *20 했음
        nsample: int = 1,
        use_middle_uncertainty: bool = True,
    ):
        self.device = device
        self.feature_dim = feature_dim
        self.window_length = window_length
        self.split = split
        self.multiply_20 = multiply_20
        self.nsample = nsample
        self.use_middle_uncertainty = use_middle_uncertainty

        with open(config_yaml_path, "r") as f:
            config = yaml.safe_load(f)

        # 실시간은 evaluation이므로 보통 unconditional=False 그대로(학습 세팅과 동일) 로드
        self.config = config

        self.model = CSDI_Physio(
            config=self.config,
            device=self.device,
            target_dim=self.feature_dim,
            ratio=0.7,  # 학습과 맞추기
        ).to(self.device)

        assert os.path.exists(ckpt_path), f"ckpt not found: {ckpt_path}"
        self.model.load_state_dict(torch.load(ckpt_path, map_location=self.device))
        self.model.eval()

    # -------------------------
    # dataset.py와 동일한 mask 생성 (TestData.get_mask)
    # -------------------------
    def _make_gt_mask(self, L: int, K: int, strategy: int) -> torch.Tensor:
        """
        strategy=0: split segment 중 짝수 segment를 gt_mask=1 (관측으로 제공)
        strategy=1: split segment 중 홀수 segment를 gt_mask=1
        """
        observed_mask = torch.ones((L, K), dtype=torch.float32)
        mask = torch.zeros_like(observed_mask)

        skip = L // self.split
        for split_index, begin in enumerate(range(0, L, skip)):
            end = min(begin + skip, L)
            if strategy == 0:
                if split_index % 2 == 0:
                    mask[begin:end, :] = 1.0
            else:
                if split_index % 2 != 0:
                    mask[begin:end, :] = 1.0
        return mask

    def _build_batch(self, window: np.ndarray, strategy: int):
        """
        main_model.CSDI_Physio.process_data 가 기대하는 배치 dict 형식 만들기
        dataset.py/TestData.__getitem__과 동일한 키 사용
        """
        assert window.shape == (self.window_length, self.feature_dim), window.shape

        x = window.astype(np.float32)
        if self.multiply_20:
            x = x * 20.0  # dataset.py에서 이렇게 했음

        observed_data = torch.from_numpy(x)                         # (L,K)
        observed_mask = torch.ones_like(observed_data)              # (L,K)
        gt_mask = self._make_gt_mask(self.window_length, self.feature_dim, strategy)  # (L,K)
        timepoints = torch.from_numpy(np.arange(self.window_length).astype(np.float32))  # (L,)
        strategy_type = torch.tensor(strategy, dtype=torch.long)

        # batch dimension 붙이기: (B=1, L, K)
        batch = {
            "observed_data": observed_data.unsqueeze(0),
            "observed_mask": observed_mask.unsqueeze(0),
            "gt_mask": gt_mask.unsqueeze(0),
            "timepoints": timepoints.unsqueeze(0),  # (1,L)
            "strategy_type": strategy_type.unsqueeze(0),  # (1,)
        }
        return batch

    # -------------------------
    # compute_score.py의 SMD 세팅 반영: compute_abs=True, compute_sum=False
    # residual[t] = max_k |diff|
    # -------------------------
    def _residual_series_smd(self, recon: torch.Tensor, target: torch.Tensor) -> np.ndarray:
        """
        recon, target: (L,K) torch (same device ok)
        return: (L,) np
        """
        diff = torch.abs(recon - target)
        residual, _ = torch.max(diff, dim=-1)  # (L,)
        return residual.detach().cpu().numpy().astype(np.float32)

    @torch.no_grad()
    def score_window(self, window: np.ndarray):
        """
        window: (L,K) numpy
        return dict: score_now, residual_series, recon_last(optional), uncertainty(optional)
        """
        # strategy 0
        b0 = self._build_batch(window, strategy=0)
        # strategy 1
        b1 = self._build_batch(window, strategy=1)

        if self.use_middle_uncertainty:
            # get_middle_evaluate -> (samples, observed_data, target_mask, observed_mask, observed_tp, middle)
            out0 = self.model.get_middle_evaluate(b0, n_samples=self.nsample)
            out1 = self.model.get_middle_evaluate(b1, n_samples=self.nsample)

            # samples: (B, nsample, K, L)  / observed_data: (B,K,L) / target_mask: (B,K,L) / middle:(B,num_steps,K,L)
            samples0, obs0, tgtmask0, _, _, mid0 = out0
            samples1, obs1, tgtmask1, _, _, mid1 = out1

            # utils.window_trick_evaluate_middle의 merge 로직 반영
            # 1) 최종 sample은 nsample=1이면 그냥 squeeze해서 (B,K,L)
            # 2) target_mask(tgtmask)는 (B,K,L)인데, utils에서는 eval_points로 사용 후 (B,L,K)로 맞추는 편
            # 여기서는 (L,K)로 맞춰서 합침.
            recon0 = samples0[:, 0]  # (B,K,L)
            recon1 = samples1[:, 0]  # (B,K,L)

            # target_mask: observed_mask - gt_mask (예측해야 하는 위치가 1)
            # strategy0/1이 서로 보완이므로 recon = recon0*tgtmask0 + recon1*tgtmask1
            recon = recon0 * tgtmask0 + recon1 * tgtmask1  # (B,K,L)

            # mid merge도 동일하게
            # mid: (B,num_steps,K,L)
            mid = mid0 * tgtmask0.unsqueeze(1) + mid1 * tgtmask1.unsqueeze(1)  # (B,num_steps,K,L)

            # 원본은 obs0 (== obs1) : (B,K,L)
            # (L,K)로 바꾸기
            recon_LK = recon[0].permute(1, 0)  # (L,K)
            target_LK = obs0[0].permute(1, 0)  # (L,K)

            residual = self._residual_series_smd(recon_LK, target_LK)
            score_now = float(residual[-1])

            # uncertainty: mid의 “시간/스텝” 방향 분산 (마지막 시점의 불확실성)
            # mid: (num_steps, K, L) -> (num_steps, L, K)
            mid_LK = mid[0].permute(0, 3, 2)  # (num_steps, L, K)
            # 마지막 시점 L-1에서 feature별 std -> max로 요약
            std_last = torch.std(mid_LK[:, -1, :], dim=0)      # (K,)
            unc_now = float(torch.max(std_last).detach().cpu().item())

            return {
                "score_now": score_now,
                "residual_series": residual,  # (L,)
                "uncertainty_now": unc_now,
            }

        else:
            out0 = self.model.evaluate(b0, n_samples=self.nsample)
            out1 = self.model.evaluate(b1, n_samples=self.nsample)

            samples0, obs0, tgtmask0, _, _ = out0
            samples1, obs1, tgtmask1, _, _ = out1

            recon0 = samples0[:, 0]  # (B,K,L)
            recon1 = samples1[:, 0]  # (B,K,L)

            recon = recon0 * tgtmask0 + recon1 * tgtmask1

            recon_LK = recon[0].permute(1, 0)
            target_LK = obs0[0].permute(1, 0)

            residual = self._residual_series_smd(recon_LK, target_LK)
            score_now = float(residual[-1])

            return {
                "score_now": score_now,
                "residual_series": residual,
            }

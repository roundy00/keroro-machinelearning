# backend/alert_engine.py
from dataclasses import dataclass
from collections import deque

@dataclass
class AlertState:
    level: str               # "OK" | "WATCH" | "ALARM"
    score_now: float
    uncertainty_now: float | None
    reason: str

class AlertEngine:
    def __init__(
        self,
        alarm_th: float,
        watch_th: float,
        unc_th: float | None = None,
        consecutive_alarm: int = 3,
        consecutive_watch: int = 2,
        history: int = 60
    ):
        self.alarm_th = alarm_th
        self.watch_th = watch_th
        self.unc_th = unc_th
        self.consecutive_alarm = consecutive_alarm
        self.consecutive_watch = consecutive_watch
        self.scores = deque(maxlen=history)
        self.uncs = deque(maxlen=history)
        self._alarm_run = 0
        self._watch_run = 0

    def update(self, score_now: float, uncertainty_now: float | None = None) -> AlertState:
        self.scores.append(score_now)
        if uncertainty_now is not None:
            self.uncs.append(uncertainty_now)

        # alarm logic
        if score_now >= self.alarm_th:
            self._alarm_run += 1
        else:
            self._alarm_run = 0

        # watch logic (score 또는 uncertainty)
        watch_hit = score_now >= self.watch_th
        if self.unc_th is not None and uncertainty_now is not None:
            watch_hit = watch_hit or (uncertainty_now >= self.unc_th)

        if watch_hit:
            self._watch_run += 1
        else:
            self._watch_run = 0

        if self._alarm_run >= self.consecutive_alarm:
            return AlertState("ALARM", score_now, uncertainty_now, reason="score>=alarm_th")
        if self._watch_run >= self.consecutive_watch:
            reason = "score>=watch_th" if score_now >= self.watch_th else "uncertainty spike"
            return AlertState("WATCH", score_now, uncertainty_now, reason=reason)
        return AlertState("OK", score_now, uncertainty_now, reason="normal")

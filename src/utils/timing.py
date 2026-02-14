from __future__ import annotations

import time
from dataclasses import dataclass


@dataclass
class Timer:
    start: float

    @classmethod
    def begin(cls) -> "Timer":
        return cls(start=time.perf_counter())

    def ms(self) -> float:
        return (time.perf_counter() - self.start) * 1000.0

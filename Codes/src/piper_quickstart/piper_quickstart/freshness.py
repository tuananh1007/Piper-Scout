"""Judge a stream's liveness by its header stamps, not by message arrival.

The Piper driver republishes the SDK's *cached* joint values at 200 Hz whether
or not new CAN frames arrived; header.stamp is the receive time of the last
real frame. So a stream can keep arriving while its data is frozen. A
message only counts as new data if its stamp is later than any seen before.
"""

from __future__ import annotations

from typing import Optional


class StampWatch:
    def __init__(self) -> None:
        self.last_stamp_ns: Optional[int] = None
        self.last_new_mono: Optional[float] = None      # when the stamp last advanced
        self.last_arrival_mono: Optional[float] = None  # when any message last arrived
        self.new_count = 0

    def update(self, stamp_ns: int, now_mono: float) -> bool:
        """Record a message; True if it carries new data (stamp advanced)."""
        self.last_arrival_mono = now_mono
        if self.last_stamp_ns is not None and stamp_ns <= self.last_stamp_ns:
            return False
        self.last_stamp_ns = stamp_ns
        self.last_new_mono = now_mono
        self.new_count += 1
        return True

    def data_age(self, now_mono: float) -> float:
        """Seconds since the stamp last advanced (inf if it never has)."""
        return float("inf") if self.last_new_mono is None else now_mono - self.last_new_mono

    def arrival_age(self, now_mono: float) -> float:
        """Seconds since any message arrived (inf if none has)."""
        return float("inf") if self.last_arrival_mono is None else now_mono - self.last_arrival_mono


def stamp_to_ns(stamp) -> int:
    return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)

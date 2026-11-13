"""Minimum normalized-aperture end-effector interface used by composed arm drivers.

Hands with multiple joints or other action schemas should expose their native Robot
adapter and recording processors instead of squeezing commands into one scalar.
"""

from typing import Protocol, runtime_checkable


@runtime_checkable
class EndEffectorDriver(Protocol):
    def connect(self) -> None: ...
    def disconnect(self) -> None: ...
    def get_position(self) -> float:
        """Measured normalized aperture: 0 closed, 1 open."""
        ...

    def set_position(self, normalized_position: float) -> None:
        """Request aperture; hardware I/O must not block the arm's realtime loop."""
        ...

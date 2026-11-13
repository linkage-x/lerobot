"""Hardware configuration catalog shared by browser forms and runtime configuration.

Plugins may register a HardwareModel after registering their Robot/Camera/Teleoperator
config. Registration describes configuration only; it never connects to a device.
"""

from .catalog import HardwareModel, catalog, get_model, register_model
from .interfaces import EndEffectorDriver

__all__ = ["HardwareModel", "EndEffectorDriver", "catalog", "get_model", "register_model"]

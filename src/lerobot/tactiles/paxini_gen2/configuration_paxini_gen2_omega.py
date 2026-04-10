# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import InitVar, dataclass, field

from serial import Serial

from ..configs import TactileConfig

PAXINI_NUM_TAXELS = 120
PAXINI_NUM_DIMENSIONS = 3


@TactileConfig.register_subclass("paxini_gen2_omega")
@dataclass
class PaxiniGen2OmegaTactileConfig(TactileConfig):
    serial_port: str
    baudrate: int
    control_mode: int
    model_name: str
    connect_ids: list[int]
    timeout: float = 1.0
    provided_serial: InitVar[Serial | None] = None
    serial: Serial | None = field(default=None, repr=False, compare=False)

    def __post_init__(self, provided_serial: Serial | None) -> None:
        self.model_config_map = {
            "GEN1-IP-S2516": {"port_offset": 0, "control_mode": 2},
            "GEN1-DP-S2716": {"port_offset": 1, "control_mode": 2},
            "GEN2-IP-L5325": {"port_offset": 0, "control_mode": 5},
            "GEN2-IP-M3025": {"port_offset": 0, "control_mode": 5},
            "GEN2-MP-M2324": {"port_offset": 1, "control_mode": 5},
            "GEN2-DP-L3530": {"port_offset": 2, "control_mode": 5},
            "GEN2-DP-M2826": {"port_offset": 2, "control_mode": 5},
            "GEN2-DP-S2716": {"port_offset": 0, "control_mode": 1},
        }

        if not self.serial_port:
            raise ValueError("`serial_port` cannot be empty.")
        if self.baudrate <= 0:
            raise ValueError(f"`baudrate` must be > 0, but {self.baudrate} is provided.")
        if self.timeout <= 0:
            raise ValueError(f"`timeout` must be > 0, but {self.timeout} is provided.")
        if not self.connect_ids:
            raise ValueError("`connect_ids` must contain at least one module id.")
        if any(connect_id <= 0 for connect_id in self.connect_ids):
            raise ValueError(f"`connect_ids` must only contain values >= 1, but {self.connect_ids} is provided.")
        if len(set(self.connect_ids)) != len(self.connect_ids):
            raise ValueError(f"`connect_ids` must not contain duplicates, but {self.connect_ids} is provided.")

        if self.model_name not in self.model_config_map:
            raise ValueError(f"Unsupported sensor model: {self.model_name}")

        model_config = self.model_config_map[self.model_name]
        expected_control_mode = model_config["control_mode"]
        if self.control_mode != expected_control_mode:
            raise ValueError(
                f"Sensor model {self.model_name} requires control_mode={expected_control_mode}, "
                f"but {self.control_mode} is provided."
            )

        if self.num_taxels is None:
            self.num_taxels = PAXINI_NUM_TAXELS * len(self.connect_ids)
        if self.num_dimensions is None:
            self.num_dimensions = PAXINI_NUM_DIMENSIONS

        self.port_offset = model_config["port_offset"]
        self.serial = provided_serial

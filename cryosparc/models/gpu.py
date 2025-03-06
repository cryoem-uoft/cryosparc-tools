# THIS FILE IS AUTO-GENERATED, DO NOT EDIT DIRECTLY
# SEE dev/api_generate_models.py
from typing import Any, Literal

from pydantic import BaseModel


class Gpu(BaseModel):
    """
    GPU details for a target.
    """

    id: int
    name: str
    mem: int


class GpuInfo(BaseModel):
    id: int
    name: str
    mem: int
    bus_id: str = ""
    compute_mode: Any = "Default"
    persistence_mode: Literal["Disabled", "Enabled"] = "Disabled"
    power_limit: float = 0.0
    sw_power_limit: Literal["Not Active", "Active"] = "Not Active"
    hw_power_limit: Literal["Not Active", "Active"] = "Not Active"
    max_pcie_link_gen: int = 0
    current_pcie_link_gen: int = 0
    temperature: int = 0
    gpu_utilization: int = 0
    memory_utilization: int = 0
    driver_version: str = ""

# THIS FILE IS AUTO-GENERATED, DO NOT EDIT DIRECTLY
# SEE dev/api_generate_models.py
from typing import Literal

from pydantic import BaseModel


class Gpu(BaseModel):
    """
    GPU details for a target.
    """

    id: int
    """
    Index of GPU. Generally based on which PCI slot the GPU occupies.
    """
    name: str
    """
    Identifiable model name for this GPU, e.g.,"GeForce RTX 3090".
    """
    mem: int
    """
    Amount of memory available on this GPU, in bytes.
    """


class GpuInfo(BaseModel):
    id: int
    """
    Index of GPU. Generally based on which PCI slot the GPU occupies.
    """
    name: str
    """
    Identifiable model name for this GPU, e.g.,"GeForce RTX 3090".
    """
    mem: int
    """
    Amount of memory available on this GPU, in bytes.
    """
    bus_id: str = ""
    """
    PCI-Express bus address at which GPU is mounted
    """
    compute_mode: Literal["Default", "Exclusive Thread", "Prohibited", "Exclusive Process"] = "Default"
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

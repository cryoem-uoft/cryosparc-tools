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
    """
    Compute mode set for this GPU.
    """
    persistence_mode: Literal["Disabled", "Enabled"] = "Disabled"
    """
    Persistence mode state for this GPU, either Enabled or Disabled.
    """
    power_limit: float = 0.0
    """
    Power limit for this GPU, in watts.
    """
    sw_power_limit: Literal["Not Active", "Active"] = "Not Active"
    """
    Whether software power limit is currently active.
    """
    hw_power_limit: Literal["Not Active", "Active"] = "Not Active"
    """
    Whether hardware power limit is currently active.
    """
    max_pcie_link_gen: int = 0
    """
    Maximum PCI-Express link generation supported by this GPU.
    """
    current_pcie_link_gen: int = 0
    """
    Current PCI-Express link generation for this GPU.
    """
    temperature: int = 0
    """
    Current temperature of this GPU, in degrees Celsius.
    """
    gpu_utilization: int = 0
    """
    Current GPU utilization, in percent.
    """
    memory_utilization: int = 0
    """
    Current memory utilization, in percent.
    """
    driver_version: str = ""
    """
    Version of the GPU driver installed on the target system.
    """

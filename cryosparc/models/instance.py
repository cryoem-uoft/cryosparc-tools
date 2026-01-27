# THIS FILE IS AUTO-GENERATED, DO NOT EDIT DIRECTLY
# SEE dev/api_generate_models.py
from typing import List, Optional

from pydantic import BaseModel

from .gpu import GpuInfo


class InstanceInformation(BaseModel):
    """
    System information for a machine running a CryoSPARC master or worker installation.
    """

    platform_node: str = ""
    """
    Node hostname
    """
    platform_release: str = ""
    """
    Linux kernel release version
    """
    platform_version: str = ""
    """
    Platform version string
    """
    platform_architecture: str = ""
    """
    Platform architecture (e.g., x86_64)
    """
    cpu_model: str = ""
    """
    CPU model name
    """
    physical_cores: int = 0
    """
    Number of physical CPU cores
    """
    max_cpu_freq: float = 0.0
    """
    Maximum CPU frequency in MHz
    """
    total_memory: str = "0B"
    """
    Total system memory, formatted as a string with units (e.g., "1.23GB")
    """
    available_memory: str = "0B"
    """
    Available system memory, formatted as a string with units (e.g., "1.23GB")
    """
    used_memory: str = "0B"
    """
    Used system memory, formatted as a string with units (e.g., "1.23GB")
    """
    ofd_soft_limit: int = 0
    """
    Maximum allowed open file descriptors (soft limit)
    """
    ofd_hard_limit: int = 0
    """
    Maximum allowed open file descriptors (hard limit)
    """
    driver_version: Optional[str] = None
    """
    Maximum CUDA version supported by Nvidia driver (e.g., "12.9")
    """
    toolkit_version: Optional[str] = None
    """
    Installed CUDA toolkit version (e.g., "12.8")
    """
    CUDA_version: Optional[str] = None
    """
    same as toolkit_version, for backwards compatibility :meta private:
    """
    nvrtc_version: Optional[str] = None
    """
    Installed NVRTC version (e.g., "12.8")
    """
    gpu_info: Optional[List[GpuInfo]] = None
    """
    List of GPU information objects
    """
    version: str = ""
    """
    Installed CryoSPARC version
    """

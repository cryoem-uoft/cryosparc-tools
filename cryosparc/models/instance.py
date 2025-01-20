# THIS FILE IS AUTO-GENERATED, DO NOT EDIT DIRECTLY
# SEE dev/api_generate_models.py
from typing import List, Optional

from pydantic import BaseModel

from .gpu import GpuInfo


class InstanceInformation(BaseModel):
    platform_node: str = ""
    platform_release: str = ""
    platform_version: str = ""
    platform_architecture: str = ""
    cpu_model: str = ""
    physical_cores: int = 0
    max_cpu_freq: float = 0.0
    total_memory: str = "0B"
    available_memory: str = "0B"
    used_memory: str = "0B"
    ofd_soft_limit: int = 0
    ofd_hard_limit: int = 0
    driver_version: Optional[str] = None
    toolkit_version: Optional[str] = None
    CUDA_version: Optional[str] = None
    nvrtc_version: Optional[str] = None
    gpu_info: Optional[List[GpuInfo]] = None
    version: str = ""

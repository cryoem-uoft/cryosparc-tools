# THIS FILE IS AUTO-GENERATED, DO NOT EDIT DIRECTLY
# SEE dev/api_generate_models.py
import datetime
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field

from .instance import InstanceInformation


class PerformanceBenchmark(BaseModel):
    id: str = Field("000000000000000000000000", alias="_id")
    updated_at: datetime.datetime = datetime.datetime(1970, 1, 1, 0, 0, tzinfo=datetime.timezone.utc)
    """
    When this object was last modified.
    """
    created_at: datetime.datetime = datetime.datetime(1970, 1, 1, 0, 0, tzinfo=datetime.timezone.utc)
    """
    When this object was first created. Imported objects such as projects
    and jobs will retain the created time from their original CryoSPARC instance.
    """
    project_uid: str
    job_uid: str
    type: Literal["fs", "cpu", "gpu", "extensive_validation"]
    cryosparc_version: str
    instance_information: InstanceInformation
    job_params: Dict[str, Any] = {}
    timings: Dict[str, Any] = {}
    gpu_name: Optional[str] = None
    gpu_bus_id: Optional[str] = None
    sequential_read_benchmark_dir: Optional[str] = None
    sequential_disk_models: List[Dict[str, Any]] = []
    particle_read_benchmark_dir: Optional[str] = None
    random_disk_models: List[Dict[str, Any]] = []


class ReferencePerformanceBenchmark(BaseModel):
    id: str = Field("000000000000000000000000", alias="_id")
    updated_at: datetime.datetime = datetime.datetime(1970, 1, 1, 0, 0, tzinfo=datetime.timezone.utc)
    """
    When this object was last modified.
    """
    created_at: datetime.datetime = datetime.datetime(1970, 1, 1, 0, 0, tzinfo=datetime.timezone.utc)
    """
    When this object was first created. Imported objects such as projects
    and jobs will retain the created time from their original CryoSPARC instance.
    """
    type: Literal["fs", "cpu", "gpu", "extensive_validation"]
    title: str
    cryosparc_version: str
    instance_information: InstanceInformation = InstanceInformation()
    job_params: Dict[str, Any] = {}
    timings: Dict[str, Any] = {}
    gpu_name: Optional[str] = None
    gpu_bus_id: Optional[str] = None
    reference_platform_node: Optional[str] = None

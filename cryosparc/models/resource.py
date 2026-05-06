# THIS FILE IS AUTO-GENERATED, DO NOT EDIT DIRECTLY
# SEE dev/api_generate_models.py
from typing import List

from pydantic import BaseModel


class ResourceSlots(BaseModel):
    """
    Available compute resources on a worker node that are allocated to a job
    when it runs.
    """

    CPU: List[int] = []
    """
    List of available CPU core numbers. For example, if worker has 4 CPU
    cores allocated, is set to ``[0, 1, 2, 3]``.
    """
    GPU: List[int] = []
    """
    List of available GPU numbers. For example, if a worker has 2 GPUs,
    is set to ``[0, 1]``. If a worker is configured with subset of GPUs, only
    that subset is listed here e.g., ``[1]``.
    """
    RAM: List[int] = []
    """
    List of available 8GB slots. For example, if a worker has 32GB of RAM,
    is set to ``[0, 1, 2, 3]``.
    """


class FixedResourceSlots(BaseModel):
    """
    Available worker node compute resources that only indicate presence, not the
    amount that may be allocated. (e.g., "SSD is available or not available")
    """

    SSD: bool = False
    """
    Whether this target thas an SSD
    """


class ResourceSpec(BaseModel):
    """
    Job resource requirements. Used to allocate compute resources for a job
    at queue time.
    """

    cpu: int = 1
    """
    Number of required CPU cores
    """
    gpu: int = 0
    """
    Number of required GPUs
    """
    ram: int = 1
    """
    Number of 8GiB RAM slots
    """
    ssd: bool = False
    """
    Whether an SSD is required for temporary storage.
    """

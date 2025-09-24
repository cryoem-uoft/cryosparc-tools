# THIS FILE IS AUTO-GENERATED, DO NOT EDIT DIRECTLY
# SEE dev/api_generate_models.py
from typing import List

from pydantic import BaseModel


class ResourceSlots(BaseModel):
    """
    Listings of available resources on a worker node that may be allocated for
    scheduling.
    """

    CPU: List[int] = []
    """
    List of available CPU core indices.
    """
    GPU: List[int] = []
    """
    List of available GPU indices.
    """
    RAM: List[int] = []
    """
    List of available 8GB slots.
    """


class FixedResourceSlots(BaseModel):
    """
    Available resource slots that only indicate presence, not the amount that
    may be allocated. (i.e., "SSD is available or not available")
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

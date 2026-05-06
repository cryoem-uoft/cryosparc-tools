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

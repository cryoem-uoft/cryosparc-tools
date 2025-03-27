from datetime import datetime
from typing import List, TypeVar

from typing_extensions import TypedDict

from .models.job import JobStatus
from .models.job_spec import Category

T = TypeVar("T")
Eq = T
In = T | List[T]
Range = tuple[T, T]


class JobSearch(TypedDict, total=False):
    """
    Type of keyword arguments for ``find_jobs`` methods.
    """

    type: In[str]
    """
    Include jobs with the given type.
    """
    status: In[JobStatus]
    """
    Include jobs with the given status.
    """
    category: In[Category]
    """
    Include jobs with the given category.
    """
    created_at: Range[datetime]
    """
    Include jobs created within the given time range.
    """
    updated_at: Range[datetime]
    """
    Include jobs updated within the given time range.
    """
    queued_at: Range[datetime]
    """
    Include jobs queued within the given time range.
    """
    started_at: Range[datetime]
    """
    Include jobs started within the given time range.
    """
    waiting_at: Range[datetime]
    """
    Include jobs that started waiting within the given time range.
    """
    completed_at: Range[datetime]
    """
    Include jobs completed within the given time range.
    """
    killed_at: Range[datetime]
    """
    Include jobs killed within the given time range.
    """
    failed_at: Range[datetime]
    """
    Include jobs failed within the given time range.
    """
    exported_at: Range[datetime]
    """
    Include jobs exported within the given time range.
    """

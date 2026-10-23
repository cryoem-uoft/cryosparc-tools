# THIS FILE IS AUTO-GENERATED, DO NOT EDIT DIRECTLY
# SEE dev/api_generate_models.py
import datetime
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


class BlueprintParameter(BaseModel):
    value: Any
    """
    """
    flagged: bool = False
    """
    """
    notes: str = ""
    """
    """


class Blueprint(BaseModel):
    id: str = Field("000000000000000000000000", alias="_id")
    """
    """
    updatedAt: datetime.datetime = datetime.datetime(1970, 1, 1, 0, 0, tzinfo=datetime.timezone.utc)
    """
    """
    createdAt: datetime.datetime = datetime.datetime(1970, 1, 1, 0, 0, tzinfo=datetime.timezone.utc)
    """
    """
    jobType: str
    """
    """
    title: str = ""
    """
    """
    description: str = ""
    """
    """
    applyTitle: bool = True
    """
    """
    applyDescription: bool = True
    """
    """
    csVersion: str
    """
    """
    blueprintVersion: str = "1.0.0"
    """
    """
    createdBy: Optional[str] = None
    """
    """
    updatedBy: Optional[str] = None
    """
    """
    pinned: bool = False
    """
    """
    imported: bool = False
    """
    """
    parameters: Dict[str, BlueprintParameter] = {}
    """
    """
    reference: bool = False
    """
    """

# THIS FILE IS AUTO-GENERATED, DO NOT EDIT DIRECTLY
# SEE dev/api_generate_models.py
import datetime
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field

from .asset import GridFSAsset


class CheckpointEvent(BaseModel):
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
    cpumem_mb: Optional[float] = None
    avail_mb: Optional[float] = None
    flags: List[str] = []
    meta: Dict[str, Any] = {}
    type: Literal["checkpoint"]


class Event(BaseModel):
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
    cpumem_mb: Optional[float] = None
    avail_mb: Optional[float] = None
    flags: List[str] = []
    meta: Dict[str, Any] = {}


class ImageEvent(BaseModel):
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
    cpumem_mb: Optional[float] = None
    avail_mb: Optional[float] = None
    flags: List[str] = []
    meta: Dict[str, Any] = {}
    type: Literal["image"]
    text: str
    imgfiles: List[GridFSAsset] = []


class InteractiveGridFSAsset(BaseModel):
    fileid: str = "000000000000000000000000"
    filename: str
    filetype: str


class InteractiveImgfile(BaseModel):
    imgfiles: List[GridFSAsset]
    components: List[int] = []


class InteractiveEvent(BaseModel):
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
    cpumem_mb: Optional[float] = None
    avail_mb: Optional[float] = None
    flags: List[str] = []
    meta: Dict[str, Any] = {}
    type: Literal["interactive"]
    subtype: str = "3dscatter"
    text: str
    datafile: InteractiveGridFSAsset
    preview_imgfiles: List[InteractiveImgfile] = []
    components: List[int] = []


class TextEvent(BaseModel):
    """
    An event with only text and no additional image or interactive data. May
    have "text", "warning" or "error" type.
    """

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
    cpumem_mb: Optional[float] = None
    avail_mb: Optional[float] = None
    flags: List[str] = []
    meta: Dict[str, Any] = {}
    type: Literal["text", "warning", "error"]
    text: str

# THIS FILE IS AUTO-GENERATED, DO NOT EDIT DIRECTLY
# SEE dev/api_generate_models.py
import datetime
from typing import List, Literal, Optional

from pydantic import BaseModel, Field

from .mongo import GridFSAsset


class CheckpointEvent(BaseModel):
    id: str = Field("000000000000000000000000", alias="_id")
    updated_at: datetime.datetime = datetime.datetime(1970, 1, 1, 0, 0, tzinfo=datetime.timezone.utc)
    created_at: datetime.datetime = datetime.datetime(1970, 1, 1, 0, 0, tzinfo=datetime.timezone.utc)
    project_uid: str
    job_uid: str
    cpumem_mb: Optional[float] = None
    avail_mb: Optional[float] = None
    flags: List[str] = []
    meta: dict = {}
    type: str


class Event(BaseModel):
    id: str = Field("000000000000000000000000", alias="_id")
    updated_at: datetime.datetime = datetime.datetime(1970, 1, 1, 0, 0, tzinfo=datetime.timezone.utc)
    created_at: datetime.datetime = datetime.datetime(1970, 1, 1, 0, 0, tzinfo=datetime.timezone.utc)
    project_uid: str
    job_uid: str
    cpumem_mb: Optional[float] = None
    avail_mb: Optional[float] = None
    flags: List[str] = []
    meta: dict = {}


class ImageEvent(BaseModel):
    id: str = Field("000000000000000000000000", alias="_id")
    updated_at: datetime.datetime = datetime.datetime(1970, 1, 1, 0, 0, tzinfo=datetime.timezone.utc)
    created_at: datetime.datetime = datetime.datetime(1970, 1, 1, 0, 0, tzinfo=datetime.timezone.utc)
    project_uid: str
    job_uid: str
    cpumem_mb: Optional[float] = None
    avail_mb: Optional[float] = None
    flags: List[str] = []
    meta: dict = {}
    type: str
    text: str
    imgfiles: List[GridFSAsset] = []


class InteractiveImgfile(BaseModel):
    imgfiles: List[GridFSAsset]
    components: List[int] = []


class InteractiveEvent(BaseModel):
    id: str = Field("000000000000000000000000", alias="_id")
    updated_at: datetime.datetime = datetime.datetime(1970, 1, 1, 0, 0, tzinfo=datetime.timezone.utc)
    created_at: datetime.datetime = datetime.datetime(1970, 1, 1, 0, 0, tzinfo=datetime.timezone.utc)
    project_uid: str
    job_uid: str
    cpumem_mb: Optional[float] = None
    avail_mb: Optional[float] = None
    flags: List[str] = []
    meta: dict = {}
    type: str
    subtype: str = "3dscatter"
    text: str
    datafile: GridFSAsset
    preview_imgfiles: List[InteractiveImgfile] = []
    components: List[int] = []


class TextEvent(BaseModel):
    """
    An event with only text and no additional image or interactive data. May
    have "text", "warning" or "error" type.
    """

    id: str = Field("000000000000000000000000", alias="_id")
    updated_at: datetime.datetime = datetime.datetime(1970, 1, 1, 0, 0, tzinfo=datetime.timezone.utc)
    created_at: datetime.datetime = datetime.datetime(1970, 1, 1, 0, 0, tzinfo=datetime.timezone.utc)
    project_uid: str
    job_uid: str
    cpumem_mb: Optional[float] = None
    avail_mb: Optional[float] = None
    flags: List[str] = []
    meta: dict = {}
    type: Literal["text", "warning", "error"]
    text: str

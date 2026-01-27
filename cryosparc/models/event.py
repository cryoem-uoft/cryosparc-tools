# THIS FILE IS AUTO-GENERATED, DO NOT EDIT DIRECTLY
# SEE dev/api_generate_models.py
import datetime
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field

from .asset import GridFSAsset


class CheckpointEvent(BaseModel):
    """
    An event type indicating a checkpoint in the job's execution.
    """

    id: str = Field("000000000000000000000000", alias="_id")
    """
    """
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
    """
    Project UID that this event belongs to.
    """
    job_uid: str
    """
    Job UID that this event belongs to.
    """
    cpumem_mb: Optional[float] = None
    """
    CPU memory used by the job process at the time of the event, in MB.
    """
    avail_mb: Optional[float] = None
    """
    Available worker system memory at the time of the event, in MB.
    """
    flags: List[str] = []
    """
    List of flags associated with this event.
    """
    meta: Dict[str, Any] = {}
    """
    Metadata associated with this event.
    """
    type: Literal["checkpoint"]
    """
    Checkpoint events always have type 'checkpoint'.
    """


class Event(BaseModel):
    """
    Base class for all event types.
    """

    id: str = Field("000000000000000000000000", alias="_id")
    """
    """
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
    """
    Project UID that this event belongs to.
    """
    job_uid: str
    """
    Job UID that this event belongs to.
    """
    cpumem_mb: Optional[float] = None
    """
    CPU memory used by the job process at the time of the event, in MB.
    """
    avail_mb: Optional[float] = None
    """
    Available worker system memory at the time of the event, in MB.
    """
    flags: List[str] = []
    """
    List of flags associated with this event.
    """
    meta: Dict[str, Any] = {}
    """
    Metadata associated with this event.
    """


class ImageEvent(BaseModel):
    id: str = Field("000000000000000000000000", alias="_id")
    """
    """
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
    """
    Project UID that this event belongs to.
    """
    job_uid: str
    """
    Job UID that this event belongs to.
    """
    cpumem_mb: Optional[float] = None
    """
    CPU memory used by the job process at the time of the event, in MB.
    """
    avail_mb: Optional[float] = None
    """
    Available worker system memory at the time of the event, in MB.
    """
    flags: List[str] = []
    """
    List of flags associated with this event.
    """
    meta: Dict[str, Any] = {}
    """
    Metadata associated with this event.
    """
    type: Literal["image"]
    """
    Image events always have type 'image'.
    """
    text: str
    """
    Text description of the image event.
    """
    imgfiles: List[GridFSAsset] = []
    """
    List of image assets associated with the event.
    """


class InteractiveGridFSAsset(BaseModel):
    """
    Basic information about an uploaded data asset stored in GridFS.
    """

    fileid: str = "000000000000000000000000"
    """
    """
    filename: str
    """
    File name, e.g,. "image.png"
    """
    filetype: str
    """
    File format extension, e.g., "png"
    """


class InteractiveImgfile(BaseModel):
    """
    An image file associated with an interactive event, along with its components.
    """

    imgfiles: List[GridFSAsset]
    """
    List of image assets.
    """
    components: List[int] = []
    """
    List of data component indices which this image file represents.
    """


class InteractiveEvent(BaseModel):
    id: str = Field("000000000000000000000000", alias="_id")
    """
    """
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
    """
    Project UID that this event belongs to.
    """
    job_uid: str
    """
    Job UID that this event belongs to.
    """
    cpumem_mb: Optional[float] = None
    """
    CPU memory used by the job process at the time of the event, in MB.
    """
    avail_mb: Optional[float] = None
    """
    Available worker system memory at the time of the event, in MB.
    """
    flags: List[str] = []
    """
    List of flags associated with this event.
    """
    meta: Dict[str, Any] = {}
    """
    Metadata associated with this event.
    """
    type: Literal["interactive"]
    """
    Interactive events always have type 'interactive'.
    """
    subtype: str = "3dscatter"
    """
    Subtype of interactive event, e.g. '3dscatter'.
    """
    text: str
    """
    Text description of the interactive event.
    """
    datafile: InteractiveGridFSAsset
    """
    Data asset associated with the interactive event.
    """
    preview_imgfiles: List[InteractiveImgfile] = []
    """
    List of preview image files associated with the interactive event.
    """
    components: List[int] = []
    """
    List of data component indices included in this interactive event.
    """


class TextEvent(BaseModel):
    """
    An event with only text and no additional image or interactive data. May
    have "text", "warning" or "error" type.
    """

    id: str = Field("000000000000000000000000", alias="_id")
    """
    """
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
    """
    Project UID that this event belongs to.
    """
    job_uid: str
    """
    Job UID that this event belongs to.
    """
    cpumem_mb: Optional[float] = None
    """
    CPU memory used by the job process at the time of the event, in MB.
    """
    avail_mb: Optional[float] = None
    """
    Available worker system memory at the time of the event, in MB.
    """
    flags: List[str] = []
    """
    List of flags associated with this event.
    """
    meta: Dict[str, Any] = {}
    """
    Metadata associated with this event.
    """
    type: Literal["text", "warning", "error"]
    """
    Type of text event: 'text', 'warning', or 'error'.
    """
    text: str
    """
    Text content of the event.
    """

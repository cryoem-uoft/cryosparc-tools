# THIS FILE IS AUTO-GENERATED, DO NOT EDIT DIRECTLY
# SEE dev/api_generate_models.py
import datetime
from typing import Literal, Optional, Union

from pydantic import BaseModel, Field


class GridFSAsset(BaseModel):
    """
    Information about an uploaded GridFS file.
    """

    fileid: str
    filename: str
    filetype: str


class GridFSFile(BaseModel):
    id: str = Field("000000000000000000000000", alias="_id")
    updated_at: datetime.datetime = datetime.datetime(1970, 1, 1, 0, 0, tzinfo=datetime.timezone.utc)
    created_at: datetime.datetime = datetime.datetime(1970, 1, 1, 0, 0, tzinfo=datetime.timezone.utc)
    filename: str
    contentType: Union[
        Literal["text/plain", "text/csv", "text/html", "application/json", "application/xml", "application/x-troff"],
        Literal["application/pdf", "image/gif", "image/jpeg", "image/png", "image/svg+xml"],
        str,
    ]
    uploadDate: datetime.datetime = datetime.datetime(1970, 1, 1, 0, 0, tzinfo=datetime.timezone.utc)
    length: int
    chunkSize: int
    md5: Optional[str] = None
    project_uid: str
    job_uid: str

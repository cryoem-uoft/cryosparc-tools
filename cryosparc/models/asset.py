# THIS FILE IS AUTO-GENERATED, DO NOT EDIT DIRECTLY
# SEE dev/api_generate_models.py
import datetime
from typing import Literal, Union

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
    filename: str
    """
    File name
    """
    contentType: Union[
        Literal["text/plain", "text/csv", "text/html", "application/json", "application/xml", "application/x-troff"],
        Literal["application/pdf", "image/gif", "image/jpeg", "image/png", "image/svg+xml"],
        Literal["application/octet-stream"],
    ]
    """
    Asset content type, e.g., "image/png"
    """
    uploadDate: datetime.datetime = datetime.datetime(1970, 1, 1, 0, 0, tzinfo=datetime.timezone.utc)
    """
    Saved as updatedAt to the database
    """
    length: int
    """
    Size of file in bytes
    """
    chunkSize: int
    """
    File chunk size in bytes
    """
    project_uid: str
    """
    Associated project UID
    """
    job_uid: str
    """
    Associated job or session UID
    """

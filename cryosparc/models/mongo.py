# THIS FILE IS AUTO-GENERATED, DO NOT EDIT DIRECTLY
# SEE dev/api_generate_models.py
from pydantic import BaseModel


class GridFSAsset(BaseModel):
    """
    Information about an uploaded GridFS file.
    """

    fileid: str
    filename: str
    filetype: str

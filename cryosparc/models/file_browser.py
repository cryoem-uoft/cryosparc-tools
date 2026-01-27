# THIS FILE IS AUTO-GENERATED, DO NOT EDIT DIRECTLY
# SEE dev/api_generate_models.py
from typing import List, Literal, Optional

from pydantic import BaseModel


class BrowseFile(BaseModel):
    """
    Representation of a file or directory in the file browser.
    """

    file_name: str
    """
    Base name of file or directory.
    """
    base_path: str
    """
    Absolute path of the parent directory.
    """
    is_hidden: bool
    """
    """
    path_abs: str
    """
    """
    is_link: bool = False
    """
    """
    mtime: Optional[float] = None
    """
    """
    size: Optional[int] = None
    """
    """
    type: Optional[Literal["dir", "file"]] = None
    """
    """
    link_path: Optional[str] = None
    """
    """
    errmesg: Optional[str] = None
    """
    """


class BrowseFileResponse(BaseModel):
    """
    Response information when using the instance file browser API.
    """

    back_path: str
    """
    """
    files: List[BrowseFile]
    """
    """
    type: str
    """
    """


class FileBrowserPrefixes(BaseModel):
    """
    Configuration for allowed file browser prefixes and default directories.
    """

    allowed_prefixes: Optional[List[str]] = None
    """
    """
    import_dir_default: Optional[str] = None
    """
    """
    project_dir_default: Optional[str] = None
    """
    """

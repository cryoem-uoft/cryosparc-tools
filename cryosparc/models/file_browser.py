# THIS FILE IS AUTO-GENERATED, DO NOT EDIT DIRECTLY
# SEE dev/api_generate_models.py
from typing import Literal

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
    mtime: float | None = None
    """
    """
    size: int | None = None
    """
    """
    type: Literal["dir", "file"] | None = None
    """
    """
    link_path: str | None = None
    """
    """
    errmesg: str | None = None
    """
    """


class BrowseFileResponse(BaseModel):
    """
    Response information when using the instance file browser API.
    """

    back_path: str
    """
    """
    files: list[BrowseFile]
    """
    """
    type: str
    """
    """


class FileBrowserSettings(BaseModel):
    allowed_prefixes: list[str] | None = None
    """
    """
    import_dir_default: str | None = None
    """
    """
    project_dir_default: str | None = None
    """
    """

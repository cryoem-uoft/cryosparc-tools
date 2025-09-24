# THIS FILE IS AUTO-GENERATED, DO NOT EDIT DIRECTLY
# SEE dev/api_generate_models.py
from typing import List, Literal, Optional

from pydantic import BaseModel


class BrowseFile(BaseModel):
    file_name: str
    base_path: str
    is_hidden: bool
    path_abs: str
    is_link: bool = False
    mtime: Optional[float] = None
    size: Optional[int] = None
    type: Optional[Literal["dir", "file"]] = None
    link_path: Optional[str] = None
    errmesg: Optional[str] = None


class BrowseFileResponse(BaseModel):
    back_path: str
    files: List[BrowseFile]
    type: str


class FileBrowserPrefixes(BaseModel):
    allowed_prefixes: Optional[List[str]] = None
    import_dir_default: Optional[str] = None
    project_dir_default: Optional[str] = None

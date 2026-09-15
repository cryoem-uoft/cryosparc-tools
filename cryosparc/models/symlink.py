# THIS FILE IS AUTO-GENERATED, DO NOT EDIT DIRECTLY
# SEE dev/api_generate_models.py
from pydantic import BaseModel


class SymlinkInfo(BaseModel):
    """
    Information about a symlink in a project directory.
    """

    path: str
    """
    Path of the symlink.
    """
    target: str
    """
    Target of the symlink.
    """
    exists: bool
    """
    Whether the target of the symlink exists.
    """

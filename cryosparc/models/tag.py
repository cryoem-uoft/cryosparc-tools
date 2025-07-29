# THIS FILE IS AUTO-GENERATED, DO NOT EDIT DIRECTLY
# SEE dev/api_generate_models.py
import datetime
from typing import Literal, Optional

from pydantic import BaseModel, Field


class TagCounts(BaseModel):
    total: int = 0
    project: int = 0
    workspace: int = 0
    session: int = 0
    job: int = 0


class Tag(BaseModel):
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
    uid: str
    title: str
    type: Literal["general", "project", "workspace", "session", "job"]
    created_by_user_id: str
    colour: Optional[
        Literal[
            "black",
            "gray",
            "red",
            "orange",
            "yellow",
            "green",
            "teal",
            "cyan",
            "sky",
            "blue",
            "indigo",
            "purple",
            "pink",
        ]
    ] = "gray"
    description: Optional[str] = None
    created_by_workflow: Optional[str] = None
    counts: TagCounts = TagCounts()
    uid_num: int

"""
Functions and classes for performing and storing user authentication operations.
"""

from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Dict, Optional
from warnings import warn

from pydantic import AwareDatetime, BaseModel, RootModel, ValidationError, field_validator

from .models.auth import Token
from .platform import user_config_path
from .util import first


@lru_cache(maxsize=1)
def get_default_sessions_config_path():
    return user_config_path() / "cryosparc-tools" / "sessions.json"


class Session(BaseModel):
    """
    A cryosparc-tools CLI session from a successful call to
    ``python -m cryosparc.tools login``.
    """

    token: Token
    expires: AwareDatetime


class UserSessions(RootModel):
    """
    Dictionary of CLI login sessions for a single instance organized by user ID.
    Only one session is allowed per-user.
    """

    root: Dict[str, Session]

    @field_validator("root", mode="after")
    @classmethod
    def remove_expired_sessions(cls, value: Dict[str, Session]):
        now = datetime.now(timezone.utc)
        return {k: v for k, v in value.items() if v.expires > now}


class InstanceSessions(RootModel):
    """
    Dictionary of CLI multiple user sessions for multiple instances,
    organized by instance URL. Stored in ~/.config directory.
    """

    root: Dict[str, UserSessions]

    @classmethod
    def load(cls, path: Optional[Path] = None):
        if not path:
            path = get_default_sessions_config_path()
        try:
            return cls.model_validate_json(path.read_bytes())
        except FileNotFoundError:
            if path != get_default_sessions_config_path():
                warn(f"Sessions file at {path} does not exist; load result is empty")
            return cls({})
        except ValidationError as err:
            warn(str(err))
            warn(f"Sessions stored at {path} are invalid; load result is empty")
            return cls({})

    def save(self, path: Optional[Path] = None):
        if not path:
            path = get_default_sessions_config_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.model_dump_json(indent=4))

    def find(self, url: str, email: Optional[str] = None) -> Optional[Session]:
        """
        Find the first available session for the given instance URL and email.
        If an email is not specified, returns the first existing session
        """
        if url in self.root:
            user_sessions = self.root[url]
            user_email = first(e for e in user_sessions.root if not email or e == email)
            if user_email:
                return user_sessions.root[user_email]

    def insert(self, url: str, email: str, token: Token, expires: AwareDatetime):
        if url not in self.root:
            self.root[url] = UserSessions({})
        self.root[url].root[email] = Session(token=token, expires=expires)

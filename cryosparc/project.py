from pathlib import Path
from typing import Optional
from .command import CommandClient
from .job import Job, CustomJob


class Project:
    def __init__(self, cli: CommandClient, uid: str) -> None:
        self.cli = cli
        self.puid = uid

    def dir(self) -> Path:
        """
        Get the path to the project directory
        """
        pass

    def find_job(self, job_uid: str) -> Optional[Job]:
        try:
            return Job(self.cli, self.puid, job_uid)
        except:
            return None

    def find_custom_job(self, job_uid: str) -> Optional[CustomJob]:
        try:
            return CustomJob(self.cli, self.puid, job_uid)
        except:
            return None

    def create_custom_job(self) -> Optional[CustomJob]:
        pass

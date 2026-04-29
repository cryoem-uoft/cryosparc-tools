"""
Definitions for various error classes raised by cryosparc-tools functions
"""

import json
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from httpx import Response

    from .controllers.job import ExternalJobController, JobController
    from .controllers.project import ProjectController
    from .controllers.workspace import WorkspaceController


class DatasetLoadError(Exception):
    """Exception type raised when a dataset cannot be loaded"""

    pass


class APIError(ValueError):
    """
    Raised by failed request to a CryoSPARC API server.
    """

    code: int
    data: Any = None

    def __init__(
        self,
        reason: str,
        *args: object,
        res: "Response",
        data: Any = None,  # must be JSON-encodable if provided
    ) -> None:
        msg = f"*** [API] ({res.request.method} {res.url}, code {res.status_code}) {reason}"
        super().__init__(msg, *args)
        self.code = res.status_code
        if data is not None:
            self.data = data
        elif res.headers.get("Content-Type") == "application/json":
            self.data = res.json()

    def __str__(self):
        s = super().__str__()
        if self.data:
            s += "\nResponse data:\n"
            s += json.dumps(self.data, indent=4)
        return s


class ProjectError(Exception):
    """Exception type raised when a project operations fails"""

    project: "ProjectController"

    def __init__(self, reason: str, *args: object, project: "ProjectController") -> None:
        msg = f"*** [Project {project.uid}] {reason}"
        super().__init__(msg, *args)
        self.project = project


class WorkspaceError(Exception):
    """Exception type raised when a workspace operation fails"""

    workspace: "WorkspaceController"

    def __init__(self, reason: str, *args: object, workspace: "WorkspaceController") -> None:
        msg = f"*** [Workspace {workspace.project_uid}-{workspace.uid}] {reason}"
        super().__init__(msg, *args)
        self.workspace = workspace


class JobError(Exception):
    """Exception type raised when a job operation fails"""

    job: "JobController"

    def __init__(self, reason: str, *args: object, job: "JobController") -> None:
        type = "External Job" if job.type == "snowflake" else "Job"
        msg = f"*** [{type} {job.project_uid}-{job.uid}] {reason}"
        super().__init__(msg, *args)
        self.job = job


class ExternalJobError(JobError):
    """
    Raised during external job lifecycle failures
    """

    job: "ExternalJobController"

    def __init__(self, reason: str, *args: object, job: "ExternalJobController") -> None:
        super().__init__(reason, *args, job=job)

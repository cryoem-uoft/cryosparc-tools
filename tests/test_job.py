import sys
import pytest
import httpretty
from cryosparc.project import Project
from cryosparc.job import Job


@pytest.fixture
def job(project: Project):
    return project.find_job("J1")


def test_job_subprocess_io(job: Job):
    job.subprocess(
        [sys.executable, "-c", 'import sys; print("hello"); print("error", file=sys.stderr); print("world")']
    )

    request = httpretty.last_request()
    body: dict = request.parsed_body  # type: ignore
    assert body["method"] == "job_send_streamlog"

    # Lines may arrive out of order, either end is okay
    params = body["params"][0]
    opt1 = {"project_uid": "P1", "job_uid": "J1", "message": "error", "error": False}
    opt2 = {"project_uid": "P1", "job_uid": "J1", "message": "world", "error": False}
    assert params == opt1 or params == opt2

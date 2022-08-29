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

    request = httpretty.latest_requests()[-3]  # last two requests are "subprocess completed" log lines
    body = request.parsed_body
    assert body["method"] == "job_send_streamlog"

    # Lines may arrive out of order, either is okay
    params = body["params"]
    opt1 = {"project_uid": "P1", "job_uid": "J1", "message": "error", "error": False}
    opt2 = {"project_uid": "P1", "job_uid": "J1", "message": "world", "error": False}
    assert params == opt1 or params == opt2

import sys

import httpretty
import pytest

from cryosparc.dataset import Dataset
from cryosparc.job import Job
from cryosparc.project import Project


@pytest.fixture
def job(cs, project: Project):
    return project.find_job("J1")


def test_queue(job: Job):
    job.queue()
    queue_request = httpretty.latest_requests()[-3]
    refresh_request = httpretty.latest_requests()[-1]
    assert queue_request.parsed_body["method"] == "enqueue_job"
    assert queue_request.parsed_body["params"] == {
        "project_uid": job.project_uid,
        "job_uid": job.uid,
        "lane": None,
        "user_id": job.cs.user_id,
        "hostname": None,
        "gpus": False,
    }
    assert refresh_request.parsed_body["method"] == "get_job"


def test_queue_worker(job: Job):
    job.queue(lane="workers", hostname="worker1", gpus=[1])
    queue_request = httpretty.latest_requests()[-3]
    refresh_request = httpretty.latest_requests()[-1]
    assert queue_request.parsed_body["method"] == "enqueue_job"
    assert queue_request.parsed_body["params"] == {
        "project_uid": job.project_uid,
        "job_uid": job.uid,
        "lane": "workers",
        "user_id": job.cs.user_id,
        "hostname": "worker1",
        "gpus": [1],
    }
    assert refresh_request.parsed_body["method"] == "get_job"


def test_queue_cluster(job: Job):
    vars = {"var1": 42, "var2": "test"}
    job.queue(lane="cluster", cluster_vars=vars)
    vars_request = httpretty.latest_requests()[-5]
    queue_request = httpretty.latest_requests()[-3]
    refresh_request = httpretty.latest_requests()[-1]
    assert vars_request.parsed_body["method"] == "set_cluster_job_custom_vars"
    assert vars_request.parsed_body["params"] == {
        "project_uid": job.project_uid,
        "job_uid": job.uid,
        "cluster_job_custom_vars": vars,
    }
    assert queue_request.parsed_body["method"] == "enqueue_job"
    assert queue_request.parsed_body["params"] == {
        "project_uid": job.project_uid,
        "job_uid": job.uid,
        "lane": "cluster",
        "user_id": job.cs.user_id,
        "hostname": None,
        "gpus": False,
    }
    assert refresh_request.parsed_body["method"] == "get_job"


def test_load_output_all_slots(job: Job):
    output = job.load_output("particles_class_0")
    assert set(output.prefixes()) == {"location", "blob", "ctf"}


def test_load_output_some_missing_slots(job: Job):
    with pytest.raises(
        ValueError,
        match=(
            "Cannot load output particles_class_0 slot pick_stats because "
            "output does not have an associated dataset file. "
        ),
    ):
        job.load_output("particles_class_0", slots=["blob", "pick_stats"])


def test_load_output_some_slots(job: Job, t20s_particles, t20s_particles_passthrough):
    particles = job.load_output("particles_class_0", slots=["location", "blob", "ctf"])
    assert particles == Dataset.innerjoin_many(t20s_particles, t20s_particles_passthrough)


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

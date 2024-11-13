import sys

import httpretty
import pytest

from cryosparc.dataset import Dataset
from cryosparc.job import ExternalJob, Job
from cryosparc.project import Project
from cryosparc.tools import CryoSPARC

from .conftest import T20S_PARTICLES


@pytest.fixture
def job(cs, project: Project):
    return project.find_job("J1")


@pytest.fixture
def mock_external_job_doc():
    return {
        "_id": "67292e95282b26b45d0e8fee",
        "uid": "J2",
        "uid_num": 2,
        "project_uid": "P1",
        "project_uid_num": 1,
        "type": "snowflake",
        "job_type": "snowflake",
        "title": "Recenter Particles",
        "description": "Enter a description.",
        "status": "building",
        "created_at": "Mon, 04 Nov 2024 20:29:09 GMT",
        "created_by_user_id": "61f0383552d791f286b796ef",
        "parents": [],
        "children": [],
        "input_slot_groups": [],
        "output_result_groups": [],
        "output_results": [],
        "params_base": {},
        "params_spec": {},
        "params_secs": {},
        "workspace_uids": ["W1"],
    }


@pytest.fixture
def external_job(
    mock_jsonrpc_procs_vis,
    mock_jsonrpc_procs_core,
    mock_external_job_doc,
    cs: CryoSPARC,
    project: Project,
):
    mock_jsonrpc_procs_vis["create_external_job"] = "J2"
    mock_jsonrpc_procs_core["get_job"] = mock_external_job_doc
    cs.cli()
    cs.vis()
    return project.create_external_job("W1", title="Recenter Particles")


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


def test_create_external_job(cs: CryoSPARC, external_job: ExternalJob):
    requests = httpretty.latest_requests()
    create_external_job_request = requests[-3]
    create_external_job_body = create_external_job_request.parsed_body
    find_external_job_request = requests[-1]
    find_external_job_body = find_external_job_request.parsed_body

    assert create_external_job_body["method"] == "create_external_job"
    assert create_external_job_body["params"] == {
        "project_uid": "P1",
        "workspace_uid": "W1",
        "user": cs.user_id,
        "title": "Recenter Particles",
        "desc": None,
    }
    assert find_external_job_body["method"] == "get_job"
    assert find_external_job_body["params"] == ["P1", "J2"]


@pytest.fixture
def external_job_output(mock_jsonrpc_procs_vis, mock_external_job_doc, cs: CryoSPARC, external_job: ExternalJob):
    mock_external_job_doc["output_result_groups"] = [
        {
            "uid": "J2-G1",
            "type": "particle",
            "name": "particles",
            "title": "Particles",
            "description": "",
            "contains": [
                {
                    "uid": "J2-R1",
                    "type": "particle.blob",
                    "group_name": "particles",
                    "name": "blob",
                    "passthrough": False,
                },
                {
                    "uid": "J2-R2",
                    "type": "particle.ctf",
                    "group_name": "particles",
                    "name": "ctf",
                    "passthrough": False,
                },
            ],
            "passthrough": False,
        }
    ]
    mock_external_job_doc["output_results"] = [
        {
            "uid": "J2-R1",
            "type": "particle.blob",
            "group_name": "particles",
            "name": "blob",
            "title": "",
            "description": "",
            "min_fields": [["path", "O"], ["idx", "u4"], ["shape", "2u4"], ["psize_A", "f4"], ["sign", "f4"]],
            "versions": [0],
            "metafiles": ["J2/particles.cs"],
            "num_items": [10],
            "passthrough": False,
        },
        {
            "uid": "J2-R2",
            "type": "particle.ctf",
            "group_name": "particles",
            "name": "ctf",
            "title": "",
            "description": "",
            "min_fields": [["type", "O"], ["exp_group_id", "u4"], ["accel_kv", "f4"], ["cs_mm", "f4"]],
            "versions": [0],
            "metafiles": ["J2/particles.cs"],
            "num_items": [10],
            "passthrough": False,
        },
    ]
    mock_jsonrpc_procs_vis["add_external_job_output"] = "particles"
    httpretty.register_uri(
        httpretty.POST,
        "http://localhost:39003/external/projects/P1/jobs/J2/outputs/particles/dataset",
        body='"particles"',
    )

    cs.vis()
    external_job.add_output("particle", name="particles", slots=["blob", "ctf"])
    external_job.save_output("particles", T20S_PARTICLES)
    return T20S_PARTICLES


def test_external_job_output(external_job_output):
    requests = httpretty.latest_requests()
    create_output_request = requests[-3]
    find_external_job_request = requests[-1]
    find_external_job_body = find_external_job_request.parsed_body

    assert len(external_job_output) > 0
    assert create_output_request.url == "http://localhost:39003/external/projects/P1/jobs/J2/outputs/particles/dataset"
    assert find_external_job_body["method"] == "get_job"
    assert find_external_job_body["params"] == ["P1", "J2"]


def test_invalid_external_job_output(external_job):
    with pytest.raises(ValueError, match="Invalid output name"):
        external_job.add_output("particle", name="particles/1", slots=["blob", "ctf"])

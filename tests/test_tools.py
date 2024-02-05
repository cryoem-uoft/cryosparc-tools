import httpretty

from cryosparc.job import Job
from cryosparc.project import Project
from cryosparc.tools import CryoSPARC


def test_create_job_basic(cs: CryoSPARC, project: Project):
    job = cs.create_job(project.uid, "W1", "homo_abinit")
    assert isinstance(job, Job)
    assert job.uid == "J1"

    latest_requests = httpretty.latest_requests()
    create_job_request = latest_requests[-3]
    get_job_request = latest_requests[-1]
    assert create_job_request.parsed_body["method"] == "make_job"
    assert create_job_request.parsed_body["params"] == {
        "job_type": "homo_abinit",
        "project_uid": project.uid,
        "workspace_uid": "W1",
        "user_id": cs.user_id,
        "params": {},
        "input_group_connects": {},
        "title": None,
        "desc": None,
    }
    assert get_job_request.parsed_body["method"] == "get_job"
    assert get_job_request.parsed_body["params"] == ["P1", "J1"]


def test_create_job_connect_params(cs: CryoSPARC, project: Project):
    job = cs.create_job(
        project.uid,
        "W1",
        "homo_abinit",
        connections={"particles": ("J2", "particles_selected")},
        params={"abinit_K": 3},
    )
    assert isinstance(job, Job)
    assert job.uid == "J1"

    latest_requests = httpretty.latest_requests()
    create_job_request = latest_requests[-3]
    get_job_request = latest_requests[-1]

    assert create_job_request.parsed_body["method"] == "make_job"
    assert create_job_request.parsed_body["params"] == {
        "job_type": "homo_abinit",
        "project_uid": project.uid,
        "workspace_uid": "W1",
        "user_id": cs.user_id,
        "params": {"abinit_K": 3},
        "input_group_connects": {"particles": ["J2.particles_selected"]},
        "title": None,
        "desc": None,
    }
    assert get_job_request.parsed_body["method"] == "get_job"
    assert get_job_request.parsed_body["params"] == ["P1", "J1"]

import httpretty
from cryosparc.tools import CryoSPARC
from cryosparc.project import Project
from cryosparc.job import Job


def test_create_job_basic(cs: CryoSPARC, project: Project):
    job = cs.create_job(project.uid, "W1", "homo_abinit")
    assert isinstance(job, Job)
    assert job.uid == "J1"

    latest_requests = httpretty.latest_requests()
    create_job_request = latest_requests[-3]
    get_job_request = latest_requests[-1]
    assert create_job_request.parsed_body["method"] == "create_new_job"
    assert create_job_request.parsed_body["params"] == {
        "job_type": "homo_abinit",
        "project_uid": project.uid,
        "workspace_uid": "W1",
        "created_by_user_id": cs.user_id,
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
    create_job_request = latest_requests[-9]
    get_job_request = latest_requests[-7]
    connect_request = latest_requests[-5]
    set_param_request = latest_requests[-3]
    refresh_request = latest_requests[-1]

    assert create_job_request.parsed_body["method"] == "create_new_job"
    assert create_job_request.parsed_body["params"] == {
        "job_type": "homo_abinit",
        "project_uid": project.uid,
        "workspace_uid": "W1",
        "created_by_user_id": cs.user_id,
        "title": None,
        "desc": None,
    }
    assert get_job_request.parsed_body["method"] == "get_job"
    assert get_job_request.parsed_body["params"] == ["P1", "J1"]
    assert connect_request.parsed_body["method"] == "job_connect_group"
    assert connect_request.parsed_body["params"] == {
        "project_uid": "P1",
        "source_group": "J2.particles_selected",
        "dest_group": "J1.particles",
    }
    assert set_param_request.parsed_body["method"] == "job_set_param"
    assert set_param_request.parsed_body["params"] == {
        "project_uid": "P1",
        "job_uid": "J1",
        "param_name": "abinit_K",
        "param_new_value": 3,
    }
    assert refresh_request.parsed_body["method"] == "get_job"
    assert refresh_request.parsed_body["params"] == ["P1", "J1"]

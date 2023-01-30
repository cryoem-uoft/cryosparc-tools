import httpretty
from cryosparc.project import Project
from cryosparc.tools import CryoSPARC


def test_create_job_basic(cs: CryoSPARC, project: Project):
    job = cs.create_job(project.uid, "W1", "homo_abinit")
    assert job

    request = httpretty.latest_requests()[-1]  # last two requests are "subprocess completed" log lines
    body = request.parsed_body
    assert body["method"] == "create_new_job"
    assert body["params"] == {
        "job_type": "homo_abinit",
        "project_uid": project.uid,
        "workspace_uid": "W1",
        "created_by_user_id": cs.user_id,
        "title": None,
        "desc": None,
    }

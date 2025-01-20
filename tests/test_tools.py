from unittest import mock

from cryosparc.api import APIClient
from cryosparc.controllers.job import Job, JobController
from cryosparc.controllers.project import Project
from cryosparc.models.job_spec import Params
from cryosparc.tools import CryoSPARC


def test_create_job_basic(cs: CryoSPARC, project: Project, mock_new_job: Job):
    assert isinstance(mock_create_endpoint := APIClient.jobs.create, mock.Mock)
    mock_create_endpoint.return_value = mock_new_job

    job = cs.create_job(project.uid, "W1", "homo_abinit")
    assert isinstance(job, JobController)
    assert job.uid == mock_new_job.uid
    assert len(job.model.spec.params.model_dump(exclude_defaults=True, exclude_none=True)) == 0
    mock_create_endpoint.assert_called_once_with(
        project.uid, "W1", type="homo_abinit", title="", description="", params={}
    )


def test_create_job_connect_params(
    cs: CryoSPARC,
    project: Project,
    mock_params: Params,
    mock_new_job_with_params: Job,
    mock_new_job_with_connection: Job,
):
    assert isinstance(mock_create_endpoint := APIClient.jobs.create, mock.Mock)
    assert isinstance(mock_connect_endpoint := APIClient.jobs.connect, mock.Mock)
    mock_create_endpoint.return_value = mock_new_job_with_params
    mock_connect_endpoint.return_value = mock_new_job_with_connection
    job = cs.create_job(
        project.uid,
        "W1",
        "homo_abinit",
        connections={"particles": ("J41", "particles")},
        params=mock_params.model_dump(),
    )
    assert isinstance(job, JobController)
    assert job.uid == mock_new_job_with_connection.uid
    assert job.model.spec.params == mock_params
    assert len(job.model.spec.inputs.root["particles"]) == 1
    mock_create_endpoint.assert_called_once_with(
        project.uid, "W1", type="homo_abinit", title="", description="", params=mock_params.model_dump()
    )
    mock_connect_endpoint.assert_called_once_with(
        project.uid, job.uid, "particles", source_job_uid="J41", source_output_name="particles"
    )

import sys
from unittest import mock

import pytest

from cryosparc.api import APIClient
from cryosparc.controllers.job import ExternalJobController, JobController
from cryosparc.controllers.project import ProjectController
from cryosparc.models.job import Job
from cryosparc.models.job_spec import (
    JobSpec,
    Output,
    OutputResult,
    OutputSlot,
    OutputSpec,
    Params,
    ResourceSpec,
)
from cryosparc.tools import CryoSPARC

from ..conftest import T20S_PARTICLES


@pytest.fixture
def mock_enqueue_endpoint(mock_job: Job):
    assert isinstance(endpoint := APIClient.jobs.enqueue, mock.Mock)
    endpoint.return_value = mock_job.model_copy(update={"status": "queued"})
    return endpoint


def test_queue(job: JobController, mock_enqueue_endpoint: mock.Mock):
    job.queue()
    assert job.model.status == "queued"
    mock_enqueue_endpoint.assert_called_once_with(job.project_uid, job.uid, lane=None, hostname=None, gpus=[])


def test_queue_worker(job: JobController, mock_enqueue_endpoint: mock.Mock):
    job.queue(lane="workers", hostname="worker1", gpus=[1])
    assert job.model.status == "queued"
    mock_enqueue_endpoint.assert_called_once_with(
        job.project_uid, job.uid, lane="workers", hostname="worker1", gpus=[1]
    )


def test_queue_cluster(job: JobController, mock_enqueue_endpoint: mock.Mock):
    assert isinstance(mock_vars_endpoint := APIClient.jobs.set_cluster_custom_vars, mock.Mock)
    vars = {"var1": 42, "var2": "test"}
    job.queue(lane="cluster", cluster_vars=vars)
    assert job.model.status == "queued"
    mock_vars_endpoint.assert_called_once_with(job.project_uid, job.uid, vars)
    mock_enqueue_endpoint.assert_called_once_with(job.project_uid, job.uid, lane="cluster", hostname=None, gpus=[])


def test_load_output_all_slots(job: JobController, t20s_particles, t20s_particles_passthrough):
    assert isinstance(mock_load_output_endpoint := APIClient.jobs.load_output, mock.Mock)
    mock_load_output_endpoint.return_value = t20s_particles.innerjoin(t20s_particles_passthrough)
    particles = job.load_output("particles_class_0")
    assert set(particles.prefixes()) == {"location", "blob", "ctf"}
    mock_load_output_endpoint.assert_called_once_with(
        job.project_uid, job.uid, "particles_class_0", slots="all", version="F"
    )


def test_load_output_some_slots(job: JobController, t20s_particles, t20s_particles_passthrough):
    assert isinstance(mock_load_output_endpoint := APIClient.jobs.load_output, mock.Mock)
    mock_load_output_endpoint.return_value = t20s_particles.innerjoin(t20s_particles_passthrough)
    slots = ["location", "blob", "ctf"]
    particles = job.load_output("particles_class_0", slots=slots)
    assert set(particles.prefixes()) == set(slots)
    mock_load_output_endpoint.assert_called_once_with(
        job.project_uid, job.uid, "particles_class_0", slots=slots, version="F"
    )


def test_job_subprocess_io(job: JobController):
    assert isinstance(mock_log_endpoint := APIClient.jobs.add_event_log, mock.Mock)

    job.subprocess(
        [sys.executable, "-c", 'import sys; print("hello"); print("error", file=sys.stderr); print("world")']
    )

    assert len(mock_log_endpoint.mock_calls) == 7  # includes some prelude/divider calls
    mock_log_endpoint.assert_has_calls(
        [
            mock.call(job.project_uid, job.uid, "hello", type="text"),
            mock.call(job.project_uid, job.uid, "error", type="text"),
            mock.call(job.project_uid, job.uid, "world", type="text"),
        ],
        any_order=True,
    )


def test_create_external_job(cs: CryoSPARC, project: ProjectController, external_job: ExternalJobController):
    assert project.uid == external_job.project_uid
    assert isinstance(mock_create_endpoint := APIClient.jobs.create, mock.Mock)
    mock_create_endpoint.assert_called_once_with(
        project.uid, "W1", type="snowflake", title="Recenter Particles", description=""
    )


@pytest.fixture
def mock_external_job(mock_user, mock_project):
    return Job(
        _id="67292e95282b26b45d0e8fee",
        uid="J43",
        uid_num=43,
        project_uid=mock_project.uid,
        project_uid_num=mock_project.uid_num,
        workspace_uids=["W1"],
        job_dir="J43",
        title="Recenter Particles",
        status="building",
        status_num=5,
        created_by_user_id=mock_user.id,
        spec=JobSpec(
            type="snowflake",
            params=Params(),
            ui_tile_width=1,
            ui_tile_height=1,
            resource_spec=ResourceSpec(),
        ),
        build_errors=[],
    )


@pytest.fixture
def external_job(project: ProjectController, mock_external_job: Job):
    APIClient.jobs.create.return_value = mock_external_job  # type: ignore
    return project.create_external_job("W1", title="Recenter Particles")


@pytest.fixture
def external_job_with_added_output(external_job: ExternalJobController, mock_external_job: Job):
    mock_external_job = mock_external_job.model_copy(deep=True)
    mock_external_job.spec.outputs.root["particles"] = Output(
        type="particle",
        title="Particles",
        results=[
            OutputResult(name="blob", dtype="blob"),
            OutputResult(name="ctf", dtype="ctf"),
        ],
    )
    APIClient.jobs.add_output.return_value = mock_external_job  # type: ignore
    external_job.add_output("particle", name="particles", slots=["blob", "ctf"])
    return external_job


@pytest.fixture
def mock_external_job_with_saved_output(external_job_with_added_output: ExternalJobController, mock_external_job: Job):
    metafile = f"{mock_external_job.uid}/particles.cs"
    mock_external_job = mock_external_job.model_copy(deep=True)
    mock_external_job.spec.outputs.root["particles"] = Output(
        type="particle",
        title="Particles",
        results=[
            OutputResult(name="blob", dtype="blob", versions=[0], metafiles=[metafile], num_items=[10]),
            OutputResult(name="ctf", dtype="ctf", versions=[0], metafiles=[metafile], num_items=[10]),
        ],
    )
    APIClient.jobs.save_output.return_value = mock_external_job  # type: ignore
    external_job_with_added_output.save_output("particles", T20S_PARTICLES)
    return external_job_with_added_output


def test_external_job_output(mock_external_job_with_saved_output: ExternalJobController):
    assert isinstance(mock_add_output_endpoint := APIClient.jobs.add_external_output, mock.Mock)
    assert isinstance(mock_save_output_endpoint := APIClient.jobs.save_output, mock.Mock)
    j = mock_external_job_with_saved_output

    mock_add_output_endpoint.assert_called_once_with(
        j.project_uid,
        j.uid,
        "particles",
        OutputSpec(
            type="particle",
            title="particles",
            slots=[OutputSlot(name="blob", dtype="blob"), OutputSlot(name="ctf", dtype="ctf")],
        ),
    )
    mock_save_output_endpoint.assert_called_once_with(j.project_uid, j.uid, "particles", T20S_PARTICLES, version=0)


def test_invalid_external_job_output(external_job):
    with pytest.raises(ValueError, match="Invalid output name"):
        external_job.add_output("particle", name="particles/1", slots=["blob", "ctf"])

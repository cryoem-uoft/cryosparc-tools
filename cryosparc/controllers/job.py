"""
Defines the Job and External job classes for accessing CryoSPARC jobs.
"""

import re
import traceback
import warnings
from contextlib import contextmanager
from io import BytesIO
from pathlib import PurePath, PurePosixPath
from time import sleep, time
from typing import (
    IO,
    TYPE_CHECKING,
    Any,
    Dict,
    Iterable,
    List,
    Literal,
    Optional,
    Pattern,
    Sequence,
    Tuple,
    Union,
    overload,
)

from ..dataset import DEFAULT_FORMAT, Dataset
from ..errors import ExternalJobError
from ..models.asset import GridFSAsset, GridFSFile
from ..models.job import Job, JobStatus
from ..models.job_spec import Input, InputSpec, Output, OutputSpec, Params
from ..spec import (
    ASSET_CONTENT_TYPES,
    IMAGE_CONTENT_TYPES,
    TEXT_CONTENT_TYPES,
    AssetFormat,
    Datatype,
    ImageFormat,
    LoadableSlots,
    SlotSpec,
    TextFormat,
)
from ..stream import Stream
from ..util import PurePosixPathProperty, first, print_table
from . import Controller, as_input_slot, as_output_slot

if TYPE_CHECKING:
    from numpy.typing import ArrayLike, NDArray

    from ..tools import CryoSPARC


GROUP_NAME_PATTERN = r"^[A-Za-z][0-9A-Za-z_]*$"
"""
Input and output result groups may only contain, letters, numbers and underscores.
"""


class JobController(Controller[Job]):
    """
    Accessor class to a job in CryoSPARC with ability to load inputs and
    outputs, add to job log, download job files. Should be created with
    :py:meth:`cs.find_job() <cryosparc.tools.CryoSPARC.find_job>` or
    :py:meth:`project.find_job() <cryosparc.controllers.project.ProjectController.find_job>`.

    Arguments:
        job (tuple[str, str] | Job): either _(Project UID, Job UID)_ tuple or
            Job model, e.g. ``("P3", "J42")``

    Attributes:
        model (Workspace): All job data from the CryoSPARC database.
            Contents may change over time, use :py:method:`refresh` to update.

    Examples:

        Find an existing job.

        >>> cs = CryoSPARC()
        >>> job = cs.find_job("P3", "J42")
        >>> job.status
        "building"

        Queue a job.

        >>> job.queue("worker_lane")
        >>> job.status
        "queued"

        Create a 3-class ab-initio job connected to existing particles.

        >>> job = cs.create_job("P3", "W1", "homo_abinit"
        ...     connections={"particles": ("J20", "particles_selected")}
        ...     params={"abinit_K": 3}
        ... )
        >>> job.queue()
        >>> job.status
        "queued"
    """

    uid: str
    """
    Job unique ID, e.g., "J42"
    """
    project_uid: str
    """
    Project unique ID, e.g., "P3"
    """

    def __init__(self, cs: "CryoSPARC", job: Union[Tuple[str, str], Job]) -> None:
        self.cs = cs
        if isinstance(job, tuple):
            self.project_uid, self.uid = job
            self.refresh()
        else:
            self.project_uid = job.project_uid
            self.uid = job.uid
            self.model = job

    @property
    def type(self) -> str:
        """Job type key"""
        return self.model.spec.type

    @property
    def title(self) -> str:
        """Job title"""
        return self.model.title

    @property
    def desc(self) -> str:
        """Job description"""
        return self.model.description

    @property
    def status(self) -> JobStatus:
        """Job scheduling status."""
        return self.model.status

    @property
    def dir(self) -> PurePosixPath:
        """Full path to the job directory."""
        return PurePosixPathProperty(self.cs.api.jobs.get_directory(self.project_uid, self.uid))

    @property
    def params(self) -> Params:
        """
        Job parameter values object.

        Example:
            >>> cs = CryoSPARC(...)
            >>> job = cs.find_job("P3", "J42")
            >>> print(job.type)
            "homo_abinit"
            >>> print(job.params.abinit_K)
            3
        """
        return self.model.spec.params

    @property
    def inputs(self) -> Dict[str, Input]:
        return self.model.spec.inputs.root

    @property
    def outputs(self) -> Dict[str, Output]:
        return self.model.spec.outputs.root

    @property
    def full_spec(self):
        """
        The full specification for job inputs, outputs and parameters, as
        defined in the job register.
        """
        spec = first(spec for spec in self.cs.job_register.specs if spec.type == self.type)
        if not spec:
            raise RuntimeError(f"Could not find job specification for type {type}")
        return spec

    def refresh(self):
        """
        Reload this job from the CryoSPARC database.

        Returns:
            JobController: self
        """
        self.model = self.cs.api.jobs.find_one(self.project_uid, self.uid)
        return self

    def queue(
        self,
        lane: Optional[str] = None,
        hostname: Optional[str] = None,
        gpus: List[int] = [],
        cluster_vars: Dict[str, Any] = {},
    ):
        """
        Queue a job to a target lane. Available lanes may be queried with
        `:py:meth:`cs.get_lanes() <cryosparc.tools.CryoSPARC.get_lanes>`.

        Optionally specify a hostname for a node or cluster in the given lane.
        Optionally specify specific GPUs indexes to use for computation.

        Available hostnames for a given lane may be queried with
        `:py:meth:`cs.get_targets() <cryosparc.tools.CryoSPARC.get_targets>`.

        Args:
            lane (str, optional): Configuried compute lane to queue to. Leave
                unspecified to run directly on the master or current
                workstation. Defaults to None.
            hostname (str, optional): Specific hostname in compute lane, if more
                than one is available. Defaults to None.
            gpus (list[int], optional): GPUs to queue to. If specified, must
                have as many GPUs as required in job parameters. Leave
                unspecified to use first available GPU(s). Defaults to [].
            cluster_vars (dict[str, Any], optional): Specify custom cluster
                variables when queuing to a cluster. Keys are variable names.
                Defaults to False.

        Examples:

            Queue a job to lane named "worker":

            >>> cs = CryoSPARC()
            >>> job = cs.find_job("P3", "J42")
            >>> job.status
            "building"
            >>> job.queue("worker")
            >>> job.status
            "queued"
        """
        if cluster_vars:
            self.cs.api.jobs.set_cluster_custom_vars(self.project_uid, self.uid, cluster_vars)
        self.model = self.cs.api.jobs.enqueue(self.project_uid, self.uid, lane=lane, hostname=hostname, gpus=gpus)

    def kill(self):
        """
        Kill this job.
        """
        self.model = self.cs.api.jobs.kill(self.project_uid, self.uid)

    def wait_for_status(self, status: Union[JobStatus, Iterable[JobStatus]], *, timeout: Optional[int] = None) -> str:
        """
        Wait for a job's status to reach the specified value. Must be one of
        the following:

        - 'building'
        - 'queued'
        - 'launched'
        - 'started'
        - 'running'
        - 'waiting'
        - 'completed'
        - 'killed'
        - 'failed'

        Args:
            status (str | set[str]): Specific status or set of statuses to wait
                for. If a set of statuses is specified, waits util job reaches
                any of the specified statuses.
            timeout (int, optional): If specified, wait at most this many
                seconds. Once timeout is reached, returns current status.
                Defaults to None.

        Returns:
            str: current job status
        """
        statuses = {status} if isinstance(status, str) else set(status)
        tic = time()
        while self.refresh().status not in statuses:
            if timeout is not None and time() - tic > timeout:
                break
            sleep(5)
        return self.status

    def wait_for_done(self, *, error_on_incomplete: bool = False, timeout: Optional[int] = None) -> str:
        """
        Wait until a job reaches status "completed", "killed" or "failed".

        Args:
            error_on_incomplete (bool, optional): If True, raises an assertion
                error when job finishes with status other than "completed" or
                timeout is reached. Defaults to False.
            timeout (int, optional): If specified, wait at most this many
                seconds. Once timeout is reached, returns current status or
                fails if ``error_on_incomplete`` is ``True``. Defaults to None.
        """
        status = self.wait_for_status({"completed", "killed", "failed"}, timeout=timeout)
        assert not error_on_incomplete or status == "completed", (
            f"Job {self.project_uid}-{self.uid} did not complete (status {status})"
        )
        return status

    def interact(self, action: str, body: Any = {}, *, timeout: int = 10, refresh: bool = False) -> Any:
        """
        Call an interactive action on a waiting interactive job. The possible
        actions and expected body depends on the job type.

        Args:
            action (str): Interactive endpoint to call.
            body (any): Body parameters for the interactive endpoint. Must be
                JSON-encodable.
            timeout (int, optional): Maximum time to wait for the action to
                complete, in seconds. Defaults to 10.
            refresh (bool, optional): If True, refresh the job document after
                posting. Defaults to False.
        """
        result: Any = self.cs.api.jobs.interactive_post(
            self.project_uid, self.uid, body=body, endpoint=action, timeout=timeout
        )
        if refresh:
            self.refresh()
        return result

    def clear(self):
        """
        Clear this job and reset to building status.
        """
        self.model = self.cs.api.jobs.clear(self.project_uid, self.uid)

    def set_param(self, name: str, value: Any, **kwargs) -> bool:
        """
        Set the given param name on the current job to the given value. Only
        works if the job is in "building" status.

        Args:
            name (str): Param name, as defined in the job document's ``params_base``.
            value (any): Target parameter value.

        Returns:
            bool: False if the job encountered a build error.

        Examples:

            Set the number of GPUs used by a supported job

            >>> cs = CryoSPARC()
            >>> job = cs.find_job("P3", "J42")
            >>> job.set_param("compute_num_gpus", 4)
            True
        """
        if "refresh" in kwargs:
            warnings.warn("refresh argument no longer applies", DeprecationWarning, stacklevel=2)
        self.model = self.cs.api.jobs.set_param(self.project_uid, self.uid, name, value=value)
        return True

    def connect(self, target_input: str, source_job_uid: str, source_output: str, **kwargs) -> bool:
        """
        Connect the given input for this job to an output with given job UID and
        name.

        Args:
            target_input (str): Input name to connect into. Will be created if
                not specified.
            source_job_uid (str): Job UID to connect from, e.g., "J42"
            source_output (str): Job output name to connect from , e.g.,
                "particles"

        Returns:
            bool: False if the job encountered a build error.

        Examples:

            Connect J3 to CTF-corrected micrographs from J2's ``micrographs``
            output.

            >>> cs = CryoSPARC()
            >>> project = cs.find_project("P3")
            >>> job = project.find_job("J3")
            >>> job.connect("input_micrographs", "J2", "micrographs")

        """
        if "refresh" in kwargs:
            warnings.warn("refresh argument no longer applies", DeprecationWarning, stacklevel=2)
        if source_job_uid == self.uid:
            raise ValueError(f"Cannot connect job {self.uid} to itself")
        self.model = self.cs.api.jobs.connect(
            self.project_uid, self.uid, target_input, source_job_uid=source_job_uid, source_output_name=source_output
        )
        return True

    def connect_result(
        self,
        target_input: str,
        connection_idx: int,
        slot: str,
        source_job_uid: str,
        source_output: str,
        source_result: str,
        source_version: Union[int, Literal["F"]] = "F",
    ):
        """
        Connect a low-level input result slot with a result from another job.

        Args:
            target_input (str): Input name to connect into, e.g., "particles"
            connection_idx (int): Connection index to connect into, use 0 for
                the job's first connection on that input, 1 for the second, etc.
            slot (str): Input slot name to connect into, e.g., "location"
            source_job_uid (str): Job UID to connect from, e.g., "J42"
            source_output (str): Job output name to connect from , e.g.,
                "particles_selected"
            source_result (str): Result name to connect from, e.g., "location"

        Returns:
            bool: False if the job encountered a build error.

        Examples:

            Connect J3 to the first connection of J2's ``particles`` input.
            >>> cs = CryoSPARC()
            >>> project = cs.find_project("P3")
            >>> job = project.find_job("J3")
            >>> job.connect_result("particles", 0, "location", "J2", "particles_selected", "location")
        """
        assert source_job_uid != self.uid, f"Cannot connect job {self.uid} to itself"
        self.model = self.cs.api.jobs.connect_result(
            self.project_uid,
            self.uid,
            target_input,
            connection_idx,
            slot,
            source_job_uid=source_job_uid,
            source_output_name=source_output,
            source_result_name=source_result,
            source_result_version=source_version,
        )
        return True

    def disconnect(self, target_input: str, connection_idx: Optional[int] = None, **kwargs):
        """
        Clear the given job input group.

        Args:
            target_input (str): Name of input to disconnect
            connection_idx (int, optional): Connection index to clear.
                Set to 0 to clear the first connection, 1 for the second, etc.
                If unspecified, clears all connections. Defaults to None.
        """
        if "refresh" in kwargs:
            warnings.warn("refresh argument no longer applies", DeprecationWarning, stacklevel=2)

        if connection_idx is None:  # Clear all input connections
            self.model = self.cs.api.jobs.disconnect_all(self.project_uid, self.uid, target_input)
        else:
            self.model = self.cs.api.jobs.disconnect(self.project_uid, self.uid, target_input, connection_idx)

    def disconnect_result(self, target_input: str, connection_idx: int, slot: str):
        """
        Clear the job's given input result slot.

        Args:
            target_input (str): Name of input to disconnect
            connection_idx (int): Connection index to modify. Set to 0 for the
                first connection, 1 for the second, etc.
            slot (str): Input slot name to disconnect, e.g., "location"

        Returns:
            bool: False if the job encountered a build error.
        """
        self.model = self.cs.api.jobs.disconnect_result(self.project_uid, self.uid, target_input, connection_idx, slot)
        return True

    def load_input(self, name: str, slots: LoadableSlots = "all"):
        """
        Load the dataset connected to the job's input with the given name.

        Args:
            name (str): Input to load
            slots (Literal["default", "passthrough", "all"] | list[str], optional):
                List of specific slots to load, such as ``movie_blob`` or
                ``locations``, or all slots if not specified (including
                passthrough). May also specify as keyword. Defaults to
                "all".

        Raises:
            TypeError: If the job doesn't have the given input or the dataset
                cannot be loaded.

        Returns:
            Dataset: Loaded dataset
        """
        return self.cs.api.jobs.load_input(self.project_uid, self.uid, name, slots=slots)

    def load_output(self, name: str, slots: LoadableSlots = "all", version: Union[int, Literal["F"]] = "F"):
        """
        Load the dataset for the job's output with the given name.

        Args:
            name (str): Output to load
            slots (Literal["default", "passthrough", "all"] | list[str], optional):
                List of specific slots to load, such as ``movie_blob`` or
                ``locations``, or all slots if not specified (including
                passthrough). May also specify as keyword. Defaults to
                "all".
            version (int | Literal["F"], optional): Specific output version to
                load. Use this to load the output at different stages of
                processing. Leave unspecified to load final verion. Defaults to
                "F"

        Raises:
            TypeError: If job does not have any results for the given output

        Returns:
            Dataset: Loaded dataset
        """
        return self.cs.api.jobs.load_output(self.project_uid, self.uid, name, slots=slots, version=version)

    def log(self, text: str, level: Literal["text", "warning", "error"] = "text"):
        """
        Append to a job's event log.

        Args:
            text (str): Text to log
            level (str, optional): Log level ("text", "warning" or "error").
                Defaults to "text".

        Returns:
            str: Created log event ID
        """
        event = self.cs.api.jobs.add_event_log(self.project_uid, self.uid, text, type=level)
        return event.id

    def log_checkpoint(self, meta: dict = {}):
        """
        Append a checkpoint to the job's event log.

        Args:
            meta (dict, optional): Additional meta information. Defaults to {}.

        Returns:
            str: Created checkpoint event ID
        """
        event = self.cs.api.jobs.add_checkpoint(self.project_uid, self.uid, meta)
        return event.id

    def log_plot(
        self,
        figure: Union[str, PurePath, IO[bytes], Any],
        text: str,
        formats: Iterable[ImageFormat] = ["png", "pdf"],
        raw_data: Union[str, bytes, None] = None,
        raw_data_file: Union[str, PurePath, IO[bytes], None] = None,
        raw_data_format: Optional[TextFormat] = None,
        flags: List[str] = ["plots"],
        savefig_kw: dict = dict(bbox_inches="tight", pad_inches=0),
    ):
        """
        Add a log line with the given figure.

        ``figure`` must be one of the following

        - Path to an existing image file in PNG, JPEG, GIF, SVG or PDF format
        - A file handle-like object with the binary data of an image
        - A matplotlib plot

        If a matplotlib figure is specified, Uploads the plots in ``png`` and
        ``pdf`` formats. Override the ``formats`` argument with
        ``formats=['<format1>', '<format2>', ...]`` to save in different image
        formats.

        If a text-version of the given plot is available (e.g., in ``csv``
        format), specify ``raw_data`` with the full contents or
        ``raw_data_file`` with a path or binary file handle pointing to the
        contents. Assumes file format from extension or ``raw_data_format``.
        Defaults to ``"txt"`` if cannot be determined.

        Args:
            figure (str | Path | IO | Figure): Image file path, file handle or
                matplotlib figure instance
            text (str): Associated description for given figure
            formats (list[ImageFormat], optional): Image formats to save plot
                into. If a ``figure`` is a file handle, specify
                ``formats=['<format>']``, where ``<format>`` is a valid image
                extension such as ``png`` or ``pdf``. Assumes ``png`` if not
                specified. Defaults to ["png", "pdf"].
            raw_data (str | bytes, optional): Raw text data for associated plot,
                generally in CSV, XML or JSON format. Cannot be specified with
                ``raw_data_file``. Defaults to None.
            raw_data_file (str | Path | IO, optional): Path to raw text data.
                Cannot be specified with ``raw_data``. Defaults to None.
            raw_data_format (TextFormat, optional): Format for raw text data.
                Defaults to None.
            flags (list[str], optional): Flags to use for UI rendering.
                Generally should not be specified. Defaults to ["plots"].
            savefig_kw (dict, optional): If a matplotlib figure is specified
                optionally specify keyword arguments for the ``savefig`` method.
                Defaults to dict(bbox_inches="tight", pad_inches=0).

        Returns:
            str: Created log event ID
        """
        imgfiles = self.upload_plot(
            figure,
            name=text,
            formats=formats,
            raw_data=raw_data,
            raw_data_file=raw_data_file,
            raw_data_format=raw_data_format,
            savefig_kw=savefig_kw,
        )
        event = self.cs.api.jobs.add_image_log(self.project_uid, self.uid, imgfiles, text=text, flags=flags)
        return event.id

    def list_files(self, prefix: Union[str, PurePosixPath] = "", recursive: bool = False) -> List[str]:
        """
        Get a list of files inside the job directory.

        Args:
            prefix (str | Path, optional): Subdirectory inside job to list.
                Defaults to "".
            recursive (bool, optional): If True, lists files recursively.
                Defaults to False.

        Returns:
            list[str]: List of file paths relative to the job directory.
        """
        root = PurePosixPath(self.uid)
        files = self.cs.list_files(self.project_uid, prefix=root / prefix, recursive=recursive)
        return [str(PurePosixPath(f).relative_to(root)) for f in files]  # Strip leading "J#/"

    def download(self, path: Union[str, PurePosixPath]):
        """
        Initiate a download request for a file inside the job's directory. Use
        to get files from a remote CryoSPARC instance where the job directory
        is not available on the client file system.

        Args:
            path (str | Path): Name or path of file in job directory.

        Yields:
            HTTPResponse: Use a context manager to read the file from the
                request body.

        Examples:

            Download a job's metadata

            >>> cs = CryoSPARC()
            >>> job = cs.find_job("P3", "J42")
            >>> with job.download("job.json") as res:
            >>>     job_data = json.loads(res.read())

        """
        path = PurePosixPath(self.uid) / path
        return self.cs.download(self.project_uid, path)

    def download_file(self, path: Union[str, PurePosixPath], target: Union[str, PurePath, IO[bytes]] = ""):
        """
        Download file from job directory to the given target path or writeable
        file handle.

        Args:
            path (str | Path): Name or path of file in job directory.
            target (str | Path | IO): Local file path, directory path or
                writeable file handle to write response data. If not specified,
                downloads to current working directory with same file name.
                Defaults to "".

        Returns:
            Path | IO: resulting target path or file handle.
        """
        path = PurePosixPath(self.uid) / path
        return self.cs.download_file(self.project_uid, path, target)

    def download_dataset(self, path: Union[str, PurePosixPath]):
        """
        Download a .cs dataset file from the given path in the job
        directory.

        Args:
            path (str | Path): Name or path of .cs file in job directory.

        Returns:
            Dataset: Loaded dataset instance
        """
        path = PurePosixPath(self.uid) / path
        return self.cs.download_dataset(self.project_uid, path)

    def download_mrc(self, path: Union[str, PurePosixPath]):
        """
        Download a .mrc file from the given relative path in the job directory.

        Args:
            path (str | Path): Name or path of .mrc file in job directory.

        Returns:
            tuple[Header, NDArray]: MRC file header and data as a numpy array
        """
        path = PurePosixPath(self.uid) / path
        return self.cs.download_mrc(self.project_uid, path)

    def list_assets(self) -> List[GridFSFile]:
        """
        Get a list of files available in the database for this job. Returns a
        list with details about the assets. Each entry is a dict with a ``_id``
        key which may be used to download the file with the ``download_asset``
        method.

        Returns:
            list[GridFSFile]: Asset details
        """
        return self.cs.list_assets(self.project_uid, self.uid)

    def download_asset(self, fileid: str, target: Union[str, PurePath, IO[bytes]]):
        """
        Download a job asset from the database with the given ID. Note that the
        file does not necessary have to belong to the current job.

        Args:
            fileid (str): GridFS file object ID
            target (str | Path | IO): Local file path or writeable file handle
                to write response data.

        Returns:
            str | Path | IO: resulting target path or file handle.

        """
        return self.cs.download_asset(fileid, target)

    def upload(
        self,
        target_path: Union[str, PurePosixPath],
        source: Union[str, bytes, PurePath, IO],
        *,
        overwrite: bool = False,
    ):
        """
        Upload the given file to the job directory at the given path. Fails if
        target already exists.

        Args:
            target_path (str | Path): Name or path of file to write in job
                directory.
            source (str | bytes | Path | IO): Local path or file handle to
                upload. May also specified as raw bytes.
            overwrite (bool, optional): If True, overwrite existing files.
                Defaults to False.
        """
        target_path = PurePosixPath(self.uid) / target_path
        return self.cs.upload(self.project_uid, target_path, source, overwrite=overwrite)

    def upload_asset(
        self,
        file: Union[str, PurePath, IO[bytes]],
        filename: Optional[str] = None,
        format: Optional[AssetFormat] = None,
    ) -> GridFSAsset:
        """
        Upload an image or text file to the current job. Specify either an image
        (PNG, JPG, GIF, PDF, SVG), text file (TXT, CSV, JSON, XML) or a binary
        IO object with data in one of those formats.

        If a binary IO object is specified, either a filename or file format
        must be specified.

        Unlike the ``upload`` method which saves files to the job directory,
        this method saves images to the database and exposes them for use in the
        job log.

        If specifying arbitrary binary I/O, specify either a filename or a file
        format.

        Args:
            file (str | Path | IO): Source asset file path or handle.
            filename (str, optional): Filename of asset. If ``file`` is a handle
                specify one of ``filename`` or ``format``. Defaults to None.
            format (AssetFormat, optional): Format of filename. If ``file`` is
                a handle, specify one of ``filename`` or ``format``. Defaults to
                None.

        Raises:
            ValueError: If incorrect arguments specified

        Returns:
            EventLogAsset: Dictionary including details about uploaded asset.
        """
        ext = None
        if format:
            ext = format
        elif filename:
            ext = filename.split(".")[-1].lower()
        elif isinstance(file, (str, PurePath)):
            file = PurePath(file)
            filename = file.name
            ext = file.suffix[1:].lower()
        else:
            raise ValueError("Must specify filename or format when saving binary asset")
        if ext not in ASSET_CONTENT_TYPES:
            raise ValueError(f"Invalid asset format {ext}")
        return self.cs.api.assets.upload(self.project_uid, self.uid, Stream.load(file), filename=filename, format=ext)

    def upload_plot(
        self,
        figure: Union[str, PurePath, IO[bytes], Any],
        name: Optional[str] = None,
        formats: Iterable[ImageFormat] = ["png", "pdf"],
        raw_data: Union[str, bytes, None] = None,
        raw_data_file: Union[str, PurePath, IO[bytes], None] = None,
        raw_data_format: Optional[TextFormat] = None,
        savefig_kw: dict = dict(bbox_inches="tight", pad_inches=0),
    ) -> List[GridFSAsset]:
        """
        Upload the given figure. Returns a list of the created asset objects.
        Avoid using directly; use ``log_plot`` instead. See ``log_plot``
        additional details.

        Args:
            figure (str | Path | IO | Figure): Image file path, file handle or
                matplotlib figure instance
            name (str): Associated name for given figure
            formats (list[ImageFormat], optional): Image formats to save plot
                into. If a ``figure`` is a file handle, specify
                ``formats=['<format>']``, where ``<format>`` is a valid image
                extension such as ``png`` or ``pdf``. Assumes ``png`` if not
                specified. Defaults to ["png", "pdf"].
            raw_data (str | bytes, optional): Raw text data for associated plot,
                generally in CSV, XML or JSON format. Cannot be specified with
                ``raw_data_file``. Defaults to None.
            raw_data_file (str | Path | IO, optional): Path to raw text data.
                Cannot be specified with ``raw_data``. Defaults to None.
            raw_data_format (TextFormat, optional): Format for raw text data.
                Defaults to None.
            savefig_kw (dict, optional): If a matplotlib figure is specified
                optionally specify keyword arguments for the ``savefig`` method.
                Defaults to dict(bbox_inches="tight", pad_inches=0).

        Raises:
            ValueError: If incorrect argument specified

        Returns:
            list[EventLogAsset]: Details about created uploaded job assets
        """
        figdata = []
        basename = name or "figure"
        if hasattr(figure, "savefig"):  # matplotlib plot
            for fmt in formats:
                if fmt not in IMAGE_CONTENT_TYPES:
                    raise ValueError(f"Invalid figure format {fmt}")
                filename = f"{basename}.{fmt}"
                data = BytesIO()
                figure.savefig(data, format=fmt, **savefig_kw)  # type: ignore
                data.seek(0)
                figdata.append((data, filename, fmt))
        elif isinstance(figure, (str, PurePath)):  # file path; assume format from filename
            path = PurePath(figure)
            basename = path.stem
            fmt = path.suffix[1:].lower()
            if fmt not in IMAGE_CONTENT_TYPES:
                raise ValueError(f"Invalid figure format {fmt}")
            filename = f"{name or path.stem}.{fmt}"
            figdata.append((figure, filename, fmt))
        else:  # Binary IO
            fmt = first(iter(formats))
            if fmt not in IMAGE_CONTENT_TYPES:
                raise ValueError(f"Invalid or unspecified figure format {fmt}")
            filename = f"{basename}.{fmt}"
            figdata.append((figure, filename, fmt))

        raw_data_filename = f"{name}.{raw_data_format or 'txt'}" if name else None
        if raw_data and raw_data_file:
            raise ValueError("Ambiguous invocation; specify only one of raw_data or raw_data_file")
        elif isinstance(raw_data, str):
            raw_data_file = BytesIO(raw_data.encode())
        elif isinstance(raw_data, bytes):
            raw_data_file = BytesIO(raw_data)
        elif isinstance(raw_data_file, (str, PurePath)):
            raw_data_path = PurePath(raw_data_file)
            raw_data_filename = raw_data_path.name
            ext = raw_data_format or raw_data_filename.split(".")[-1]
            if ext not in TEXT_CONTENT_TYPES:
                raise ValueError(f"Invalid raw data filename {raw_data_file}")
            raw_data_format = ext

        assets = [self.upload_asset(data, filename, fmt) for data, filename, fmt in figdata]
        if raw_data_file:
            asset = self.upload_asset(raw_data_file, raw_data_filename, raw_data_format or "txt")
            assets.append(asset)

        return assets

    def upload_dataset(
        self,
        target_path: Union[str, PurePosixPath],
        dset: Dataset,
        *,
        format: int = DEFAULT_FORMAT,
        overwrite: bool = False,
    ):
        """
        Upload a dataset as a CS file into the job directory. Fails if target
        already exists.

        Args:
            target_path (str | Path): Name or path of dataset to save in the job
                directory. Should have a ``.cs`` extension.
            dset (Dataset): Dataset to save.
            format (int): Format to save in from ``cryosparc.dataset.*_FORMAT``,
                defaults to NUMPY_FORMAT)
            overwrite (bool, optional): If True, overwrite existing files.
                Defaults to False.
        """
        target_path = PurePosixPath(self.uid) / target_path
        return self.cs.upload_dataset(self.project_uid, target_path, dset, format=format, overwrite=overwrite)

    def upload_mrc(
        self,
        target_path: Union[str, PurePosixPath],
        data: "NDArray",
        psize: float,
        *,
        overwrite: bool = False,
    ):
        """
        Upload a numpy 2D or 3D array to the job directory as an MRC file. Fails
        if target already exists.

        Args:
            target_path (str | Path): Name or path of MRC file to save in the
                job directory. Should have a ``.mrc`` extension.
            data (NDArray): Numpy array with MRC file data.
            psize (float): Pixel size to include in MRC header.
            overwrite (bool, optional): If True, overwrite existing files.
                Defaults to False.
        """
        target_path = PurePosixPath(self.uid) / target_path
        return self.cs.upload_mrc(self.project_uid, target_path, data, psize, overwrite=overwrite)

    def mkdir(
        self,
        target_path: Union[str, PurePosixPath],
        parents: bool = False,
        exist_ok: bool = False,
    ):
        """
        Create a folder in the given job.

        Args:
            target_path (str | Path): Name or path of folder to create inside
                the job directory.
            parents (bool, optional): If True, any missing parents are created
                as needed. Defaults to False.
            exist_ok (bool, optional): If True, does not raise an error for
                existing directories. Still raises if the target path is not a
                directory. Defaults to False.
        """
        self.cs.mkdir(
            project_uid=self.project_uid,
            target_path=PurePosixPath(self.uid) / target_path,
            parents=parents,
            exist_ok=exist_ok,
        )

    def cp(self, source_path: Union[str, PurePosixPath], target_path: Union[str, PurePosixPath] = ""):
        """
        Copy a file or folder into the job directory.

        Args:
            source_path (str | Path): Relative or absolute path of source file
                or folder to copy. If relative, assumed to be within the job
                directory.
            target_path (str | Path, optional): Name or path in the job
                directory to copy into. If not specified, uses the same file
                name as the source. Defaults to "".
        """
        self.cs.cp(
            project_uid=self.project_uid,
            source_path=PurePosixPath(self.uid) / source_path,
            target_path=PurePosixPath(self.uid) / target_path,
        )

    def symlink(self, source_path: Union[str, PurePosixPath], target_path: Union[str, PurePosixPath] = ""):
        """
        Create a symbolic link in job's directory.

        Args:
            source_path (str | Path): Relative or absolute path of source file
                or folder to create a link to. If relative, assumed to be within
                the job directory.
            target_path (str | Path): Name or path of new symlink in the job
                directory. If not specified, creates link with the same file
                name as the source. Defaults to "".
        """
        self.cs.symlink(
            project_uid=self.project_uid,
            source_path=PurePosixPath(self.uid) / source_path,
            target_path=PurePosixPath(self.uid) / target_path,
        )

    def subprocess(
        self,
        args: Union[str, list],
        mute: bool = False,
        checkpoint: bool = False,
        checkpoint_line_pattern: Union[str, Pattern[str], None] = None,
        **kwargs,
    ):
        """
        Launch a subprocess and write its text-based output and error to the job
        log.

        Args:
            args (str | list): Process arguments to run
            mute (bool, optional): If True, does not also forward process output
                to standard output. Defaults to False.
            checkpoint (bool, optional): If True, creates a checkpoint in the
                job event log just before process output begins. Defaults to
                False.
            checkpoint_line_pattern (str | Pattern[str], optional): Regular
                expression to match checkpoint lines for processes with a lot of
                output. If a process outputs a line that matches this pattern, a
                checkpoint is created in the event log before this line is
                forwarded. Defaults to None.
            **kwargs: Additional keyword arguments for ``subprocess.Popen``.

        Raises:
            TypeError: For invalid arguments
            RuntimeError: If process exists with non-zero status code
        """
        import re
        import subprocess

        pttrn = None
        if checkpoint_line_pattern and isinstance(checkpoint_line_pattern, str):
            pttrn = re.compile(checkpoint_line_pattern)
        elif isinstance(checkpoint_line_pattern, re.Pattern):
            pttrn = checkpoint_line_pattern
        elif checkpoint_line_pattern:
            raise TypeError(f"Invalid checkpoint_line_pattern argument type: {type(checkpoint_line_pattern)}")

        args = args if isinstance(args, str) else list(map(str, args))
        with subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, **kwargs) as proc:
            assert proc.stdout, f"Subprocess {args} has no standard output"
            if checkpoint:
                self.log_checkpoint()

            self.log("─────── Forwarding subprocess output for the following command ───────")
            self.log(str(args))
            self.log("──────────────────────────────────────────────────────────────────────")

            for line in proc.stdout:
                line = line.rstrip()
                if pttrn and pttrn.match(line):
                    self.log_checkpoint()
                self.log(line)
                if not mute:
                    print(line)

            while proc.poll() is None:
                sleep(1)

            if proc.returncode != 0:
                msg = f"Subprocess {args} exited with status {proc.returncode}"
                self.log(msg, level="error")
                raise RuntimeError(msg)

            self.log("─────────────────────── Subprocess complete. ─────────────────────────")

    def print_param_spec(self):
        """
        Print a table of parameter keys, their title, type and default to
        standard output:

        Examples:

            >>> cs = CryoSPARC()
            >>> job = cs.find_job("P3", "J42")
            >>> job.doc['type']
            'extract_micrographs_multi'
            >>> job.print_param_spec()
            Param                       | Title                 | Type    | Default
            =======================================================================
            box_size_pix                | Extraction box size   | number  | 256
            bin_size_pix                | Fourier crop box size | number  | None
            compute_num_gpus            | Number of GPUs        | number  | 1
            ...

        """
        headings = ["Param", "Title", "Type", "Default"]
        rows = []
        for key, details in self.full_spec.params.items():
            if details.hidden is True:
                continue
            type, format = (details.type, details.format)
            if details.anyOf:
                type, format = (details.anyOf[0].type, details.anyOf[0].format)
            rows.append([key, details.title or key, format or type or "any", repr(details.default)])
        print_table(headings, rows)

    def print_input_spec(self):
        """
        Print a table of input keys, their title, type, connection requirements
        and details about their low-level required slots.

        The "Required?" heading also shows the number of outputs that must be
        connected to the input for this job to run.

        Examples:

            >>> cs = CryoSPARC()
            >>> job = cs.find_job("P3", "J42")
            >>> job.doc['type']
            'extract_micrographs_multi'
            >>> job.print_output_spec()
            Input       | Title       | Type     | Required? | Input Slots     | Slot Types      | Slot Required?
            =====================================================================================================
            micrographs | Micrographs | exposure | ✓ (1+)    | micrograph_blob | micrograph_blob | ✓
                        |             |          |           | mscope_params   | mscope_params   | ✓
                        |             |          |           | background_blob | stat_blob       | ✕
                        |             |          |           | ctf             | ctf             | ✕
            particles   | Particles   | particle | ✕ (0+)    | location        | location        | ✓
                        |             |          |           | alignments2D    | alignments2D    | ✕
                        |             |          |           | alignments3D    | alignments3D    | ✕
        """
        headings = ["Input", "Title", "Type", "Required?", "Input Slots", "Slot Types", "Slot Required?"]
        rows = []
        for key, input in self.model.spec.inputs.root.items():
            name, title, type = key, input.title, input.type
            required = f"✓ ({input.count_min}" if input.count_min else "✕ (0"
            if input.count_max in (0, "inf"):
                required += "+)"  # unlimited connections
            elif input.count_min == input.count_max:
                required += ")"
            else:
                required += f"-{input.count_max})"
            for slot in input.slots:
                slot = as_input_slot(slot)
                rows.append([name, title, type, required, slot.name, slot.dtype, "✓" if slot.required else "✕"])
                name, title, type, required = ("",) * 4  # only show group info on first iter
        print_table(headings, rows)

    def print_output_spec(self):
        """
        Print a table of output keys, their title, type and details about their
        low-level results.

        Examples:

            >>> cs = CryoSPARC()
            >>> job = cs.find_job("P3", "J42")
            >>> job.doc['type']
            'extract_micrographs_multi'
            >>> job.print_output_spec()
            Output                 | Title       | Type     | Result Slots           | Result Types    | Passthrough?
            =========================================================================================================
            micrographs            | Micrographs | exposure | micrograph_blob        | micrograph_blob | ✕
                                   |             |          | micrograph_blob_non_dw | micrograph_blob | ✓
                                   |             |          | background_blob        | stat_blob       | ✓
                                   |             |          | ctf                    | ctf             | ✓
                                   |             |          | ctf_stats              | ctf_stats       | ✓
                                   |             |          | mscope_params          | mscope_params   | ✓
            particles              | Particles   | particle | blob                   | blob            | ✕
                                   |             |          | ctf                    | ctf             | ✕
        """
        headings = ["Output", "Title", "Type", "Result Slots", "Result Types", "Passthrough?"]
        rows = []
        for key, output in self.model.spec.outputs.root.items():
            name, title, type = key, output.title, output.type
            for result in output.results:
                rows.append([name, title, type, result.name, result.dtype, "✓" if result.passthrough else "✕"])
                name, title, type = "", "", ""  # only these print once per group
        print_table(headings, rows)


class ExternalJobController(JobController):
    """
    Mutable custom output job with customizeble input slots and output results.
    Use External jobs to save data save cryo-EM data generated by a software
    package outside of CryoSPARC.

    Created external jobs may be connected to any other CryoSPARC job result as
    an input. Its outputs must be created manually and may be configured to
    passthrough inherited input fields, just as with regular CryoSPARC jobs.

    Create a new External Job with :py:meth:`project.create_external_job() <cryosparc.project.ProjectController.create_external_job>`.
    or :py:meth:`workspace.create_external_job() <cryosparc.workspace.WorkspaceController.create_external_job>`.
    ``ExternalJobController`` is a subclass of :py:class:`JobController`
    and inherits all its methods and attributes.

    Examples:

        Import multiple exposure groups into a single job

        >>> from cryosparc.tools import CryoSPARC
        >>> cs = CryoSPARC()
        >>> project = cs.find_project("P3")
        >>> job = project.create_external_job("W3", title="Import Image Sets")
        >>> for i in range(3):
        ...     dset = job.add_output(
        ...         type="exposure",
        ...         name=f"images_{i}",
        ...         slots=["movie_blob", "mscope_params", "gain_ref_blob"],
        ...         alloc=10  # allocate a dataset for this output with 10 rows
        ...     )
        ...     dset['movie_blob/path'] = ...  # populate dataset
        ...     job.save_output(output_name, dset)
    """

    def __init__(self, cs: "CryoSPARC", job: Union[Tuple[str, str], Job]) -> None:
        super().__init__(cs, job)
        if self.model.spec.type != "snowflake":
            raise TypeError(f"Job {self.model.project_uid}-{self.model.uid} is not an external job")

    def add_input(
        self,
        type: Datatype,
        name: Optional[str] = None,
        min: int = 0,
        max: Union[int, Literal["inf"]] = "inf",
        slots: Sequence[SlotSpec] = [],
        title: Optional[str] = None,
        desc: Optional[str] = None,
    ):
        """
        Add an input slot to the current job. May be connected to zero or more
        outputs from other jobs (depending on the min and max values).

        Args:
            type (Datatype): cryo-EM data type for this output, e.g., "particle"
            name (str, optional): Output name key, e.g., "picked_particles".
                Same as ``type`` if not specified. Defaults to None.
            min (int, optional): Minimum number of required input connections.
                Defaults to 0.
            max (int | Literal["inf"], optional): Maximum number of input
                connections. Specify ``"inf"`` for unlimited connections.
                Defaults to "inf".
            slots (list[SlotSpec], optional): List of slots that should
                be connected to this input, such as ``"location"`` or  ``"blob"``.
                When connecting the input, if the source job output is missing
                these slots, the external job cannot start or accept outputs.
                Defaults to [].
            title (str, optional): Human-readable title for this input. Defaults
                to name.
            desc (str, optional): Human-readable description for this input.
                Defaults to None.

        Raises:
            CommandError: General CryoSPARC network access error such as
                timeout, URL or HTTP
            InvalidSlotsError: slots argument is invalid

        Returns:
            str: name of created input

        Examples:

            Create an external job that accepts micrographs as input:

            >>> cs = CryoSPARC()
            >>> project = cs.find_project("P3")
            >>> job = project.create_external_job("W1", title="Custom Picker")
            >>> job.uid
            "J3"
            >>> job.add_input(
            ...     type="exposure",
            ...     name="input_micrographs",
            ...     min=1,
            ...     slots=["micrograph_blob", "ctf"],
            ...     title="Input micrographs for picking
            ... )
            "input_micrographs"
        """
        if name and not re.fullmatch(GROUP_NAME_PATTERN, name):
            raise ValueError(
                f'Invalid input name "{name}"; may only contain letters, numbers and underscores, '
                "and must start with a letter"
            )
        if not slots:
            raise ValueError("Must must provide slots=[...] argument with at least one slot")
        if any(isinstance(s, dict) and "prefix" in s for s in slots):
            warnings.warn("'prefix' slot key is deprecated. Use 'name' instead.", DeprecationWarning, stacklevel=2)
        if not name:
            name = type
        if not title:
            title = name
        self.model = self.cs.api.jobs.add_external_input(
            self.project_uid,
            self.uid,
            name,
            InputSpec(
                type=type,
                title=title,
                description=desc or "",
                slots=[as_input_slot(slot) for slot in slots],
                count_min=min,
                count_max=max,
            ),
        )
        return name

    # fmt: off
    @overload
    def add_output(self, type: Datatype, name: Optional[str] = ..., slots: Sequence[SlotSpec] = ..., passthrough: Optional[str] = ..., title: Optional[str] = ...) -> str: ...
    @overload
    def add_output(self, type: Datatype, name: Optional[str] = ..., slots: Sequence[SlotSpec] = ..., passthrough: Optional[str] = ..., title: Optional[str] = ..., *, alloc: Union[int, Dataset]) -> Dataset: ...
    # fmt: on
    def add_output(
        self,
        type: Datatype,
        name: Optional[str] = None,
        slots: Sequence[SlotSpec] = [],
        passthrough: Optional[str] = None,
        title: Optional[str] = None,
        *,
        alloc: Union[int, Dataset, None] = None,
    ) -> Union[str, Dataset]:
        """
        Add an output slot to the current job. Optionally returns the
        corresponding empty dataset if ``alloc`` is specified.

        Args:
            type (Datatype): cryo-EM datatype for this output, e.g., "particle"
            name (str, optional): Output name key, e.g., "selected_particles".
                Same as ``type`` if not specified. Defaults to None.
            slots (list[SlotSpec], optional): List of slot expected to be
                created for this output, such as ``location`` or ``blob``. Do
                not specify any slots that were passed through from an input
                unless those slots are modified in the output. Defaults to [].
            passthrough (str, optional): Indicates that this output inherits
                slots from an existing input with the specified name. The input
                must first be added with ``add_input()``. Defaults to False.
            title (str, optional): Human-readable title for this input. Defaults
                to None.
            alloc (int | Dataset, optional): If specified, pre-allocate and
                return a dataset with the requested slots. Specify an integer
                to allocate a specific number of rows. Specify a Dataset from
                which to inherit unique row IDs (useful when adding passthrough
                outputs). Defaults to None.

        Raises:
            CommandError: General CryoSPARC network access error such as
                timeout, URL or HTTP
            InvalidSlotsError: slots argument is invalid

        Returns:
            str | Dataset: Name of the created output. If ``alloc`` is
                specified as an integer, instead returns blank dataset with the
                given size and random UIDs. If ``alloc`` is specified as a
                Dataset, returns blank dataset with the same UIDs.

        Examples:

            Create and allocate an output for new particle picks

            >>> cs = CryoSPARC()
            >>> project = cs.find_project("P3")
            >>> job = project.find_external_job("J3")
            >>> particles_dset = job.add_output(
            ...     type="particle",
            ...     name="picked_particles",
            ...     slots=["location", "pick_stats"],
            ...     alloc=10000
            ... )

            Create an inheritied output for input micrographs

            >>> job.add_output(
            ...     type="exposures",
            ...     name="picked_micrographs",
            ...     passthrough="input_micrographs",
            ...     title="Passthrough picked micrographs"
            ... )
            "picked_micrographs"

            Create an output with multiple slots of the same type

            >>> job.add_output(
            ...     type="particle",
            ...     name="particle_alignments",
            ...     slots=[
            ...         {"name": "alignments_class_0", "dtype": "alignments3D", "required": True},
            ...         {"name": "alignments_class_1", "dtype": "alignments3D", "required": True},
            ...         {"name": "alignments_class_2", "dtype": "alignments3D", "required": True},
            ...     ]
            ... )
            "particle_alignments"
        """
        if name and not re.fullmatch(GROUP_NAME_PATTERN, name):
            raise ValueError(
                f'Invalid output name "{name}"; may only contain letters, numbers and underscores, '
                "and must start with a letter"
            )
        if not slots:
            raise ValueError("Must must provide slots=[...] argument with at least one slot")
        if any(isinstance(s, dict) and "prefix" in s for s in slots):
            warnings.warn("'prefix' slot key is deprecated. Use 'name' instead.", DeprecationWarning, stacklevel=2)
        if not name:
            name = type
        if not title:
            title = name
        self.model = self.cs.api.jobs.add_external_output(
            self.project_uid,
            self.uid,
            name,
            OutputSpec(type=type, title=title, slots=[as_output_slot(slot) for slot in slots], passthrough=passthrough),
        )
        return name if alloc is None else self.alloc_output(name, alloc)

    def connect(
        self,
        target_input: str,
        source_job_uid: str,
        source_output: str,
        *,
        slots: Sequence[SlotSpec] = [],
        title: Optional[str] = None,
        desc: Optional[str] = None,
        **kwargs,
    ) -> bool:
        """
        Connect the given input for this job to an output with given job UID and
        name. If this input does not exist, it will be added with the given
        slots.

        Args:
            target_input (str): Input name to connect into. Will be created if
                does not already exist.
            source_job_uid (str): Job UID to connect from, e.g., "J42"
            source_output (str): Job output name to connect from , e.g.,
                "particles"
            slots (list[SlotSpec], optional): List of input slots (e.g.,
                "particle" or "blob") to explicitly required for the created
                input. If the given source job is missing these slots, the
                external job cannot start or accept outputs. Defaults to [].
            title (str, optional): Human readable title for created input.
                Defaults to target input name.
            desc (str, optional): Human readable description for created input.
                Defaults to "".

        Raises:
            CommandError: General CryoSPARC network access error such as
                timeout, URL or HTTP
            InvalidSlotsError: slots argument is invalid

        Examples:

            Connect J3 to CTF-corrected micrographs from J2's ``micrographs``
            output.

            >>> cs = CryoSPARC()
            >>> project = cs.find_project("P3")
            >>> job = project.find_external_job("J3")
            >>> job.connect("input_micrographs", "J2", "micrographs")

        """
        if "refresh" in kwargs:
            warnings.warn("refresh argument no longer applies", DeprecationWarning, stacklevel=2)
        if source_job_uid == self.uid:
            raise ValueError(f"Cannot connect job {self.uid} to itself")
        source_job = self.cs.api.jobs.find_one(self.project_uid, source_job_uid)
        if source_output not in source_job.spec.outputs.root:
            raise ValueError(f"Source job {source_job_uid} does not have output {source_output}")
        output = source_job.spec.outputs.root[source_output]
        if target_input not in self.model.spec.inputs.root:
            if any(isinstance(s, dict) and "prefix" in s for s in slots):
                warnings.warn("'prefix' slot key is deprecated. Use 'name' instead.", DeprecationWarning, stacklevel=2)
                # convert to prevent from warning again
                slots = [as_input_slot(slot) for slot in slots]  # type: ignore
            self.add_input(output.type, target_input, min=1, slots=slots, title=title, desc=desc)
        return super().connect(target_input, source_job_uid, source_output)

    def alloc_output(
        self, name: str, alloc: Union[int, "ArrayLike", Dataset] = 0, *, dtype_params: Dict[str, Any] = {}
    ) -> Dataset:
        """
        Allocate an empty dataset for the given output with the given name.
        Initialize with the given number of empty rows. The result may be
        used with ``save_output`` with the same output name.

        Args:
            name (str): Name of job output to allocate
            size (int | ArrayLike | Dataset, optional): Specify as one of the
                following: (A) integer to allocate a specific number of rows,
                (B) a numpy array of numbers to use for UIDs in the allocated
                dataset or (C) a dataset from which to inherit unique row IDs
                (useful  for allocating passthrough outputs). Defaults to 0.
            dtype_params (dict, optional): Data type parameters when allocating
                results with dynamic column sizes such as ``particle`` ->
                ``alignments3D_multi``. Defaults to {}.

        Returns:
            Dataset: Empty dataset with the given number of rows

        Examples:

            Allocate a dataset of size 10,000 for an output for new particle
            picks

            >>> cs = CryoSPARC()
            >>> project = cs.find_project("P3")
            >>> job = project.find_external_job("J3")
            >>> job.alloc_output("picked_particles", 10000)
            Dataset([  # 10000 items, 11 fields
                ("uid": [...]),
                ("location/micrograph_path", ["", ...]),
                ...
            ])

            Allocate a dataset from an existing input passthrough dataset

            >>> input_micrographs = job.load_input("input_micrographs")
            >>> job.alloc_output("picked_micrographs", input_micrographs)
            Dataset([  # same "uid" field as input_micrographs
                ("uid": [...]),
            ])

        """
        expected_fields = self.cs.api.jobs.get_output_fields(self.project_uid, self.uid, name, dtype_params)
        if isinstance(alloc, int):
            return Dataset.allocate(alloc, expected_fields)
        elif isinstance(alloc, Dataset):
            return Dataset({"uid": alloc["uid"]}).add_fields(expected_fields)
        else:
            return Dataset({"uid": alloc}).add_fields(expected_fields)

    def save_output(self, name: str, dataset: Dataset, *, version: int = 0, **kwargs):
        """
        Save output dataset to external job.

        Args:
            name (str): Name of output on this job.
            dataset (Dataset): Value of output with only required fields.
            version (int, optional): Version number, when saving multiple
                intermediate iterations. Only the last saved version is kept.
                Defaults to 0.

        Examples:

            Save a previously-allocated output.

            >>> cs = CryoSPARC()
            >>> project = cs.find_project("P3")
            >>> job = project.find_external_job("J3")
            >>> particles = job.alloc_output("picked_particles", 10000)
            >>> job.save_output("picked_particles", particles)

        """
        if "refresh" in kwargs:
            warnings.warn("refresh argument no longer applies", DeprecationWarning, stacklevel=2)
        self.model = self.cs.api.jobs.save_output(self.project_uid, self.uid, name, dataset, version=version)

    def start(self, status: Literal["running", "waiting"] = "waiting"):
        """
        Set job status to "running" or "waiting"

        Args:
            status (str, optional): "running" or "waiting". Defaults to "waiting".
        """
        self.model = self.cs.api.jobs.mark_running(self.project_uid, self.uid, status=status)

    def stop(self, error: str = ""):
        """
        Set job status to "completed" or "failed" if there was an error.

        Args:
            error (str, optional): Error message, will add to event log and set
                job to status to failed if specified. Defaults to "".
        """
        if isinstance(error, bool):  # allowed bool in previous version
            warnings.warn("error should be specified as a string", DeprecationWarning, stacklevel=2)
            error = "An error occurred" if error else ""
        if error:
            self.model = self.cs.api.jobs.mark_failed(self.project_uid, self.uid, error=error)
        else:
            self.model = self.cs.api.jobs.mark_completed(self.project_uid, self.uid)

    @contextmanager
    def run(self):
        """
        Start a job within a context manager and stop the job when the context
        ends.

        Yields:
            ExternalJob: self.

        Examples:

            Job will be marked as "failed" if the contents of the block throw an
            exception

            >>> with job.run():
            ...     job.save_output(...)

        """
        error = ""
        try:
            self.start("running")
            yield self
        except Exception:
            error = traceback.format_exc()
            raise
        finally:
            self.stop(error=error)

    def queue(
        self,
        lane: Optional[str] = None,
        hostname: Optional[str] = None,
        gpus: Sequence[int] = [],
        cluster_vars: Dict[str, Any] = {},
    ):
        raise ExternalJobError(
            "Cannot queue an external job; use `job.start()`/`job.stop()` or `with job.run()` instead"
        )

    def kill(self):
        raise ExternalJobError("Cannot kill an external job; use `job.stop()` instead")

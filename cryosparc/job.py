"""
Defines the Job and External job classes for accessing CryoSPARC jobs.
"""
from contextlib import contextmanager
from io import BytesIO
import json
from pathlib import PurePath, PurePosixPath
from time import sleep, time
from typing import IO, TYPE_CHECKING, Any, Iterable, List, Optional, Pattern, Union, overload
from typing_extensions import Literal

from .command import make_json_request, make_request
from .dataset import Dataset, DEFAULT_FORMAT
from .spec import (
    ASSET_CONTENT_TYPES,
    IMAGE_CONTENT_TYPES,
    TEXT_CONTENT_TYPES,
    AssetDetails,
    AssetFormat,
    MongoController,
    ImageFormat,
    JobStatus,
    TextFormat,
    EventLogAsset,
    Datatype,
    Datafield,
    JobDocument,
)
from .util import bopen, first


if TYPE_CHECKING:
    from numpy.typing import NDArray, ArrayLike
    from .tools import CryoSPARC


class Job(MongoController[JobDocument]):
    """
    Accessor class to a job in CryoSPARC with ability to load inputs and
    outputs, add to job log, download job files. Should be instantiated
    through `CryoSPARC.find_job`_ or `Project.find_job`_.

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

    .. _CryoSPARC.find_job:
        tools.html#cryosparc.tools.CryoSPARC.find_job

    .. _Project.find_job:
        project.html#cryosparc.project.Project.find_job
    """

    def __init__(self, cs: "CryoSPARC", project_uid: str, uid: str) -> None:
        self.cs = cs
        self.project_uid = project_uid
        self.uid = uid

    @property
    def status(self) -> JobStatus:
        """
        JobStatus: scheduling status.
        """
        return self.doc["status"]

    def refresh(self):
        """
        Reload this job from the CryoSPARC database.

        Returns:
            Job: self
        """
        self._doc = self.cs.cli.get_job(self.project_uid, self.uid)  # type: ignore
        return self

    def dir(self) -> PurePosixPath:
        """
        Get the path to the job directory.

        Returns:
            Path: job directory Pure Path instance
        """
        return PurePosixPath(self.cs.cli.get_job_dir_abs(self.project_uid, self.uid))  # type: ignore

    def queue(self, lane: str, hostname: Optional[str] = None, gpus: List[int] = []):
        """
        Queue a job to a target lane. Available lanes may be queried from
        `CryoSPARC.get_lanes`_.

        Optionally specify a hostname in that lane and/or specific GPUs to use
        for computation. Available hostnames for a given lane may be queried
        with `CryoSPARC.get_targets`_.

        Args:
            lane (str): Configuried compute lane to queue to.
            hostname (str, optional): Specific hostname in compute lane, if more
                than one is available. Defaults to None.
            gpus (list[int], optional): GPUs to queue to. If specified, must
                have as many GPUs as required in job parameters. Leave
                unspecified to use first available GPU(s). Defaults to [].

        Examples:

            Queue a job to lane named "worker":

            >>> cs = CryoSPARC()
            >>> job = cs.find_job("P3", "J42")
            >>> job.status
            "building"
            >>> job.queue("worker")
            >>> job.status
            "queued"

        .. _CryoSPARC.get_lanes:
            tools.html#cryosparc.tools.CryoSPARC.get_lanes
        .. _CryoSPARC.get_targets:
            tools.html#cryosparc.tools.CryoSPARC.get_targets
        """
        self.cs.cli.enqueue_job(  # type: ignore
            project_uid=self.project_uid,
            job_uid=self.uid,
            lane=lane,
            user_id=self.cs.user_id,
            hostname=hostname,
            gpus=gpus if gpus else False,
        )
        self.refresh()

    def kill(self):
        """
        Kill this job.
        """
        self.cs.cli.kill_job(  # type: ignore
            project_uid=self.project_uid, job_uid=self.uid, killed_by_user_id=self.cs.user_id
        )
        self.refresh()

    def wait_for_status(self, status: Union[JobStatus, Iterable[JobStatus]], timeout: Optional[int] = None) -> str:
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
        self.refresh()
        while self.status not in statuses:
            if timeout is not None and time() - tic > timeout:
                break
            sleep(5)
            self.refresh()
        return self.status

    def wait_for_done(self, error_on_incomplete: bool = False, timeout: Optional[int] = None) -> str:
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
        assert (
            not error_on_incomplete or status == "completed"
        ), f"Job {self.project_uid}-{self.uid} did not complete (status {status})"
        return status

    def clear(self):
        """
        Clear this job and reset to building status.
        """
        self.cs.cli.clear_job(self.project_uid, self.uid)  # type: ignore
        self.refresh()

    def set_param(self, name: str, value: Any, refresh: bool = True) -> bool:
        """
        Set the given param name on the current job to the given value. Only
        works if the job is in "building" status.

        Args:
            name (str): Param name, as defined in the job document's ``params_base``.
            value (any): Target parameter value.
            refresh (bool, optional): Auto-refresh job document after
                connecting. Defaults to True.

        Returns:
            bool: False if the job encountered a build error.

        Examples:

            Set the number of GPUs used by a supported job

            >>> cs = CryoSPARC()
            >>> job = cs.find_job("P3", "J42")
            >>> job.set_param("compute_num_gpus", 4)
            True
        """
        result: bool = self.cs.cli.job_set_param(  # type: ignore
            project_uid=self.project_uid, job_uid=self.uid, param_name=name, param_new_value=value
        )
        if refresh:
            self.refresh()
        return result

    def connect(self, target_input: str, source_job_uid: str, source_output: str, refresh: bool = True) -> bool:
        """
        Connect the given input for this job to an output with given job UID and
        name.

        Args:
            target_input (str): Input name to connect into. Will be created if
                not specified.
            source_job_uid (str): Job UID to connect from, e.g., "J42"
            source_output (str): Job output name to connect from , e.g.,
                "particles"
            refresh (bool, optional): Auto-refresh job document after
                connecting. Defaults to True.

        Returns:
            bool: False if the job encountered a build error.

        Examples:

            Connect J3 to CTF-corrected micrographs from J2's ``micrographs``
            output.

            >>> cs = CryoSPARC()
            >>> project = cs.find_project("P3")
            >>> job = project.find_job("J3")
            >>> job.connect("J2", "micrographs", "input_micrographs")

        """
        assert source_job_uid != self.uid, f"Cannot connect job {self.uid} to itself"
        result: bool = self.cs.cli.job_connect_group(  # type: ignore
            project_uid=self.project_uid,
            source_group=f"{source_job_uid}.{source_output}",
            dest_group=f"{self.uid}.{target_input}",
        )
        if refresh:
            self.refresh()
        return result

    def disconnect(self, target_input: str, connection_idx: Optional[int] = None, refresh: bool = True):
        """
        Clear the given job input group.

        Args:
            target_input (str): Name of input to disconnect
            connection_idx (int, optional): Connection index to clear.
                Set to 0 to clear the first connection, 1 for the second, etc.
                If unspecified, clears all connections. Defaults to None.
            refresh (bool, optional): Auto-refresh job document after
                connecting. Defaults to True.
        """
        if connection_idx is None:
            # Clear all input connections
            input_group = first(group for group in self.doc["input_slot_groups"] if group["name"] == target_input)
            if not input_group:
                raise ValueError(f"Unknown input group {target_input} for job {self.project_uid}-{self.uid}")
            for _ in input_group["connections"]:
                self.cs.cli.job_connected_group_clear(  # type: ignore
                    project_uid=self.project_uid,
                    dest_group=f"{self.uid}.{target_input}",
                    connect_idx=0,
                )
        else:
            self.cs.cli.job_connected_group_clear(  # type: ignore
                project_uid=self.project_uid,
                dest_group=f"{self.uid}.{target_input}",
                connect_idx=connection_idx,
            )

        if refresh:
            self.refresh()

    def load_input(self, name: str, slots: Iterable[str] = []):
        """
        Load the dataset connected to the job's input with the given name.

        Args:
            name (str): Input to load
            fields (list[str], optional): List of specific slots to load, such
                as ``movie_blob`` or ``locations``, or all slots if not
                specified. Defaults to [].

        Raises:
            TypeError: If the job doesn't have the given input or the dataset
                cannot be loaded.

        Returns:
            Dataset: Loaded dataset
        """
        job = self.doc
        group = first(s for s in job["input_slot_groups"] if s["name"] == name)
        if not group:
            raise TypeError(f"Job {self.project_uid}-{self.uid} does not have an input {name}")

        data = {"project_uid": self.project_uid, "job_uid": self.uid, "input_name": name, "slots": list(slots)}
        with make_json_request(self.cs.vis, "/load_job_input", data=data) as response:
            mime = response.headers.get("Content-Type")
            if mime != "application/x-cryosparc-dataset":
                raise TypeError(f"Unable to load dataset for job {self.project_uid}-{self.uid} input {name}")
            return Dataset.load(response)

    def load_output(self, name: str, slots: Iterable[str] = [], version: Union[int, Literal["F"]] = "F"):
        """
        Load the dataset for the job's output with the given name.

        Args:
            name (str): Output to load
            slots (list[str], optional): List of specific slots to load,
                such as ``movie_blob`` or ``locations``, or all slots if
                not specified (including passthrough). Defaults to [].
            version (int | Literal["F"], optional): Specific output version to
                load. Use this to load the output at different stages of
                processing. Leave unspecified to load final verion. Defaults to
                "F"

        Raises:
            TypeError: If job does not have any results for the given output

        Returns:
            Dataset: Loaded dataset
        """
        job = self.doc
        slots = set(slots)
        version = -1 if version == "F" else version
        results = [
            result
            for result in job["output_results"]
            if result["group_name"] == name and (not slots or result["name"] in slots)
        ]
        if not results:
            raise TypeError(f"Job {self.project_uid}-{self.uid} does not have any results for output {name}")

        metafiles = set(r["metafiles"][0 if r["passthrough"] else version] for r in results)
        datasets = [self.cs.download_dataset(self.project_uid, f) for f in metafiles]
        return Dataset.innerjoin(*datasets)

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
        return self.cs.cli.job_send_streamlog(  # type: ignore
            project_uid=self.project_uid, job_uid=self.uid, message=text, error=level != "text"
        )

    def log_checkpoint(self, meta: dict = {}):
        """
        Append a checkpoint to the job's event log.

        Args:
            meta (dict, optional): Additional meta information. Defaults to {}.

        Returns:
            str: Created checkpoint event ID
        """
        return self.cs.cli.job_checkpoint_streamlog(  # type: ignore
            project_uid=self.project_uid, job_uid=self.uid, meta=meta
        )

    def log_plot(
        self,
        figure: Union[str, PurePath, IO[bytes], Any],
        text: str,
        formats: Iterable[ImageFormat] = ["png", "pdf"],
        raw_data: Union[str, bytes, Literal[None]] = None,
        raw_data_file: Union[str, PurePath, IO[bytes], Literal[None]] = None,
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

        return self.cs.cli.job_send_streamlog(  # type: ignore
            project_uid=self.project_uid, job_uid=self.uid, message=text, flags=flags, imgfiles=imgfiles
        )

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

    def download(self, path_rel: Union[str, PurePosixPath]):
        """
        Initiate a download request for a file inside the job's diretory

        Args:
            path_rel (str | Path): Relative path to file in job directory.

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
        path_rel = PurePosixPath(self.uid) / path_rel
        return self.cs.download(self.project_uid, path_rel)

    def download_file(self, path_rel: Union[str, PurePosixPath], target: Union[str, PurePath, IO[bytes]]):
        """
        Download file from job directory to the given target path or writeable
        file handle.

        Args:
            path_rel (str | Path): Relative path to file in job directory.
            target (str | Path | IO): Local file path, directory path or writeable
                file handle to write response data.

        Returns:
            Path | IO: resulting target path or file handle.
        """
        path_rel = PurePosixPath(self.uid) / path_rel
        return self.cs.download_file(self.project_uid, path_rel, target)

    def download_dataset(self, path_rel: Union[str, PurePosixPath]):
        """
        Download a .cs dataset file from the given relative path in the job
        directory.

        Args:
            path_rel (str | Path): Relative path to .cs file in job directory.

        Returns:
            Dataset: Loaded dataset instance
        """
        path_rel = PurePosixPath(self.uid) / path_rel
        return self.cs.download_dataset(self.project_uid, path_rel)

    def download_mrc(self, path_rel: Union[str, PurePosixPath]):
        """
        Download a .mrc file from the given relative path in the job directory.

        Args:
            path (str | Path): Relative path to .mrc file in job directory.

        Returns:
            tuple[Header, NDArray]: MRC file header and data as a numpy array
        """
        path_rel = PurePosixPath(self.uid) / path_rel
        return self.cs.download_mrc(self.project_uid, path_rel)

    def list_assets(self) -> List[AssetDetails]:
        """
        Get a list of files available in the database for this job. Returns a
        list with details about the assets. Each entry is a dict with a ``_id``
        key which may be used to download the file with the ``download_asset``
        method.

        Returns:
            list[AssetDetails]: Asset details
        """
        return self.cs.vis.list_job_files(project_uid=self.project_uid, job_uid=self.uid)  # type: ignore

    def download_asset(self, fileid: str, target: Union[str, PurePath, IO[bytes]]):
        """
        Download a job asset from the database with the given ID. Note that the
        file does not necessary have to belong to the current job.

        Args:
            fileid (str): GridFS file object ID
            target (str | Path | IO): Local file path, directory path or
                writeable file handle to write response data.

        Returns:
            Path | IO: resulting target path or file handle.

        """
        return self.cs.download_asset(fileid, target)

    def upload(self, target_path_rel: Union[str, PurePosixPath], source: Union[str, bytes, PurePath, IO]):
        """
        Upload the given file to the job directory at the given path.

        Args:
            target_path_rel (str | Path): Relative target path in job directory
            source (str | bytes | Path | IO): Local path or file handle to
                upload. May also specified as raw bytes.
        """
        target_path_rel = PurePosixPath(self.uid) / target_path_rel
        return self.cs.upload(self.project_uid, target_path_rel, source)

    def upload_asset(
        self,
        file: Union[str, PurePath, IO[bytes]],
        filename: Optional[str] = None,
        format: Optional[AssetFormat] = None,
    ) -> EventLogAsset:
        """
        Upload an image or text file to the current job. Specify either an image
        (PNG, JPG, GIF, PDF, SVG), text file (TXT, CSV, JSON, XML) or a binary
        IO object with data in one of those formats.

        If a binary IO object is specified, either a filename or mimetype must
        be specified.

        Unlike the ``upload`` method which saves files to the job directory,
        this method saves images to the database and exposes them for use in the
        job log.

        If specifying arbitrary binary I/O, specify either a filename or a file
        format.

        Args:
            file (str | Path | IO): Source asset file path or handle
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
        if format:
            assert format in ASSET_CONTENT_TYPES, f"Invalid asset format {format}"
        elif filename:
            ext = filename.split(".")[-1]
            assert ext in ASSET_CONTENT_TYPES, f"Invalid asset format {ext}"
            format = ext
        elif isinstance(file, (str, PurePath)):
            file = PurePath(file)
            filename = file.name
            ext = filename.split(".")[-1]
            assert ext in ASSET_CONTENT_TYPES, f"Invalid asset format {ext}"
            format = ext
        else:
            raise ValueError("Must specify filename or format when saving binary asset handle")

        with bopen(file) as f:
            url = f"/projects/{self.project_uid}/jobs/{self.uid}/files"
            query = {"format": format}
            if filename:
                query["filename"] = filename

            with make_request(self.cs.vis, url=url, query=query, data=f) as res:
                assert res.status >= 200 and res.status < 300, (
                    f"Could not upload project {self.project_uid} asset {file}.\n"
                    f"Response from CryoSPARC: {res.read().decode()}"
                )
                return json.loads(res.read())

    def upload_plot(
        self,
        figure: Union[str, PurePath, IO[bytes], Any],
        name: Optional[str] = None,
        formats: Iterable[ImageFormat] = ["png", "pdf"],
        raw_data: Union[str, bytes, Literal[None]] = None,
        raw_data_file: Union[str, PurePath, IO[bytes], Literal[None]] = None,
        raw_data_format: Optional[TextFormat] = None,
        savefig_kw: dict = dict(bbox_inches="tight", pad_inches=0),
    ) -> List[EventLogAsset]:
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
                assert fmt in IMAGE_CONTENT_TYPES, f"Invalid figure format {fmt}"
                filename = f"{basename}.{fmt}"
                data = BytesIO()
                figure.savefig(data, format=fmt, **savefig_kw)  # type: ignore
                data.seek(0)
                figdata.append((data, filename, fmt))
        elif isinstance(figure, (str, PurePath)):  # file path; assume format from filename
            figure = PurePath(figure)
            basename = figure.stem
            fmt = str(figure).split(".")[-1]
            assert fmt in IMAGE_CONTENT_TYPES, f"Invalid figure format {fmt}"
            filename = f"{name or figure.stem}.{fmt}"
            figdata.append((figure, filename, fmt))
        else:  # Binary IO
            fmt = first(iter(formats))
            assert fmt in IMAGE_CONTENT_TYPES, f"Invalid or unspecified figure format {fmt}"
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
            assert ext in TEXT_CONTENT_TYPES, f"Invalid raw data filename {raw_data_file}"
            raw_data_format = ext

        assets = []
        for data, filename, fmt in figdata:
            asset = self.upload_asset(data, filename=filename, format=fmt)
            assets.append(asset)

        if raw_data_file:
            raw_data_format = raw_data_format or "txt"
            asset = self.upload_asset(raw_data_file, filename=raw_data_filename, format=raw_data_format)
            assets.append(asset)

        return assets

    def upload_dataset(self, target_path_rel: Union[str, PurePosixPath], dset: Dataset, format: int = DEFAULT_FORMAT):
        """
        Upload a dataset as a CS file into the job directory.

        Args:
            target_path_rel (str | Path): relative path to save dataset in job
                directory. Should have a ``.cs`` extension.
            dset (Dataset): dataset to save.
            format (int): format to save in from ``cryosparc.dataset.*_FORMAT``,
                defaults to NUMPY_FORMAT)

        """
        target_path_rel = PurePosixPath(self.uid) / target_path_rel
        return self.cs.upload_dataset(self.project_uid, target_path_rel, dset, format=format)

    def upload_mrc(self, target_path_rel: Union[str, PurePosixPath], data: "NDArray", psize: float):
        """
        Upload a numpy 2D or 3D array to the job directory as an MRC file.

        Args:
            target_path_rel (str | Path): relative path to save array in job
                directory. Should have ``.mrc`` extension.
            data (NDArray): Numpy array with MRC file data.
            psize (float): Pixel size to include in MRC header.
        """
        target_path_rel = PurePosixPath(self.uid) / target_path_rel
        return self.cs.upload_mrc(self.project_uid, target_path_rel, data, psize)

    def mkdir(
        self,
        target_path_rel: Union[str, PurePosixPath],
        parents: bool = False,
        exist_ok: bool = False,
    ):
        """
        Create a directory in the given job.

        Args:
            target_path_rel (str | Path): Relative path to create inside project
                directory.
            parents (bool, optional): If True, any missing parents are created
                as needed. Defaults to False.
            exist_ok (bool, optional): If True, does not raise an error for
                existing directories. Still raises if the target path is not a
                directory. Defaults to False.
        """
        self.cs.mkdir(
            project_uid=self.project_uid,
            target_path_rel=PurePosixPath(self.uid) / target_path_rel,
            parents=parents,
            exist_ok=exist_ok,
        )

    def cp(self, source_path_rel: Union[str, PurePosixPath], target_path_rel: Union[str, PurePosixPath]):
        """
        Copy a file or folder within a project to another location within that
        same project. Note that argument order is reversed from
        equivalent ``cp`` command.

        Args:
            source_path_rel (str | Path): Relative path in project of source
                file or folder to copy.
            target_path_rel (str | Path): Relative path in project to copy to.
        """
        self.cs.cp(
            project_uid=self.project_uid,
            source_path_rel=PurePosixPath(self.uid) / source_path_rel,
            target_path_rel=PurePosixPath(self.uid) / target_path_rel,
        )

    def symlink(self, source_path_rel: Union[str, PurePosixPath], target_path_rel: Union[str, PurePosixPath]):
        """
        Create a symbolic link in the given project. May only create links for
        files within the project. Note that argument order is reversed from
        ``ln -s``.

        Args:
            project_uid (str): Target project UID, e.g., "P3".
            source_path_rel (str | Path): Relative path in project to file from
                which to create symlink.
            target_path_rel (str | Path): Relative path in project to new
                symlink.
        """
        self.cs.symlink(
            project_uid=self.project_uid,
            source_path_rel=PurePosixPath(self.uid) / source_path_rel,
            target_path_rel=PurePosixPath(self.uid) / target_path_rel,
        )

    def subprocess(
        self,
        args: Union[str, list],
        mute: bool = False,
        checkpoint: bool = False,
        checkpoint_line_pattern: Union[str, Pattern[str], Literal[None]] = None,
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
        import subprocess
        import re

        pttrn = None
        if checkpoint_line_pattern and isinstance(checkpoint_line_pattern, str):
            pttrn = re.compile(checkpoint_line_pattern)
        elif isinstance(checkpoint_line_pattern, re.Pattern):
            pttrn = checkpoint_line_pattern
        elif checkpoint_line_pattern:
            raise TypeError(f"Invalid checkpoint_line_pattern argument type: {type(checkpoint_line_pattern)}")

        args = args if isinstance(args, str) else list(map(str, args))
        with subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, **kwargs) as proc:
            assert proc.stdout, f"Subprocess {args} has not standard output"
            if checkpoint:
                self.log_checkpoint()

            self.log("======= Forwarding subprocess output for the following command =======")
            self.log(str(args))
            self.log("======================================================================")

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

            self.log("======================= Subprocess complete. =========================")


class ExternalJob(Job):
    """
    Mutable custom output job with customizeble input slots and output results.
    Use External jobs to save data save cryo-EM data generated by a software
    package outside of CryoSPARC.

    Created external jobs may be connected to any other CryoSPARC job result as
    an input. Its outputs must be created manually and may be configured to
    passthrough inherited input fields, just as with regular CryoSPARC jobs.

    Create a new External Job with `Project.create_external_job`_.

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

    .. _Project.create_external_job:
        project.html#cryosparc.project.Project.create_external_job

    """

    def add_input(
        self,
        type: Datatype,
        name: Optional[str] = None,
        min: int = 0,
        max: Union[int, Literal["inf"]] = "inf",
        slots: Iterable[Union[str, Datafield]] = [],
        title: Optional[str] = None,
    ):
        """
        Add an input slot to the current job. May be connected to zero or more
        outputs from other jobs (depending on the min and max values).

        Args:
            type (Datatype): cryo-EM data type for this output, e.g., "particle"
            name (str, optional): Output name key, e.g., "picked_particles".
                Defaults to None.
            min (int, optional): Minimum number of required input connections.
                Defaults to 0.
            max (int | Literal["inf"], optional): Maximum number of input
                connections. Specify ``"inf"`` for unlimited connections.
                Defaults to "inf".
            slots (list[str | Datafield], optional): List of slots that should
                be connected to this input, such as ``"location"`` or ``"blob"``
                Defaults to [].
            title (str, optional): Human-readable title for this input. Defaults
                to None.

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
        self.cs.vis.add_external_job_input(  # type: ignore
            project_uid=self.project_uid,
            job_uid=self.uid,
            type=type,
            name=name,
            min=min,
            max=max,
            slots=slots,
            title=title,
        )
        self.refresh()
        return self.doc["input_slot_groups"][-1]["name"]

    @overload
    def add_output(
        self,
        type: Datatype,
        name: Optional[str] = ...,
        slots: List[Union[str, Datafield]] = ...,
        passthrough: Optional[str] = ...,
        title: Optional[str] = None,
    ) -> str:
        ...

    @overload
    def add_output(
        self,
        type: Datatype,
        name: Optional[str] = ...,
        slots: List[Union[str, Datafield]] = ...,
        passthrough: Optional[str] = ...,
        title: Optional[str] = None,
        alloc: Union[int, Dataset] = ...,
    ) -> Dataset:
        ...

    def add_output(
        self,
        type: Datatype,
        name: Optional[str] = None,
        slots: List[Union[str, Datafield]] = [],
        passthrough: Optional[str] = None,
        title: Optional[str] = None,
        alloc: Union[int, Dataset, Literal[None]] = None,
    ) -> Union[str, Dataset]:
        """
        Add an output slot to the current job. Optionally returns the
        corresponding empty dataset if ``alloc`` is specified.

        Args:
            type (Datatype): cryo-EM datatype for this output, e.g., "particle"
            name (str, optional): Output name key, e.g., "selected_particles".
                Same as ``type`` if not specified. Defaults to None.
            slots (list[str, Datafield], optional): List of slot expected to be
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
            ...         {"dtype": "alignments3D", "prefix": "alignments_class_0", "required": True},
            ...         {"dtype": "alignments3D", "prefix": "alignments_class_1", "required": True},
            ...         {"dtype": "alignments3D", "prefix": "alignments_class_2", "required": True},
            ...     ]
            ... )
            "particle_alignments"
        """
        self.cs.vis.add_external_job_output(  # type: ignore
            project_uid=self.project_uid,
            job_uid=self.uid,
            type=type,
            name=name,
            slots=slots,
            passthrough=passthrough,
            title=title,
        )
        self.refresh()
        result_name = self.doc["output_result_groups"][-1]["name"]
        return result_name if alloc is None else self.alloc_output(result_name, alloc)

    def connect(
        self,
        target_input: str,
        source_job_uid: str,
        source_output: str,
        slots: List[Union[str, Datafield]] = [],
        title: str = "",
        desc: str = "",
        refresh: bool = True,
    ):
        """
        Connect the given input for this job to an output with given job UID and
        name. If this input does not exist, it will be added with the given
        slots. At least one slot must be specified if the input does not exist.

        Args:
            target_input (str): Input name to connect into. Will be created if
                does not already exist.
            source_job_uid (str): Job UID to connect from, e.g., "J42"
            source_output (str): Job output name to connect from , e.g.,
                "particles"
            slots (list[str | Datafield], optional): List of slots to add to
                created input. All if not specified. Defaults to [].
            title (str, optional): Human readable title for created input.
                Defaults to "".
            desc (str, optional): Human readable description for created input.
                Defaults to "".
            refresh (bool, optional): Auto-refresh job document after
                connecting. Defaults to True.

        Examples:

            Connect J3 to CTF-corrected micrographs from J2's ``micrographs``
            output.

            >>> cs = CryoSPARC()
            >>> project = cs.find_project("P3")
            >>> job = project.find_external_job("J3")
            >>> job.connect("J2", "micrographs", "input_micrographs")

        """
        assert source_job_uid != self.uid, f"Cannot connect job {self.uid} to itself"
        self.cs.vis.connect_external_job(  # type: ignore
            project_uid=self.project_uid,
            source_job_uid=source_job_uid,
            source_output=source_output,
            target_job_uid=self.uid,
            target_input=target_input,
            slots=slots,
            title=title,
            desc=desc,
        )
        if refresh:
            self.refresh()

    def alloc_output(self, name: str, alloc: Union[int, "ArrayLike", Dataset] = 0) -> Dataset:
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
        expected_fields = []
        for result in self.doc["output_results"]:
            if result["group_name"] != name or result["passthrough"]:
                continue
            prefix = result["name"]
            for field, dtype in result["min_fields"]:
                expected_fields.append((f"{prefix}/{field}", dtype))

        if not expected_fields:
            raise ValueError(f"No such output {name} on {self.project_uid}-{self.uid}")

        if isinstance(alloc, int):
            return Dataset.allocate(alloc, expected_fields)
        elif isinstance(alloc, Dataset):
            return Dataset({"uid": alloc["uid"]}).add_fields(expected_fields)
        else:
            return Dataset({"uid": alloc}).add_fields(expected_fields)

    def save_output(self, name: str, dataset: Dataset, refresh: bool = True):
        """
        Save output dataset to external job.

        Args:
            name (str): Name of output on this job.
            dataset (Dataset): Value of output with only required fields.
            refresh (bool, Optional): Auto-refresh job document after saving.
                Defaults to True

        Examples:

            Save a previously-allocated output.

            >>> cs = CryoSPARC()
            >>> project = cs.find_project("P3")
            >>> job = project.find_external_job("J3")
            >>> particles = job.alloc_output("picked_particles", 10000)
            >>> job.save_output("picked_particles", particles)

        """
        url = f"/external/projects/{self.project_uid}/jobs/{self.uid}/outputs/{name}/dataset"
        with make_request(self.cs.vis, url=url, data=dataset.stream()) as res:
            result = res.read().decode()
            assert res.status >= 200 and res.status < 400, f"Save output failed with message: {result}"
        if refresh:
            self.refresh()

    def start(self, status: Literal["running", "waiting"] = "waiting"):
        """
        Set job status to "running" or "waiting"

        Args:
            status (str, optional): "running" or "waiting". Defaults to "waiting".
        """
        assert status in {"running", "waiting"}, f"Invalid start status {status}"
        assert self.doc["status"] not in {
            "running",
            "waiting",
        }, f"Job {self.project_uid}-{self.uid} is already in running status"
        self.cs.cli.run_external_job(self.project_uid, self.uid, status)  # type: ignore
        self.refresh()

    def stop(self, error=False):
        """
        Set job status to "completed" or "failed"

        Args:
            error (bool, optional): Job completed with errors. Defaults to False.
        """
        status = "failed" if error else "completed"
        self.cs.cli.set_job_status(self.project_uid, self.uid, status)  # type: ignore
        self.refresh()

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
        error = False
        self.start("running")
        self.refresh()
        try:
            yield self
        except Exception:
            error = True
            raise
        finally:
            self.stop(error)  # TODO: Write Error to job log, if possible
            self.refresh()

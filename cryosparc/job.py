from contextlib import contextmanager
from io import BytesIO
import json
from pathlib import PurePath, PurePosixPath
from time import sleep
from typing import IO, TYPE_CHECKING, Any, Dict, Iterable, List, Optional, Pattern, Union, overload
from typing_extensions import Literal, TypedDict

from .command import make_json_request, make_request
from .dataset import Dataset
from .spec import Datatype, Datafield, JobDocument
from .dtype import decode_fields
from .util import bopen, first


if TYPE_CHECKING:
    from numpy.typing import NDArray
    from .tools import CryoSPARC


# Valid plot file types
TextFormat = Literal["txt", "csv", "json", "xml"]
ImageFormat = Literal["pdf", "gif", "jpg", "jpeg", "png", "svg"]
AssetFormat = Union[TextFormat, ImageFormat]
TextContentType = Literal[
    "text/plain",
    "text/csv",
    "application/json",
    "application/xml",
]
ImageContentType = Literal[
    "application/pdf",
    "image/gif",
    "image/jpeg",
    "image/png",
    "image/svg+xml",
]
AssetContentType = Union[TextContentType, ImageContentType]

TEXT_CONTENT_TYPES: Dict[TextFormat, TextContentType] = {
    "txt": "text/plain",
    "csv": "text/csv",
    "json": "application/json",
    "xml": "application/xml",
}

IMAGE_CONTENT_TYPES: Dict[ImageFormat, ImageContentType] = {
    "pdf": "application/pdf",
    "gif": "image/gif",
    "jpeg": "image/jpeg",
    "jpg": "image/jpeg",
    "png": "image/png",
    "svg": "image/svg+xml",
}

ASSET_CONTENT_TYPES: Dict[AssetFormat, AssetContentType] = {**TEXT_CONTENT_TYPES, **IMAGE_CONTENT_TYPES}  # type: ignore


class AssetFileData(TypedDict):
    """
    Result of job files query
    """

    _id: str
    filename: str
    contentType: AssetContentType
    uploadDate: str  # ISO formatted
    length: int  # in bytes
    chunkSize: int  # in bytes
    md5: str
    project_uid: str
    job_uid: str  # also used for Session UID


class Job:
    """
    Accessor class to a job in CryoSPARC with ability to load inputs and
    outputs, add to job log, download job files
    """

    _doc: Optional[JobDocument] = None

    def __init__(self, cs: "CryoSPARC", project_uid: str, uid: str) -> None:
        self.cs = cs
        self.project_uid = project_uid
        self.uid = uid

    @property
    def doc(self) -> JobDocument:
        if not self._doc:
            self.refresh()
        assert self._doc, "Could not refresh job document"
        return self._doc

    def refresh(self):
        self._doc = self.cs.cli.get_job(self.project_uid, self.uid)  # type: ignore
        return self

    def dir(self) -> PurePosixPath:
        """
        Get the path to the project directory
        """
        return PurePosixPath(self.cs.cli.get_job_dir_abs(self.project_uid, self.uid))  # type: ignore

    def clear(self):
        return self.cs.cli.clear_job(self.project_uid, self.uid)  # type: ignore

    def load_input(self, name: str, fields: Iterable[str] = []):
        job = self.doc
        group = first(s for s in job["input_slot_groups"] if s["name"] == name)
        if not group:
            raise TypeError(f"Job {self.project_uid}-{self.uid} does not have an input {name}")

        data = {"project_uid": self.project_uid, "job_uid": self.uid, "input_name": name, "slots": list(fields)}

        with make_json_request(self.cs.vis, "/load_job_input", data=data) as response:
            mime = response.headers.get("Content-Type")
            if mime != "application/x-cryosparc-dataset":
                raise TypeError(f"Unable to load dataset for job {self.project_uid}-{self.uid} input {name}")
            return Dataset.load(response)

    def load_output(self, name: str, fields: Iterable[str] = []):
        job = self.doc
        fields = set(fields)
        results = [
            result
            for result in job["output_results"]
            if result["group_name"] == name and (not fields or result["name"] in fields)
        ]
        if not results:
            raise TypeError(f"Job {self.project_uid}-{self.uid} does not have any results for output {name}")

        metafiles = set().union(*(r["metafiles"] for r in results))
        datasets = [self.cs.download_dataset(self.project_uid, f) for f in metafiles]
        return Dataset.innerjoin(*datasets)

    def log(self, text: str, level: Literal["text", "warning", "error"] = "text"):
        """
        Append to a job's event log
        """
        return self.cs.cli.job_send_streamlog(  # type: ignore
            project_uid=self.project_uid, job_uid=self.uid, message=text, error=level != "text"
        )

    def log_checkpoint(self, meta: dict = {}):
        """
        Append a checkpoint to the job's event log
        """
        return self.cs.cli.job_checkpoint_streamlog(  # type: ignore
            project_uid=self.project_uid, job_uid=self.uid, meta=meta
        )

    def log_plot(
        self,
        figure: Union[str, PurePath, IO[bytes], Any],
        text: str,
        formats: Iterable[ImageFormat] = ["png", "pdf"],
        flags: List[str] = ["plots"],
        raw_data: Union[str, bytes, Literal[None]] = None,
        raw_data_file: Union[str, PurePath, IO[bytes], Literal[None]] = None,
        raw_data_format: Optional[TextFormat] = None,
        savefig_kw: dict = dict(bbox_inches="tight", pad_inches=0),
    ):
        """
        Add a log line with the given figure.

        `figure` must be one of the following

        - Path to an existing image file in PNG, JPEG, GIF, SVG or PDF format
        - A file handle-like object with the binary data of an image
        - A matplotlib plot

        If a matplotlib figure is specified, Uploads the plots in `png` and
        `pdf` formats. Override the `formats` argument with
        `formats=['<format1>', '<format2>', ...]` to save in different image
        formats.

        If a file handle is specified, also specify `formats=['<format>']`,
        where `<format>` is a valid image extension such as `png` or `pdf`.
        Assumes `png` if not specified.

        If a text-version of the given plot is available (e.g., csv), specify
        `raw_data` with the full contents or `raw_data_file` with a path or
        binary file handle pointing to the contents. Assumes file format from
        extension or `raw_data_format`. Defaults to txt if cannot be determined.
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

    def download(self, path: Union[str, PurePosixPath]):
        path = PurePosixPath(self.uid) / path
        return self.cs.download(self.project_uid, path)

    def download_file(self, path: Union[str, PurePosixPath], target: Union[str, PurePath, IO[bytes]]):
        path = PurePosixPath(self.uid) / path
        return self.cs.download_file(self.project_uid, path, target)

    def download_dataset(self, path: Union[str, PurePosixPath]):
        path = PurePosixPath(self.uid) / path
        return self.cs.download_dataset(self.project_uid, path)

    def download_mrc(self, path: Union[str, PurePosixPath]):
        path = PurePosixPath(self.uid) / path
        return self.cs.download_mrc(self.project_uid, path)

    def list_assets(self) -> List[AssetFileData]:
        """
        Get a list of files available in the database for this job. Returns a
        list with details about the assets. Each entry is a dict with a ``_id``
        key which may be used to download the file with the ``download_asset``
        method.

        Returns:
            list[AssetFileData]: Asset details
        """
        return self.cs.vis.list_job_files(project_uid=self.project_uid, job_uid=self.uid)  # type: ignore

    def download_asset(self, fileid: str, target: Union[str, PurePath, IO[bytes]]):
        """
        Download a job asset from the database with the given ID. Note that the
        file does not necessary have to belong to the current job.

        Args:
            fileid (str): GridFS file object ID
            target (str | Path | IO): Writable download destination path or file handle
        """
        return self.cs.download_asset(fileid, target)

    def upload(self, path: Union[str, PurePosixPath], file: Union[str, PurePath, IO[bytes]]):
        """
        Upload the given file to the job directory at the given path.
        """
        path = PurePosixPath(self.uid) / path
        return self.cs.upload(self.project_uid, path, file)

    def upload_asset(
        self,
        file: Union[str, PurePath, IO[bytes]],
        filename: Optional[str] = None,
        format: Optional[AssetFormat] = None,
    ):
        """
        Upload an image or text file to the current job. Specify either an image
        (PNG, JPG, GIF, PDF, SVG), text file (TXT, CSV, JSON, XML) or a binary
        IO object with data in one of those formats.

        If a binary IO object is specified, either a filename or mimetype must
        be specified.

        Unlike the `upload` method which saves files to the job directory, this
        method saves images to the database and exposes them for use in the job
        log.

        If specifying arbitrary binary I/O, specify either a filename or a file
        format.
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
    ):
        """
        Upload the given figure. Returns a list of the created asset objects.
        See `log_plot` for argument explanations.
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

    def upload_dataset(self, path: Union[str, PurePosixPath], dset: Dataset):
        path = PurePosixPath(self.uid) / path
        return self.cs.upload_dataset(self.project_uid, path, dset)

    def upload_mrc(self, path: Union[str, PurePosixPath], data: "NDArray", psize: float):
        path = PurePosixPath(self.uid) / path
        return self.cs.upload_mrc(self.project_uid, path, data, psize)

    def subprocess(
        self,
        args: Union[str, list],
        mute: bool = False,
        checkpoint: bool = False,
        checkpoint_line_pattern: Union[str, Pattern[str], Literal[None]] = None,
        **kwargs,
    ):
        """
        Launch a subprocess and write its output and error to the job log.

        Set `mute=True` to prevent forwarding the output to standard output.

        Specify `checkpoint=True` to add a checkpoint to the stream log just
        before the output begins.

        Specify `checkpoint_line_pattern` as a regular expression. If a
        given line matches the pattern, adds checkpoint to the job log _before_
        that line is added to the log. Use this for processes with a lot of
        structured output.
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
    Mutable custom job with customizeble input slots and saveable results. Used
    to save data
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
        Add an input slot to the current job.
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
        return self.doc["output_result_groups"][-1]["name"]

    @overload
    def add_output(
        self,
        type: Datatype,
        name: Optional[str] = ...,
        slots: List[Union[str, Datafield]] = ...,
        passthrough: Union[str, Literal[False]] = ...,
        title: Optional[str] = None,
    ) -> str:
        ...

    @overload
    def add_output(
        self,
        type: Datatype,
        name: Optional[str] = ...,
        slots: List[Union[str, Datafield]] = ...,
        passthrough: Union[str, Literal[False]] = ...,
        title: Optional[str] = None,
        alloc: int = ...,
    ) -> Dataset:
        ...

    def add_output(
        self,
        type: Datatype,
        name: Optional[str] = None,
        slots: List[Union[str, Datafield]] = [],
        passthrough: Optional[str] = None,
        title: Optional[str] = None,
        alloc: Optional[int] = None,
    ) -> Union[str, Dataset]:
        """
        Add an output slot to the current job.

        One of `type` or `passthrough` must be specified, where `passthrough` is
        the name of an existing input (added via `add_input`).

        Returns the name of the created output. If `init` is set to an integer,
        returns blank dataset initialized with the given number of items.
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
        name = self.doc["output_result_groups"][-1]["name"]
        return name if alloc is None else self.alloc_output(name, alloc)

    def connect(
        self,
        source_job_uid: str,
        source_output: str,
        target_input: str,
        slots: List[Union[str, Datafield]] = [],
        title: str = "",
        desc: str = "",
    ):
        """
        Connect the given input for this job to an output with given job UID and
        name. If this input does not exist, it will be added with the given
        slots. At least one slot must be specified if the input does not exist.
        """
        assert source_job_uid != self.uid, f"Cannot connect job {self.uid} to itself"
        status: bool = self.cs.vis.connect_external_job(  # type: ignore
            project_uid=self.project_uid,
            source_job_uid=source_job_uid,
            source_output=source_output,
            target_job_uid=self.uid,
            target_input=target_input,
            slots=slots,
            title=title,
            desc=desc,
        )
        self.refresh()
        return status

    def alloc_output(self, name: str, size: int = 0):
        """
        Allocate an empty dataset for the given output with the given name.
        Initialize with the given number of empty rows.
        """
        fields = self.cs.cli.get_job_output_min_fields(self.project_uid, self.uid, name)  # type: ignore
        fields = decode_fields(fields)
        return Dataset.allocate(size, fields)

    def save_output(self, name: str, dataset: Dataset):
        """
        Job must have status "running" for this to work
        """
        url = f"/external/projects/{self.project_uid}/jobs/{self.uid}/outputs/{name}/dataset"
        with make_request(self.cs.vis, url, data=dataset.stream()) as res:
            result = res.read().decode()
            assert res.status >= 200 and res.status < 400, f"Save output failed with message: {result}"

    def start(self, status: Literal["running", "waiting"] = "waiting"):
        # Set job status to "running"
        assert status in {"running", "waiting"}, f"Invalid start status {status}"
        self.cs.cli.set_job_status(self.project_uid, self.uid, status)  # type: ignore

    def stop(self, error=False):
        # Set job status to "completed" or "failed"
        status = "failed" if error else "completed"
        self.cs.cli.set_job_status(self.project_uid, self.uid, status)  # type: ignore

    @contextmanager
    def run(self):
        error = False
        self.start("running")
        try:
            yield self
        except Exception:
            error = True
            raise
        finally:
            self.stop(error)

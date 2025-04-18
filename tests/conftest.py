import shutil
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from time import time
from unittest import mock

import numpy as n
import pytest

from cryosparc.api import APIClient
from cryosparc.controllers.project import ProjectController
from cryosparc.dataset import Dataset as BaseDataset
from cryosparc.dataset import Row
from cryosparc.models.auth import Token
from cryosparc.models.job import Job
from cryosparc.models.job_spec import (
    Connection,
    Input,
    InputResult,
    Inputs,
    JobSpec,
    Output,
    OutputResult,
    Outputs,
    Params,
    ResourceSpec,
)
from cryosparc.models.project import Project
from cryosparc.models.user import Email, User
from cryosparc.tools import CryoSPARC
from cryosparc.util import default_rng


# Always use this class for testing to ensure Dataset#items property is never
# used internally. Downstream CryoSPARC relies on this.
class Dataset(BaseDataset[Row]):
    # Override items like the Particles class does in CryoSPARC
    @property
    def items(self):  # type: ignore
        return self.rows()

    def shuffle(self):
        idxs = n.arange(len(self))
        default_rng().shuffle(idxs)
        return self.take(idxs)


# fmt: off
T20S_PARTICLES = Dataset(
    n.rec.array([
        (  531905114944910449, 'J30/extract/012951868257382468663_14sep05c_00024sq_00004hl_00002es.frames_patch_aligned_doseweighted_particles.mrc', 176, [448, 448], 0.6575, -1., 0, 'spline', 0, 300., 2.7, 0.1, 16400.2  , 16232.468,  4.6313896, 0., 1., 1., [0., 0.], [0., 0.], [0., 0.], [0., 0., 0., 0.], [0., 0., 0., 0.], 0.),  # noqa
        ( 1832753233363106142, 'J30/extract/012756078269171603280_14sep05c_c_00003gr_00014sq_00005hl_00005es.frames_patch_aligned_doseweighted_particles.mrc', 210, [448, 448], 0.6575, -1., 0, 'spline', 0, 300., 2.7, 0.1, 13942.286, 13810.533,  4.695857 , 0., 1., 1., [0., 0.], [0., 0.], [0., 0.], [0., 0., 0., 0.], [0., 0., 0., 0.], 0.),  # noqa
        ( 2101618993165418746, 'J30/extract/000588255143468195995_14sep05c_c_00003gr_00014sq_00011hl_00004es.frames_patch_aligned_doseweighted_particles.mrc', 325, [448, 448], 0.6575, -1., 0, 'spline', 0, 300., 2.7, 0.1, 15820.722, 15637.411, -1.5558333, 0., 1., 1., [0., 0.], [0., 0.], [0., 0.], [0., 0., 0., 0.], [0., 0., 0., 0.], 0.),  # noqa
        ( 2803196405397048440, 'J30/extract/008578565574161745010_14sep05c_c_00003gr_00014sq_00006hl_00003es.frames_patch_aligned_doseweighted_particles.mrc',  29, [448, 448], 0.6575, -1., 0, 'spline', 0, 300., 2.7, 0.1, 19080.65 , 18854.29 , -1.5461981, 0., 1., 1., [0., 0.], [0., 0.], [0., 0.], [0., 0., 0., 0.], [0., 0., 0., 0.], 0.),  # noqa
        ( 3817134099570652656, 'J30/extract/006843432895979504867_14sep05c_c_00003gr_00014sq_00006hl_00002es.frames_patch_aligned_doseweighted_particles.mrc', 613, [448, 448], 0.6575, -1., 0, 'spline', 0, 300., 2.7, 0.1, 21752.17 , 21490.023,  4.695863 , 0., 1., 1., [0., 0.], [0., 0.], [0., 0.], [0., 0., 0., 0.], [0., 0., 0., 0.], 0.),  # noqa
        ( 9207589919858288823, 'J30/extract/012951868257382468663_14sep05c_00024sq_00004hl_00002es.frames_patch_aligned_doseweighted_particles.mrc', 398, [448, 448], 0.6575, -1., 0, 'spline', 0, 300., 2.7, 0.1, 16378.698, 16210.968,  4.6313896, 0., 1., 1., [0., 0.], [0., 0.], [0., 0.], [0., 0., 0., 0.], [0., 0., 0., 0.], 0.),  # noqa
        ( 9881411471502859237, 'J30/extract/012756078269171603280_14sep05c_c_00003gr_00014sq_00005hl_00005es.frames_patch_aligned_doseweighted_particles.mrc', 177, [448, 448], 0.6575, -1., 0, 'spline', 0, 300., 2.7, 0.1, 13765.75 , 13633.997,  4.695857 , 0., 1., 1., [0., 0.], [0., 0.], [0., 0.], [0., 0., 0., 0.], [0., 0., 0., 0.], 0.),  # noqa
        (13075914070757904223, 'J30/extract/003729228794286345575_14sep05c_c_00003gr_00014sq_00008hl_00005es.frames_patch_aligned_doseweighted_particles.mrc', 446, [448, 448], 0.6575, -1., 0, 'spline', 0, 300., 2.7, 0.1, 18536.885, 18341.818,  4.678856 , 0., 1., 1., [0., 0.], [0., 0.], [0., 0.], [0., 0., 0., 0.], [0., 0., 0., 0.], 0.),  # noqa
        (13385778774240615983, 'J30/extract/011450458613449160526_14sep05c_c_00003gr_00014sq_00010hl_00002es.frames_patch_aligned_doseweighted_particles.mrc',  22, [448, 448], 0.6575, -1., 0, 'spline', 0, 300., 2.7, 0.1, 17423.697, 17225.484, -1.5046247, 0., 1., 1., [0., 0.], [0., 0.], [0., 0.], [0., 0., 0., 0.], [0., 0., 0., 0.], 0.),  # noqa
        (13864605955862944880, 'J30/extract/008578565574161745010_14sep05c_c_00003gr_00014sq_00006hl_00003es.frames_patch_aligned_doseweighted_particles.mrc',  92, [448, 448], 0.6575, -1., 0, 'spline', 0, 300., 2.7, 0.1, 18994.4  , 18768.04 , -1.5461981, 0., 1., 1., [0., 0.], [0., 0.], [0., 0.], [0., 0., 0., 0.], [0., 0., 0., 0.], 0.),  # noqa
    ], dtype=[  # type: ignore
            ('uid', '<u8'),
            ('blob/path', 'O'),
            ('blob/idx', '<u4'),
            ('blob/shape', '<u4', (2,)),
            ('blob/psize_A', '<f4'),
            ('blob/sign', '<f4'),
            ('blob/import_sig', '<u8'),
            ('ctf/type', 'O'),
            ('ctf/exp_group_id', '<u4'),
            ('ctf/accel_kv', '<f4'),
            ('ctf/cs_mm', '<f4'),
            ('ctf/amp_contrast', '<f4'),
            ('ctf/df1_A', '<f4'),
            ('ctf/df2_A', '<f4'),
            ('ctf/df_angle_rad', '<f4'),
            ('ctf/phase_shift_rad', '<f4'),
            ('ctf/scale', '<f4'),
            ('ctf/scale_const', '<f4'),
            ('ctf/shift_A', '<f4', (2,)),
            ('ctf/tilt_A', '<f4', (2,)),
            ('ctf/trefoil_A', '<f4', (2,)),
            ('ctf/tetra_A', '<f4', (4,)),
            ('ctf/anisomag', '<f4', (4,)),
            ('ctf/bfactor', '<f4')
        ]
    )
)

T20S_PARTICLES_PASSTHROUGH = Dataset(
    n.rec.array([
        (  531905114944910449, 12951868257382468663, 0, 'J14/motioncorrected/012951868257382468663_14sep05c_00024sq_00004hl_00002es.frames_patch_aligned_doseweighted.mrc', [7676, 7420], 0.22586207, 0.10166667, 100.),  # noqa
        ( 1832753233363106142, 12756078269171603280, 0, 'J14/motioncorrected/012756078269171603280_14sep05c_c_00003gr_00014sq_00005hl_00005es.frames_patch_aligned_doseweighted.mrc', [7676, 7420], 0.08965518, 0.52166665, 100.),  # noqa
        ( 2101618993165418746,   588255143468195995, 0, 'J14/motioncorrected/000588255143468195995_14sep05c_c_00003gr_00014sq_00011hl_00004es.frames_patch_aligned_doseweighted.mrc', [7676, 7420], 0.8913793 , 0.945     , 100.),  # noqa
        ( 2803196405397048440,  8578565574161745010, 0, 'J14/motioncorrected/008578565574161745010_14sep05c_c_00003gr_00014sq_00006hl_00003es.frames_patch_aligned_doseweighted.mrc', [7676, 7420], 0.38448277, 0.18333334, 100.),  # noqa
        ( 3817134099570652656,  6843432895979504867, 0, 'J14/motioncorrected/006843432895979504867_14sep05c_c_00003gr_00014sq_00006hl_00002es.frames_patch_aligned_doseweighted.mrc', [7676, 7420], 0.9155173 , 0.55      , 100.),  # noqa
        ( 9207589919858288823, 12951868257382468663, 0, 'J14/motioncorrected/012951868257382468663_14sep05c_00024sq_00004hl_00002es.frames_patch_aligned_doseweighted.mrc', [7676, 7420], 0.26206896, 0.13333334, 100.),  # noqa
        ( 9881411471502859237, 12756078269171603280, 0, 'J14/motioncorrected/012756078269171603280_14sep05c_c_00003gr_00014sq_00005hl_00005es.frames_patch_aligned_doseweighted.mrc', [7676, 7420], 0.19137931, 0.84833336, 100.),  # noqa
        (13075914070757904223,  3729228794286345575, 0, 'J14/motioncorrected/003729228794286345575_14sep05c_c_00003gr_00014sq_00008hl_00005es.frames_patch_aligned_doseweighted.mrc', [7676, 7420], 0.8413793 , 0.105     , 100.),  # noqa
        (13385778774240615983, 11450458613449160526, 0, 'J14/motioncorrected/011450458613449160526_14sep05c_c_00003gr_00014sq_00010hl_00002es.frames_patch_aligned_doseweighted.mrc', [7676, 7420], 0.7413793 , 0.34      , 100.),  # noqa
        (13864605955862944880,  8578565574161745010, 0, 'J14/motioncorrected/008578565574161745010_14sep05c_c_00003gr_00014sq_00006hl_00003es.frames_patch_aligned_doseweighted.mrc', [7676, 7420], 0.0862069 , 0.29333332, 100.),  # noqa
    ], dtype=[  # type: ignore
            ('uid', '<u8'),
            ('location/micrograph_uid', '<u8'),
            ('location/exp_group_id', '<u4'),
            ('location/micrograph_path', 'O'),
            ('location/micrograph_shape', '<u4', (2,)),
            ('location/center_x_frac', '<f4'),
            ('location/center_y_frac', '<f4'),
            ('location/min_dist_A', '<f4')
    ])
)
# fmt: on


@pytest.fixture(scope="session")
def big_dset_path():
    basename = "bench_big_dset"
    existing_path = Path.home() / f"{basename}.cs"

    print("")  # Keeps printing clean
    if existing_path.exists():
        print(f"Using local sample data {existing_path}; loading...", end="\r")
        yield existing_path
    else:
        import gzip
        import tempfile

        download_url = f"https://cryosparc-test-data-dist.s3.amazonaws.com/{basename}.cs.gz"
        with tempfile.TemporaryDirectory() as tmpdir:
            compressed_path = Path(tmpdir) / f"{basename}.cs.gz"
            cs_path = Path(tmpdir) / f"{basename}.cs"

            def download_report_hook(chunk, chunk_size, total_size):
                total_chunks = total_size / chunk_size
                progress = chunk / total_chunks * 100
                print(f"Downloading big dataset sample data ({total_size} bytes, {progress:.0f}%)", end="\r")

            urllib.request.urlretrieve(download_url, compressed_path, reporthook=download_report_hook)
            with gzip.open(compressed_path, "rb") as compressed_file, open(cs_path, "wb") as cs_file:
                shutil.copyfileobj(compressed_file, cs_file)

            print("")
            print("Downloaded big dataset sample data; loading...", end="\r")
            yield cs_path


@pytest.fixture(scope="session")
def big_dset(big_dset_path):
    tic = time()
    dset = Dataset.load(big_dset_path)
    print(f"Loaded big dataset in {time() - tic:.3f} seconds")
    return dset


@pytest.fixture
def t20s_dset():
    # fmt: off
    return Dataset(n.rec.array([
        (15762140416835289736, "J1/imported/015762140416835289736_14sep05c_00024sq_00003hl_00002es.frames.tif", [  38, 7676, 7420], 0.6575, 0, "TIFF", 0, 14011218726240589193, "J1/imported/norm-amibox05-0.mrc", 0, [7676, 7420], 0, 0, 0, 300., 2.7, 53., 0, 0, 8, "", "", 0, [0, 0], 0., "", 0, 0., 0., 0),  # noqa
        ( 7260969837812210921, "J1/imported/007260969837812210921_14sep05c_00024sq_00003hl_00005es.frames.tif", [  38, 7676, 7420], 0.6575, 0, "TIFF", 0,  5837173768828397691, "J1/imported/norm-amibox05-0.mrc", 0, [7676, 7420], 0, 0, 0, 300., 2.7, 53., 0, 0, 8, "", "", 0, [0, 0], 0., "", 0, 0., 0., 0),  # noqa
        (10800491058062947733, "J1/imported/010800491058062947733_14sep05c_00024sq_00004hl_00002es.frames.tif", [  38, 7676, 7420], 0.6575, 0, "TIFF", 0,  1196980374560176443, "J1/imported/norm-amibox05-0.mrc", 0, [7676, 7420], 0, 0, 0, 300., 2.7, 53., 0, 0, 8, "", "", 0, [0, 0], 0., "", 0, 0., 0., 0),  # noqa
        (11991710339096669080, "J1/imported/011991710339096669080_14sep05c_00024sq_00006hl_00003es.frames.tif", [  38, 7676, 7420], 0.6575, 0, "TIFF", 0,  7129025008161402787, "J1/imported/norm-amibox05-0.mrc", 0, [7676, 7420], 0, 0, 0, 300., 2.7, 53., 0, 0, 8, "", "", 0, [0, 0], 0., "", 0, 0., 0., 0),  # noqa
        (10066607318144328742, "J1/imported/010066607318144328742_14sep05c_c_00003gr_00014sq_00002hl_00005es.frames.tif", [  38, 7676, 7420], 0.6575, 0, "TIFF", 0,  9773685899756902191, "J1/imported/norm-amibox05-0.mrc", 0, [7676, 7420], 0, 0, 0, 300., 2.7, 53., 0, 0, 8, "", "", 0, [0, 0], 0., "", 0, 0., 0., 0),  # noqa
        (14059663819601300436, "J1/imported/014059663819601300436_14sep05c_c_00003gr_00014sq_00004hl_00004es.frames.tif", [  38, 7676, 7420], 0.6575, 0, "TIFF", 0,  3988647519168980973, "J1/imported/norm-amibox05-0.mrc", 0, [7676, 7420], 0, 0, 0, 300., 2.7, 53., 0, 0, 8, "", "", 0, [0, 0], 0., "", 0, 0., 0., 0),  # noqa
        (  273322377584193832, "J1/imported/000273322377584193832_14sep05c_c_00003gr_00014sq_00005hl_00002es.frames.tif", [  38, 7676, 7420], 0.6575, 0, "TIFF", 0, 16231222940328059768, "J1/imported/norm-amibox05-0.mrc", 0, [7676, 7420], 0, 0, 0, 300., 2.7, 53., 0, 0, 8, "", "", 0, [0, 0], 0., "", 0, 0., 0., 0),  # noqa
        ( 3499866221971477386, "J1/imported/003499866221971477386_14sep05c_c_00003gr_00014sq_00005hl_00003es.frames.tif", [  38, 7676, 7420], 0.6575, 0, "TIFF", 0,  9286395140289536592, "J1/imported/norm-amibox05-0.mrc", 0, [7676, 7420], 0, 0, 0, 300., 2.7, 53., 0, 0, 8, "", "", 0, [0, 0], 0., "", 0, 0., 0., 0),  # noqa
        ( 3848558957898561278, "J1/imported/003848558957898561278_14sep05c_c_00003gr_00014sq_00005hl_00005es.frames.tif", [  38, 7676, 7420], 0.6575, 0, "TIFF", 0,  3782318515582039935, "J1/imported/norm-amibox05-0.mrc", 0, [7676, 7420], 0, 0, 0, 300., 2.7, 53., 0, 0, 8, "", "", 0, [0, 0], 0., "", 0, 0., 0., 0),  # noqa
        ( 9829431748377878115, "J1/imported/009829431748377878115_14sep05c_c_00003gr_00014sq_00006hl_00002es.frames.tif", [  38, 7676, 7420], 0.6575, 0, "TIFF", 0, 17466118442802771446, "J1/imported/norm-amibox05-0.mrc", 0, [7676, 7420], 0, 0, 0, 300., 2.7, 53., 0, 0, 8, "", "", 0, [0, 0], 0., "", 0, 0., 0., 0),  # noqa
        ( 2867928639359948788, "J1/imported/002867928639359948788_14sep05c_c_00003gr_00014sq_00006hl_00003es.frames.tif", [  38, 7676, 7420], 0.6575, 0, "TIFF", 0, 17119601282827635162, "J1/imported/norm-amibox05-0.mrc", 0, [7676, 7420], 0, 0, 0, 300., 2.7, 53., 0, 0, 8, "", "", 0, [0, 0], 0., "", 0, 0., 0., 0),  # noqa
        ( 8455357976034962133, "J1/imported/008455357976034962133_14sep05c_c_00003gr_00014sq_00006hl_00005es.frames.tif", [  38, 7676, 7420], 0.6575, 0, "TIFF", 0, 10475424305309211320, "J1/imported/norm-amibox05-0.mrc", 0, [7676, 7420], 0, 0, 0, 300., 2.7, 53., 0, 0, 8, "", "", 0, [0, 0], 0., "", 0, 0., 0., 0),  # noqa
        ( 4503027956576488873, "J1/imported/004503027956576488873_14sep05c_c_00003gr_00014sq_00007hl_00004es.frames.tif", [  38, 7676, 7420], 0.6575, 0, "TIFF", 0,  4382433041497215537, "J1/imported/norm-amibox05-0.mrc", 0, [7676, 7420], 0, 0, 0, 300., 2.7, 53., 0, 0, 8, "", "", 0, [0, 0], 0., "", 0, 0., 0., 0),  # noqa
        ( 6657609670738933979, "J1/imported/006657609670738933979_14sep05c_c_00003gr_00014sq_00007hl_00005es.frames.tif", [  38, 7676, 7420], 0.6575, 0, "TIFF", 0,  7889866034561464467, "J1/imported/norm-amibox05-0.mrc", 0, [7676, 7420], 0, 0, 0, 300., 2.7, 53., 0, 0, 8, "", "", 0, [0, 0], 0., "", 0, 0., 0., 0),  # noqa
        ( 2455776040951224707, "J1/imported/002455776040951224707_14sep05c_c_00003gr_00014sq_00008hl_00005es.frames.tif", [  38, 7676, 7420], 0.6575, 0, "TIFF", 0,  5928873226622420600, "J1/imported/norm-amibox05-0.mrc", 0, [7676, 7420], 0, 0, 0, 300., 2.7, 53., 0, 0, 8, "", "", 0, [0, 0], 0., "", 0, 0., 0., 0),  # noqa
        ( 8270327040298559173, "J1/imported/008270327040298559173_14sep05c_c_00003gr_00014sq_00009hl_00004es.frames.tif", [  38, 7676, 7420], 0.6575, 0, "TIFF", 0,  2857337051791529295, "J1/imported/norm-amibox05-0.mrc", 0, [7676, 7420], 0, 0, 0, 300., 2.7, 53., 0, 0, 8, "", "", 0, [0, 0], 0., "", 0, 0., 0., 0),  # noqa
        ( 2424724526127669191, "J1/imported/002424724526127669191_14sep05c_c_00003gr_00014sq_00010hl_00002es.frames.tif", [  38, 7676, 7420], 0.6575, 0, "TIFF", 0,  1126123402376461244, "J1/imported/norm-amibox05-0.mrc", 0, [7676, 7420], 0, 0, 0, 300., 2.7, 53., 0, 0, 8, "", "", 0, [0, 0], 0., "", 0, 0., 0., 0),  # noqa
        (15890047193456637277, "J1/imported/015890047193456637277_14sep05c_c_00003gr_00014sq_00011hl_00002es.frames.tif", [  38, 7676, 7420], 0.6575, 0, "TIFF", 0,  4134148380692987552, "J1/imported/norm-amibox05-0.mrc", 0, [7676, 7420], 0, 0, 0, 300., 2.7, 53., 0, 0, 8, "", "", 0, [0, 0], 0., "", 0, 0., 0., 0),  # noqa
        (11981388473018700511, "J1/imported/011981388473018700511_14sep05c_c_00003gr_00014sq_00011hl_00003es.frames.tif", [  38, 7676, 7420], 0.6575, 0, "TIFF", 0,  8989915083856388927, "J1/imported/norm-amibox05-0.mrc", 0, [7676, 7420], 0, 0, 0, 300., 2.7, 53., 0, 0, 8, "", "", 0, [0, 0], 0., "", 0, 0., 0., 0),  # noqa
        (12098792569994765948, "J1/imported/012098792569994765948_14sep05c_c_00003gr_00014sq_00011hl_00004es.frames.tif", [  38, 7676, 7420], 0.6575, 0, "TIFF", 0,  7878917727182588741, "J1/imported/norm-amibox05-0.mrc", 0, [7676, 7420], 0, 0, 0, 300., 2.7, 53., 0, 0, 8, "", "", 0, [0, 0], 0., "", 0, 0., 0., 0)  # noqa
    ], dtype=[  # type: ignore
        ("uid", "<u8"),
        ("movie_blob/path", "O"),
        ("movie_blob/shape", "<u4", (3,)),
        ("movie_blob/psize_A", "<f4"),
        ("movie_blob/is_gain_corrected", "<u4"),
        ("movie_blob/format", "O"),
        ("movie_blob/has_defect_file", "<u4"),
        ("movie_blob/import_sig", "<u8"),
        ("gain_ref_blob/path", "O"),
        ("gain_ref_blob/idx", "<u4"),
        ("gain_ref_blob/shape", "<u4", (2,)),
        ("gain_ref_blob/flip_x", "<u4"),
        ("gain_ref_blob/flip_y", "<u4"),
        ("gain_ref_blob/rotate_num", "<u4"),
        ("mscope_params/accel_kv", "<f4"),
        ("mscope_params/cs_mm", "<f4"),
        ("mscope_params/total_dose_e_per_A2", "<f4"),
        ("mscope_params/phase_plate", "<u4"),
        ("mscope_params/neg_stain", "<u4"),
        ("mscope_params/exp_group_id", "<u4"),
        ("mscope_params/defect_path", "O"),
        ("micrograph_blob/path", "O"),
        ("micrograph_blob/idx", "<u4"),
        ("micrograph_blob/shape", "<u4", (2,)),
        ("micrograph_blob/psize_A", "<f4"),
        ("micrograph_blob/format", "O"),
        ("micrograph_blob/is_background_subtracted", "<u4"),
        ("micrograph_blob/vmin", "<f4"),
        ("micrograph_blob/vmax", "<f4"),
        ("micrograph_blob/import_sig", "<u8")
    ]))
    # fmt: on


@pytest.fixture
def t20s_particles():
    return T20S_PARTICLES


@pytest.fixture
def t20s_particles_passthrough():
    return T20S_PARTICLES_PASSTHROUGH


@pytest.fixture
def mock_user():
    return User(
        _id="6372a35e821ed2b71d9fe4e3",
        name="test",
        first_name="Testy",
        last_name="Tester",
        emails=[Email(address="test@example.com", verified=True)],
        roles={"__global_roles__": ["admin"]},
        register_token=None,
        allowed_prefix_dir="/",
        created_at=datetime(2017, 1, 1, tzinfo=timezone.utc),
    )


@pytest.fixture
def mock_api_client_class(mock_user, monkeypatch):
    monkeypatch.setattr(APIClient, "__call__", mock.Mock(return_value=None))
    APIClient.health = mock.Mock(return_value="OK")
    APIClient.login = mock.Mock(return_value=Token(access_token="abc123", token_type="bearer"))
    APIClient.users = mock.MagicMock()
    APIClient.config = mock.MagicMock()
    APIClient.projects = mock.MagicMock()
    APIClient.workspaces = mock.MagicMock()
    APIClient.jobs = mock.MagicMock()
    APIClient.users.me.return_value = mock_user
    APIClient.config.get_version.return_value = "develop"
    return APIClient


@pytest.fixture
def cs(mock_api_client_class):
    return CryoSPARC(
        "https://cryosparc.example.com",
        email="structura@example.com",
        password="password",
        host=None,
        base_port=None,
    )


@pytest.fixture
def mock_project(mock_user):
    return Project(
        _id="67292e95282b26b45d0e8fae",
        uid="P1",
        uid_num=1,
        title="Test Project",
        project_dir="/home/cryosparc/projects",
        owner_user_id=mock_user.id,
        size_last_updated=datetime.now(timezone.utc),
    )


@pytest.fixture
def project(cs: CryoSPARC, mock_project):
    APIClient.projects.find_one.return_value = mock_project  # type: ignore
    return cs.find_project("P1")


@pytest.fixture
def mock_new_job(mock_user, mock_project):
    return Job(
        _id="67743226e66c192db762b689",
        uid="J42",
        uid_num=42,
        project_uid=mock_project.uid,
        project_uid_num=mock_project.uid_num,
        workspace_uids=["W1"],
        job_dir="J42",
        status="building",
        status_num=5,
        created_by_user_id=mock_user.id,
        spec=JobSpec(
            type="homo_abinit",
            params=Params(),
            inputs=Inputs({"particles": Input(type="particle", title="Particles")}),
            outputs=Outputs(
                {
                    "particles_class_0": Output(
                        type="particle",
                        title="Particles Class 1",
                        passthrough="particles",
                        results=[
                            OutputResult(name="blob", dtype="blob"),
                            OutputResult(name="ctf", dtype="ctf"),
                            OutputResult(name="alignments3D", dtype="alignments3D"),
                        ],
                    ),
                    "volume_class_0": Output(
                        type="volume",
                        title="Volume Class 0",
                        results=[
                            OutputResult(name="map", dtype="blob"),
                        ],
                    ),
                }
            ),
            ui_tile_width=1,
            ui_tile_height=1,
            resource_spec=ResourceSpec(),
        ),
        build_errors=[],
        job_dir_size_last_updated=datetime.now(timezone.utc),
    )


@pytest.fixture
def mock_params():
    return Params(abinit_K=1, generate_intermediate_results=True, random_seed=2056920808)


@pytest.fixture
def mock_new_job_with_params(mock_new_job: Job, mock_params: Params):
    job = mock_new_job.model_copy(deep=True)
    job.spec.params = mock_params
    return job


@pytest.fixture
def mock_new_job_with_connection(mock_new_job_with_params: Job):
    job = mock_new_job_with_params.model_copy(deep=True)
    input_particles = Connection(
        job_uid="J41",
        output="particles",
        results=[
            InputResult(name="blob", dtype="blob", job_uid="J42", output="particles", result="blob"),
            InputResult(name="ctf", dtype="ctf", job_uid="J42", output="particles", result="ctf"),
            # passthrough:
            InputResult(name=None, dtype="location", job_uid="J42", output="particles", result="location"),
        ],
    )
    passthrough_result = OutputResult(name="location", dtype="location", passthrough=True)
    job.spec.inputs.root["particles"] = Input(type="particle", title="Particles", connections=[input_particles])
    job.spec.outputs.root["particles_class_0"].results.append(passthrough_result)
    return job


@pytest.fixture
def mock_job(mock_new_job_with_connection: Job):  # completed
    job = mock_new_job_with_connection.model_copy(update={"status": "completed"}, deep=True)
    # fmt: off
    output_particles_class_0 = Output(
        type="particle",
        title="Particles Class 0",
        passthrough="particles",
        results=[
            OutputResult(
                name="blob",
                dtype="blob",
                versions=[0, 100, 200, 300, 400, 863],
                metafiles=["J42/class_00_00000_particles.cs", "J42/class_00_00100_particles.cs", "J42/class_00_00200_particles.cs", "J42/class_00_00300_particles.cs", "J42/class_00_00400_particles.cs", "J42/class_00_final_particles.cs"],
                num_items=[90, 9090, 10000, 10000, 10000, 10000],
            ),
            OutputResult(
                name="ctf",
                dtype="ctf",
                versions=[0, 100, 200, 300, 400, 863],
                metafiles=["J42/class_00_00000_particles.cs", "J42/class_00_00100_particles.cs", "J42/class_00_00200_particles.cs", "J42/class_00_00300_particles.cs", "J42/class_00_00400_particles.cs", "J42/class_00_final_particles.cs"],
                num_items=[90, 9090, 10000, 10000, 10000, 10000],
            ),
            OutputResult(
                name="alignments3D",
                dtype="alignments3D",
                versions=[0, 100, 200, 300, 400, 863],
                metafiles=["J42/class_00_00000_particles.cs", "J42/class_00_00100_particles.cs", "J42/class_00_00200_particles.cs", "J42/class_00_00300_particles.cs", "J42/class_00_00400_particles.cs", "J42/class_00_final_particles.cs"],
                num_items=[90, 9090, 10000, 10000, 10000, 10000],
            ),
            OutputResult(
                name="location",
                dtype="location",
                versions=[0],
                metafiles=["J42/passthrough_particles_class_0.cs"],
                num_items=[10000],
                passthrough=True,
            ),
        ],
        num_items=10000,
    )
    output_volume_class_0 = Output(
        type="volume",
        title="Volume Class 0",
        results=[
            OutputResult(
                name="map",
                dtype="blob",
                versions=[0, 100, 200, 300, 400, 862],
                metafiles=["J42/class_00_00000_volume.cs", "J42/class_00_00100_volume.cs", "J42/class_00_00200_volume.cs", "J42/class_00_00300_volume.cs", "J42/class_00_00400_volume.cs", "J42/class_00_final_volume.cs"],
                num_items=[1, 1, 1, 1, 1, 1],
            )
        ],
        num_items=1,
    )
    # fmt: on
    job.spec.outputs.root["particles_class_0"] = output_particles_class_0
    job.spec.outputs.root["volume_class_0"] = output_volume_class_0
    return job


@pytest.fixture
def job(cs: CryoSPARC, project: ProjectController, mock_job: Job):
    APIClient.jobs.find_one.return_value = mock_job  # type: ignore
    return project.find_job("J42")

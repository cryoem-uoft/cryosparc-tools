# Changelog

## v4.4.1

- Fixed: Correct command response error data formatting

## v4.4.0

- Added: Python 3.11 and 3.12 support
- Added: `overwrite` keyword argument for file upload functions for extra confirmation when uploading project files to a path that already exist (raises error if not specified)
- Added: Smaller and faster compressed dataset format when saving with `CSDAT_FORMAT` (datasets saved in this format prior to this release may no longer be readable)
- Updated: Calls to CryoSPARC’s command servers automatically retry while CryoSPARC is down for up to one minute
- Updated: Raise `cryosparc.errors.CommandError` instead of `AssertionError` when a CryoSPARC command server request fails
- Updated: Raise `cryosparc.errors.DatasetLoadError` to show target file path when `Dataset.load` encounters an exception
- Updated: Improved server- and slot- related error messages
- Updated: Warn when connecting to CryoSPARC version that doesn’t match cryosparc-tools version
- Docs: Delete Rejected Exposures example
- Docs: Instructions for plotting scale bars on 2D Classes
- Docs: Revert downsampled, symmetry expanded particles example
- Docs: Connect a volume series to Class3D example

## v4.3.1

- Fixed: `Job.queue` method `lane` argument is now optional when queuing directly to master
- Fixed: Dataset memory leak due to missing reference count decrease during dataset garbage collection
- Docs: pip update instructions
- Docs: Recommend Python environment as a prerequisite

## v4.3.0

- Added: Updated project doc spec with updated generate intermediate results settings field
- Fixed: Allow project file retrieval from when project directory contains a symbolic link
- Fixed: Auto-exclude slots with missing metafile and better error message when output metafile is missing
- Fixed: Extend `CommandClient.Error` from `Exception`
- Fixed: Correctly catch command request timeout
- Docs: Added T20S workflow example

## v4.2.0

- Support for CryoSPARC v4.2

## v4.1.3

- Fix error when creating a job with `CryoSPARC.create_job`
- Fix Dataset allocation error due to slot generation overflow
- Docs minor fixes

## v4.1.2

- Preliminary access to CryoSPARC Live via CryoSPARC.rtp command client
- Documenation fixes

## v4.1.1

- Use correct numpy object type for newer versions of Numpy
- Fix limit on number of active datasets
- Use correct C types in Cython header definition

## v4.1.0

- Initial release
- Manage CryoSPARC Projects, Workspaces and Jobs from Python
- Load input and output datasets
- Read and write MRC files
- Read and write STAR files

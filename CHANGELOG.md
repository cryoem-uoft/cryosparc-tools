# Changelog

## v5.0.1

- Fixed: Corretly use order argument when listing workspaces

## v5.0.0

- BREAKING: replaced low-level `CryoSPARC.cli`, `CryoSPARC.rtp` and `CryoSPARC.vis` attributes with single unified `CryoSPARC.api`
- BREAKING: Can no longer add inputs or outputs when an external job has completed status, first clear the job with `job.clear()`
- BREAKING: Can no longer save outputs when an external job has building or completed status, first clear the job, call `job.start()`, save outputs, and call `job.stop()`
- BREAKING: `CryoSPARC.download_asset(fileid, target)` no longer accepts a directory target. Must specify a filename.
- BREAKING: removed `CryoSPARC.get_job_specs()`. Use `CryoSPARC.job_register` instead
- BREAKING: `CryoSPARC.list_assets()` and `Job.list_assets()` return list of models instead of list of dictionaries, accessible with dot-notation
  - OLD: `job.list_assets()[0]['filename']`
  - NEW: `job.list_assets()[0].filename`
- BREAKING: `CryoSPARC.get_lanes()` now returns a list of models instead of dictionaries
  - OLD: `cs.get_lanes()[0]['name']`
  - NEW: `cs.get_lanes()[0].name`
- BREAKING: `CryoSPARC.get_targets` now returns a list of models instead of dictionaries
  - OLD: `cs.get_targets()[0]['hostname']`
  - NEW: `cs.get_targets()[0].hostname`
  - Some top-level target attributes have also been moved into the `.config` attribute
- BREAKING: `CryoSPARC.print_job_types` `section` argument renamed to `category`
  - OLD: `cs.print_job_types(section=["extraction", "refinement"])`
  - NEW: `cs.print_job_types(category=["extraction", "refinement"])`
- BREAKING: Restructured schema for Job models, many Job.doc fields have been internally rearranged. Some fields like `params_spec` and `output_result_groups` are no longer available (replaced with a unified `spec` field)
- Added: Python 3.14 support
- Added: Secure command-line login with shell command `python -m cryosparc.tools login --url <URL>`, to avoid exposing login credentials in plain text
- Added: Option to initialize `CryoSPARC` instance with a URL, e.g., `cs = CryoSPARC("https://cryosparc.example.com:61000")`
- Added: `find_projects()`, `find_workspaces()` and `find_jobs()` methods to find lists of Projects, Workspaces and Jobs; filters are available for attributes like project UID and creation date
- Added: `CryoSPARC.job_register` attribute
- Added: New `title` and `desc` attributes for Project, Workspace and Job objects
- Added: New `status` attribute for Job objects
- Added: Methods to update title and description for Projects, Workspaces and Jobs
- Added: Ability to set External job tile and output images
- Added: `job.load_input()` and `job.load_output()` now accept `"default"`, `"passthrough"` and `"all"` keywords for their `slots` argument
- Added: `job.alloc_output()` now accepts `dtype_params` argument for fields with dynamic shapes
- Added: `CryoSPARC.print_job_types` now includes a job stability column
- Added: `Job.print_output_spec` now includes a passthrough indicator column for results
- Added: `Job.log()` now accepts a `name` argument to create or update the same event log
- Updated: Improved type definitions
- Updated: `cryosparc.mrc` always converts float16 arrays to float32 before writing
- Updated: Access to `BASE_PORT + 2`, `BASE_PORT + 3` and `BASE_PORT + 5` no longer required, only `BASE_PORT` or the web URL is required
- Fixed: Prevent error on some datasets with many columns when using Numpy >= 1.24
- Fixed: Querying a dataset with a non-existent field now correctly raises a KeyError
- Deprecated: When adding external inputs and outputs, expanded slot definitions now expect `"name"` key instead of `"prefix"`, support for which will be removed in a future release.
  - OLD: `job.add_input("particle", slots=[{"prefix": "component_mode_1", "dtype": "component", "required": True}])`
  - NEW: `job.add_input("particle", slots=[{"name": "component_mode_1", "dtype": "component", "required": True}])`
- Deprecated: `license` argument no longer required when creating a `CryoSPARC`
  instance, will be removed in a future release
- Deprecated: `external_job.stop()` now expects optional error string instead of boolean, support for boolean errors will be removed in a future release
- Deprecated: `CryoSPARC.get_job_sections()` will be removed in a future release,
  use `CryoSPARC.job_register` instead
- Deprecated: Most functions no longer require a `refresh` argument, including `job.set_param()`, `job.connect()`, `job.disconnect()` and `external_job.save_output()`
- Deprecated: Attributes `Project.doc`, `Workspace.doc` and `Job.doc` will be removed in a future release, use `.model` attribute instead
- Deprecated: `project.dir()` and `job.dir()` methods will be removed in a future release, use `.dir` attribute instead

## v4.7.1

- Added: `job.connect_result` and `job.disconnect_result` for connecting and disconnecting low-level results

## v4.7.0

- Added: Datasets appear as tables in a Jupyter notebook without requiring pandas or similar
- Updated: Dropped official support for Python 3.7
- Fixed: Prevent dataset load or save error when dataset is empty
- Fixed: Prevent dataset load error on Linux when dataset file is exactly 4 kiB

## v4.6.1

- Added: Python 3.13 support
- Fix: Disallow non-URL-encodable characters when creating external job inputs and outputs
- Fix: Prevent queuing and killing external jobs (must use `job.start/job.stop()` or `with job.run()`)

## v4.6.0

- Added: `Dataset.is_equivalent` method to check if two datasets have identical fields, but in a different order
- Added: `Dataset.inspect` class method which accepts a path to a dataset and returns its header without loading the entire dataset from disk
- Added: Load a subset of dataset fields with `Dataset.load` by specifying `prefixes` or `fields` keyword arguments
- Added: Support for uploading job assets in `bild` format
- Updated: `ExternalJob.add_output` method now expects `alloc` argument to be specified as a keyword arg
- Updated: Dataset methods which accept a `copy` argument now expect it to be specified as a keyword arg
- Fixed: Significantly reduced memory when loading large datasets with `Dataset.load`

## v4.5.1

- Added: Numpy 2.0 support
- Fixed: Add missing result data types added in CryoSPARC v4.5

## v4.5.0

- Added: New `cluster_vars` argument for `Job.queue` method for queueing to clusters
- Added: Various `print` methods for inspecting available job types, inputs, outputs and parameters
- Updated: Allow copy and symlink operations from outside project directory
- Updated: Allow larger shapes for dataset fields
- Fixed: Always raise explicit errors when dataset allocation fails
- Fixed: Correctly retain fields when appending or creating a union of empty datasets with non-empty fields
- Fixed: Correctly allow excluding `uid` field in `Dataset.descr` method
- Docs: New guide for job access and inspection
- Docs: Fix broken code in custom latent trajectory example
- Docs: Generalize 3D Flex custom trajectory example to >2 dimensions

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

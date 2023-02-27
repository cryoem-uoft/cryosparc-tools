# Changelog

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

__version__ = "4.1.0"


def get_include():
    """
    Get the include directory for the ``<cryosparc-tools/dataset.h>`` header
    file to access CryoSPARC dataset handles from other languages.

    Returns:
        str: Include path for C compilers.
    """
    from pathlib import Path

    return str(Path(__file__).parent / "include")

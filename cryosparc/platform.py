import os
import sys
from functools import lru_cache
from pathlib import Path
from typing import Callable

__all__ = ("user_config_path",)


def user_config_path() -> Path:
    """
    config directory tied to the user, e.g. ``~/.config`` or ``$XDG_CONFIG_HOME``
    """
    if sys.platform == "win32":
        path = os.path.normpath(get_win_appdata_folder())
    else:
        path = os.environ.get("XDG_CONFIG_HOME", "")
        if not path.strip():
            path = os.path.expanduser("~/.config")
    return Path(path)


# Windows platform directory code code adapted from platformdirs package
# https://github.com/tox-dev/platformdirs/blob/4.3.6/src/platformdirs/windows.py
# MIT License
# Copyright (c) 2010-202x The platformdirs developers
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


def get_win_appdata_folder_from_env_vars(csidl_name: str) -> str:
    """Get folder from environment variables."""
    env_var_name = "APPDATA"
    result = os.environ.get(env_var_name)
    if result is None:
        raise ValueError(f"Unset environment variable: {env_var_name}")
    return result


def get_win_appdata_folder_from_registry(csidl_name: str) -> str:
    """
    Get folder from the registry.

    This is a fallback technique at best. I'm not sure if using the registry for these guarantees us the correct answer
    for all CSIDL_* names.

    """
    shell_folder_name = "AppData"
    if sys.platform != "win32":
        # only needed for type checker to know that this code runs only on Windows
        raise NotImplementedError

    import winreg

    key = winreg.OpenKey(winreg.HKEY_CURRENT_USER, r"Software\Microsoft\Windows\CurrentVersion\Explorer\Shell Folders")
    directory, _ = winreg.QueryValueEx(key, shell_folder_name)
    return str(directory)


def get_win_appdata_folder_via_ctypes(csidl_name: str) -> str:
    """Get folder with ctypes."""
    import ctypes

    csidl_const = 26  # for APPDATA
    buf = ctypes.create_unicode_buffer(1024)
    windll = getattr(ctypes, "windll")
    windll.shell32.SHGetFolderPathW(None, csidl_const, None, 0, buf)

    # Downgrade to short path name if it has high-bit chars.
    if any(ord(c) > 255 for c in buf):
        buf2 = ctypes.create_unicode_buffer(1024)
        if windll.kernel32.GetShortPathNameW(buf.value, buf2, 1024):
            buf = buf2
    return buf.value


def _pick_get_win_appdata_folder() -> Callable[[str], str]:
    try:
        import ctypes
    except ImportError:
        pass
    else:
        if hasattr(ctypes, "windll"):
            return get_win_appdata_folder_via_ctypes
    try:
        import winreg  # noqa: F401
    except ImportError:
        return get_win_appdata_folder_from_env_vars
    else:
        return get_win_appdata_folder_from_registry


get_win_appdata_folder = lru_cache(maxsize=None)(_pick_get_win_appdata_folder())

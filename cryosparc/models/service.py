# THIS FILE IS AUTO-GENERATED, DO NOT EDIT DIRECTLY
# SEE dev/api_generate_models.py
from typing import Literal

LoggingService = Literal[
    "app", "database", "cache", "api", "scheduler", "command_vis", "app_api", "supervisord", "update"
]
"""
Same as Service, but also includes supervisord and update logs.
"""

ServiceLogLevel = Literal["ERROR", "WARNING", "INFO", "DEBUG"]
"""
Possible values for event log messages used in CLI functions that write to
them.
"""

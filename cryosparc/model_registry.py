# THIS FILE IS AUTO-GENERATED, DO NOT EDIT DIRECTLY
# SEE dev/api_generate_models.py
from .models import (
    api_request,
    api_response,
    auth,
    diagnostics,
    event,
    exposure,
    file,
    gpu,
    instance,
    job,
    job_register,
    job_spec,
    license,
    notification,
    project,
    scheduler_lane,
    scheduler_target,
    service,
    session,
    session_config_profile,
    session_params,
    session_spec,
    signature,
    tag,
    user,
    workspace,
)
from .registry import register_model_module

register_model_module(session)
register_model_module(job)
register_model_module(scheduler_target)
register_model_module(gpu)
register_model_module(api_request)
register_model_module(api_response)
register_model_module(job_spec)
register_model_module(exposure)
register_model_module(event)
register_model_module(user)
register_model_module(session_params)
register_model_module(project)
register_model_module(file)
register_model_module(signature)
register_model_module(instance)
register_model_module(job_register)
register_model_module(license)
register_model_module(service)
register_model_module(notification)
register_model_module(diagnostics)
register_model_module(scheduler_lane)
register_model_module(workspace)
register_model_module(session_spec)
register_model_module(session_config_profile)
register_model_module(tag)
register_model_module(auth)
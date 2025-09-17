# THIS FILE IS AUTO-GENERATED, DO NOT EDIT DIRECTLY
# SEE dev/api_generate_models.py
from .models import (
    api_request,
    api_response,
    asset,
    auth,
    benchmarks,
    config,
    diagnostics,
    event,
    exposure,
    external,
    file_browser,
    gpu,
    instance,
    job,
    job_register,
    job_spec,
    license,
    notification,
    params,
    preview,
    project,
    resource,
    scheduler_lane,
    scheduler_target,
    services,
    session,
    session_config_profile,
    session_params,
    session_spec,
    signature,
    tag,
    user,
    when,
    workspace,
)
from .registry import register_model_module

register_model_module(job, include_literals=True)
register_model_module(scheduler_target, include_literals=True)
register_model_module(resource, include_literals=True)
register_model_module(gpu, include_literals=True)
register_model_module(api_request, include_literals=True)
register_model_module(session, include_literals=True)
register_model_module(user, include_literals=True)
register_model_module(file_browser, include_literals=True)
register_model_module(job_spec, include_literals=True)
register_model_module(exposure, include_literals=True)
register_model_module(event, include_literals=True)
register_model_module(preview, include_literals=True)
register_model_module(session_params, include_literals=True)
register_model_module(external, include_literals=True)
register_model_module(project, include_literals=True)
register_model_module(api_response, include_literals=True)
register_model_module(asset, include_literals=True)
register_model_module(signature, include_literals=True)
register_model_module(instance, include_literals=True)
register_model_module(workspace, include_literals=True)
register_model_module(job_register, include_literals=True)
register_model_module(when, include_literals=True)
register_model_module(params, include_literals=True)
register_model_module(license, include_literals=True)
register_model_module(services, include_literals=True)
register_model_module(notification, include_literals=True)
register_model_module(benchmarks, include_literals=True)
register_model_module(diagnostics, include_literals=True)
register_model_module(scheduler_lane, include_literals=True)
register_model_module(session_spec, include_literals=True)
register_model_module(session_config_profile, include_literals=True)
register_model_module(config, include_literals=True)
register_model_module(tag, include_literals=True)
register_model_module(auth, include_literals=True)

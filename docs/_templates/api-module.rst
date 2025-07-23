{{ name | escape | underline}}

.. automodule:: {{ fullname }}
    :autosummary:
    :members:
    :exclude-members: model_config

{%- block modules %}
{%- if modules %}
.. rubric:: Modules

.. autosummary::
    :toctree:
    :template: api-module.rst
    :recursive:

{% for item in modules %}
    {{ item }}
{%- endfor %}
{% endif %}
{%- endblock %}

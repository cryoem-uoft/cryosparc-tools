{{ name | escape | underline}}

.. automodule:: {{ fullname }}
    :autosummary:
    :members:

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

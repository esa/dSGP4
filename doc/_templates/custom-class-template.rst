{{ fullname | escape | underline }}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}
   :members:
   :special-members: __init__, __call__

.. rubric:: Methods

.. autosummary::
   {% for item in members %}
      {% if item not in parent_members %}
         ~{{ name }}.{{ item }}
      {% endif %}
   {% endfor %}

.. rubric:: Attributes

.. autosummary::
   {% for item in attributes %}
      {% if item not in parent_attributes %}
         ~{{ name }}.{{ item }}
      {% endif %}
   {% endfor %}
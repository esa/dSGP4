{{ fullname | escape | underline}}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}
   :members:
   :special-members: __init__, __call__

   {% block methods %}
   {# We go through this circumlocution because __call__ for some reason     #}
   {# does not show up in the list of methods, only in the list of members.  #}
   {# If we do not do this, the full documentation of __call__ will show up, #}
   {# but its entry in autosummary will not.                                 #}
   {% if methods or '__call__' in members %}
   .. rubric:: {{ _('Methods') }}

   .. autosummary::
   {% for item in members %}
      {% if item in methods or item == '__call__' %}
         ~{{ name }}.{{ item }}
      {% endif %}
   {%- endfor %}
   {% endif %}
   {% endblock %}

   {% block attributes %}
   {% if attributes %}
   .. rubric:: {{ _('Attributes') }}

   .. autosummary::
   {% for item in attributes %}
      ~{{ name }}.{{ item }}
   {%- endfor %}
   {% endif %}
   {% endblock %}

---
title: Notes
layout: archive
permalink: /notes/
author_profile: true
entries_layout: list
---

{% assign notes = site.posts | where_exp: 'p', 'p.categories contains "note"' %}
<p><strong>총 개수:</strong> {{ notes | size }}</p>

{% assign notes = notes | sort: 'date' | reverse %}

{% if notes and notes.size > 0 %}
  {% for post in notes %}
    {% include archive-single.html %}
  {% endfor %}
{% else %}
  <p>No notes yet. Add <code>categories: [note]</code> to a post.</p>
{% endif %}

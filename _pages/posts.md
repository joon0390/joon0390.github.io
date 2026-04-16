---
title: "Study"
layout: archive
permalink: /posts/
author_profile: false
entries_layout: grid
sidebar:
  nav: "sidebar-category"
---

{% assign study_posts = "" | split: "" %}
{% for post in site.posts %}
  {% unless post.categories contains 'Blog' or post.categories contains 'Life' or post.categories contains 'note' or post.categories contains 'Note' or post.categories contains 'Dev' or post.categories contains 'Troubleshooting' or post.categories contains 'Implementation' or post.categories contains 'Code' %}
    {% assign study_posts = study_posts | push: post %}
  {% endunless %}
{% endfor %}

{% assign study_posts = study_posts | sort: "date" | reverse %}
{% assign entries_layout = page.entries_layout | default: "grid" %}

<div class="entries-{{ entries_layout }}">
  <div class="grid__wrapper">
    {% include documents-collection.html entries=study_posts type=entries_layout %}
  </div>
</div>

---
title: "Study"
layout: archive
permalink: /posts/
author_profile: true
sidebar:
  nav: "sidebar-category"
---

{% assign study_posts = "" | split: "" %}
{% for post in site.posts %}
  {% unless post.categories contains 'Blog' or post.categories contains 'Life' or post.categories contains 'note' or post.categories contains 'Note' %}
    {% assign study_posts = study_posts | push: post %}
  {% endunless %}
{% endfor %}

<div class="entries-list">
  {% for post in study_posts %}
    {% include archive-single.html type="list" %}
  {% endfor %}
</div>

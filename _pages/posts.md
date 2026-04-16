---
title: "Study"
layout: archive
permalink: /posts/
author_profile: false
entries_layout: grid
classes:
  - wide
  - study-page
---

{% assign study_posts = "" | split: "" %}
{% assign ordered_categories = "Bayesian|Statistics|Machine Learning|Deep Learning|Graph|Applied Topics|Time Series|Computation|Optimization" | split: "|" %}
{% for post in site.posts %}
  {% unless post.categories contains 'Blog' or post.categories contains 'Life' or post.categories contains 'note' or post.categories contains 'Note' or post.categories contains 'Dev' or post.categories contains 'Troubleshooting' or post.categories contains 'Implementation' or post.categories contains 'Code' %}
    {% assign study_posts = study_posts | push: post %}
  {% endunless %}
{% endfor %}

{% assign study_posts = study_posts | sort: "date" | reverse %}
{% assign entries_layout = page.entries_layout | default: "grid" %}
{% assign study_post_count = study_posts | size %}

<section class="study-filter-bar" data-study-filters>
  <div class="study-filter-bar__header">
    <p class="study-filter-bar__eyebrow">Reading Tracks</p>
    <p class="study-filter-bar__status" data-study-filter-status>All {{ study_post_count }} posts</p>
  </div>

  <div class="study-filter-bar__chips" role="toolbar" aria-label="Study category filters">
    <button class="study-filter-bar__chip is-active" type="button" data-study-filter="all" aria-pressed="true">
      All
      <span class="study-filter-bar__count">{{ study_post_count }}</span>
    </button>

    {% for category_name in ordered_categories %}
      {% assign category_posts = site.categories[category_name] %}
      {% assign category_size = category_posts | size | default: 0 %}
      {% if category_size > 0 %}
        <button class="study-filter-bar__chip" type="button" data-study-filter="{{ category_name | slugify }}" aria-pressed="false">
          {{ category_name }}
          <span class="study-filter-bar__count">{{ category_size }}</span>
        </button>
      {% endif %}
    {% endfor %}
  </div>
</section>

<div class="entries-{{ entries_layout }}">
  <div class="grid__wrapper study-grid">
    {% for post in study_posts %}
      {% include study-card.html %}
    {% endfor %}
  </div>
</div>

<script src="{{ '/assets/js/study-filters.js' | relative_url }}"></script>

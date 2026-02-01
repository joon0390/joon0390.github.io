---
title: "Notes"
layout: archive
permalink: /notes/
author_profile: true
sidebar:
  nav: "sidebar-category"
---

간단한 메모와 유용한 링크들을 모아두는 공간입니다.

<div class="entries-list">
  {% assign note_posts = site.categories.note | concat: site.categories.Note | uniq | sort: 'date' | reverse %}
  
  {% for post in note_posts %}
    {% include archive-single.html type="list" %}
  {% endfor %}
</div>

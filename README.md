# Blog Workflow Notes

이 문서는 이 블로그에서 앞으로 글을 쓸 때 필요한 운영 메모를 정리한 파일입니다.

## Preview

- 로컬 실행: `bundle exec jekyll serve`
- 빌드 확인: `bundle exec jekyll build`
- `_config.yml`은 자동 재로딩되지 않으므로 바꾸면 서버를 재시작해야 합니다.
- 현재 `_config.yml`의 `incremental`은 `false`입니다. 메뉴명이나 레이아웃 변경이 일부 페이지만 반영되는 문제를 피하기 위해 꺼둔 상태입니다.

## Section Rules

- `Study`
  - 페이지: `/posts/`
  - 파일: [_pages/posts.md](/Users/heekim/Desktop/heekimjun/joon0390.github.io/_pages/posts.md)
  - 현재 구조: 상단 `Reading Tracks` filter chips + 카드 grid
  - `Blog`, `Life`, `note`, `Note`, `Dev`, `Troubleshooting`, `Implementation`, `Code` 카테고리는 여기서 제외됩니다.

- `Uncertain Space`
  - 페이지: `/dev-notes/`
  - 파일: [_pages/blog.md](/Users/heekim/Desktop/heekimjun/joon0390.github.io/_pages/blog.md)
  - 화면 제목만 `Uncertain Space`이고 URL은 `/dev-notes/` 그대로 유지합니다.
  - 아래 카테고리 중 하나라도 포함하면 이 페이지에 잡힙니다.
    - `Dev`
    - `Troubleshooting`
    - `Implementation`
    - `Code`

- `Category`
  - 페이지: `/category-archive/`
  - 상단에 `Graph Map`이 있습니다.
  - 카테고리 노드 이미지는 [_data/category_map_visuals.yml](/Users/heekim/Desktop/heekimjun/joon0390.github.io/_data/category_map_visuals.yml) 에서 관리합니다.

## Writing A New Study Post

기본 예시는 아래 형식으로 맞추면 됩니다.

```yaml
---
title: "Post Title"
date: 2026-04-17
categories:
  - Statistics
  - Bayesian
tags:
  - variational inference
  - predictive inference
excerpt: "홈/Study 카드에서 보여줄 짧은 설명"
header:
  teaser: /assets/img/post-slug/teaser.png
---
```

규칙:

- `categories`는 `Study`/`Uncertain Space` 노출 위치를 결정합니다.
- `excerpt`는 카드 설명, 검색, 일부 소개 문구에 그대로 쓰입니다.
- `header.teaser`가 있으면 `Study` 카드와 일반 grid 카드에서 이미지 썸네일로 사용됩니다.
- `header.teaser`가 없으면 `Study`에서는 카테고리 기반 placeholder가 대신 보입니다.
- 현재 전역 fallback teaser는 비어 있습니다. 즉, 개별 `header.teaser`가 없으면 일반 archive grid에서는 이미지가 안 나올 수 있습니다.

## Teaser Rules

- 추천 경로: `/assets/img/<post-slug>/teaser.png`
- 추천 비율: `16:9`
- 추천 크기: `1200x675`
- 권장 내용: 핵심 figure 하나를 중앙에 두기

예시:

```yaml
header:
  teaser: /assets/img/pvi/teaser.png
```

권장 방식:

- 글마다 대표 figure를 한 장 정해서 teaser로 사용
- 중요한 도형이나 텍스트가 이미지 가장자리에 붙지 않게 만들기
- 가능한 한 같은 비율로 저장해서 카드 인상이 일정하게 보이게 만들기

## Study Page Filters

- `Study` 상단 chip은 [_pages/posts.md](/Users/heekim/Desktop/heekimjun/joon0390.github.io/_pages/posts.md) 의 `ordered_categories` 순서를 따릅니다.
- 새 카테고리를 실제로 filter chip에 보이게 하려면 아래를 함께 수정해야 합니다.
  - [_pages/posts.md](/Users/heekim/Desktop/heekimjun/joon0390.github.io/_pages/posts.md) 의 `ordered_categories`
  - [assets/css/main.scss](/Users/heekim/Desktop/heekimjun/joon0390.github.io/assets/css/main.scss) 의 `study-card__placeholder--<slug>`
  - [assets/css/main.scss](/Users/heekim/Desktop/heekimjun/joon0390.github.io/assets/css/main.scss) 의 `study-card__category--<slug>`

## Home Page

- 홈 갤러리는 [_data/home_posterior_gallery.yml](/Users/heekim/Desktop/heekimjun/joon0390.github.io/_data/home_posterior_gallery.yml) 로 수동 관리합니다.
- 구조:
  - `title`
  - `url`
  - `image`
  - `alt`
  - `label`
  - `description`
- 홈의 `Recent Posts`는 최대 4개만 노출됩니다.

## Category Graph Map

- 카테고리 허브 노드 이미지는 [_data/category_map_visuals.yml](/Users/heekim/Desktop/heekimjun/joon0390.github.io/_data/category_map_visuals.yml) 에서 관리합니다.
- 카테고리 이미지를 바꾸려면 `image`와 `image_position`을 수정하면 됩니다.
- 맵 배경 클릭 시 기본 상태로 리셋됩니다.

## Current UI Decisions

- `Study`는 왼쪽 author sidebar를 쓰지 않습니다.
- `Study`는 텍스트 목록 대신 카드/grid를 씁니다.
- `Study`는 카테고리 목록 대신 상단 filter chips를 씁니다.
- `Uncertain Space`는 이름만 바꿨고 permalink는 `/dev-notes/`를 유지합니다.
- `Category`는 지도형 인터랙션, `Study`는 읽기 트랙형 인터랙션으로 역할을 분리했습니다.

## Before Push

- `bundle exec jekyll build`가 통과하는지 확인
- 메뉴명이나 `_config.yml`을 바꿨으면 `jekyll serve` 재시작
- 새 글이면 아래 3개 확인
  - `categories`
  - `excerpt`
  - `header.teaser`

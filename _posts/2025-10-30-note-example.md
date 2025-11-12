---
layout: single
title: "Note: Gaussian Process Reminder"
date: 2025-10-30
categories:
  - note
tags:
  - gaussian-process
  - memo
toc: false
author_profile: false
---

짧은 메모 형식의 포스트 예시입니다.

## 오늘의 요약

- 커널 행렬 $K(X, X)$ 는 입력의 쌍별 유사도를 표현한다.
- 노이즈 항을 포함하는 관측 공분산은 $K(X, X) + \sigma_n^2 I$.
- 테스트 포인트 $x_\*$ 의 예측 분포는
  $$
  p(f_\* \mid X, y, x_\*) = \mathcal{N}\left(k_\*^\top K_y^{-1} y,\; k_{\*\*} - k_\*^\top K_y^{-1} k_\*\right).
  $$

## 다음에 살펴볼 것

1. Matern 커널 하이퍼파라미터 감각 익히기  
2. Sparse GP 구현 비교 (FITC vs. VFE)  

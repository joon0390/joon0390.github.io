---
title: "강화학습을 이용한 조난자 이동 경로 예측"
excerpt: "강화학습 기반 의사결정 모델로 조난자의 이동 경로를 예측하는 프로젝트."
collection: projects
layout: single
order: 3
tags:
  - Reinforcement Learning
  - Path Prediction
  - DQN
  - Geographic Information System
  - Sequential Decision Making
---

## 프로젝트 개요

재난 상황이나 실종 상황에서 조난자가 이동할 가능성이 높은 경로를 예측하기 위한 프로젝트입니다. 지형 정보와 이동 제약 조건을 반영해 경로 선택 문제를 강화학습 기반 순차 의사결정 문제로 다루는 방향으로 구성했습니다.

## 핵심 내용

- 후보 경로 집합과 지형 정보를 상태로 정의해 다음 이동 경로를 예측합니다.
- 보상 함수를 통해 실제 탐색 효율에 도움이 되는 경로 선택 전략을 학습합니다.
- 구조 시뮬레이션과 연계해 탐색 우선순위 설정 문제로 확장할 수 있도록 설계합니다.

## 사용 기술

- Reinforcement Learning
- DQN / Sequential Decision Making
- GIS Data
- Route Optimization
- Simulation

## 프로젝트 의의

정적인 경로 추천이 아니라, 상황에 따라 달라지는 이동 가능성을 정책 학습으로 다룬다는 점이 특징입니다. 재난 대응, 구조 탐색, 이동 예측 문제를 AI 기반 의사결정 모델로 연결한 사례로 정리할 수 있습니다.

---
title: "대조학습을 이용한 주행 패턴 이상 탐지"
excerpt: "대조학습 기반 표현 학습으로 주행 시계열의 비정상 패턴을 탐지하는 프로젝트."
collection: projects
layout: single
order: 2
tags:
  - Contrastive Learning
  - Self-Supervised Learning
  - Driving Pattern Detection
  - Time Series
  - Change Point Detection
---

## 프로젝트 개요

운행 중 수집되는 시계열 센서 데이터를 이용해 정상 주행과 이상 주행 패턴을 구분하는 프로젝트입니다. 라벨이 제한적인 환경을 고려해 대조학습 기반 표현 학습을 적용하고, 다운스트림 이상 탐지 성능을 높이는 방향으로 설계했습니다.

## 핵심 내용

- 주행 시계열을 표현 공간으로 변환해 정상 패턴과 이상 패턴 간 거리를 학습합니다.
- 센서 신호를 스펙트럼 기반 특징으로 변환해 시계열 변화를 더 잘 반영하도록 구성합니다.
- 표현 학습 결과를 활용해 이상 상황 탐지와 변화 구간 식별 문제로 확장합니다.

## 사용 기술

- Contrastive Learning
- Self-Supervised Learning
- Spectrogram / STFT
- Time Series Classification
- Change Point Detection

## 프로젝트 의의

지도학습만으로는 구축하기 어려운 주행 데이터 문제를 자기지도학습 관점에서 풀어낸 프로젝트입니다. 실제 도로 환경에서 발생하는 비정상 패턴을 표현 학습으로 다룬다는 점에서 연구성과와 응용 가능성을 함께 정리하기 좋습니다.

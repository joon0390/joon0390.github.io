---
title: "Contrastive Learning 기반 주행패턴 분석"
excerpt: "스마트폰 센서를 활용한 운전패턴 변화 탐지 및 분류 시스템 개발"
collection: projects
layout: posts
date: 2024-02-23
tags:
  - Self-supervised Learning
  - Contrastive Learning
  - Driving Pattern Detection
  - Change Point Detection
  - Smartphone Sensors
---

## 🧠 프로젝트 개요

이 프로젝트는 **대조학습(Contrastive Learning)** 기반의 방법론을 활용하여, **스마트폰 센서 데이터로부터 운전자의 주행 패턴을 분석**하고, 운전 중 발생하는 **이상 행동이나 변화점을 자동으로 탐지**하는 시스템을 개발한 연구입니다.

레이블링이 어려운 운전 데이터를 효율적으로 처리하기 위해 **자기지도학습(Self-supervised Learning)**을 기반으로 하였고, 이후 적은 양의 레이블만으로 분류까지 수행할 수 있는 **하이브리드 학습 방식**을 채택했습니다.

---

## 🔍 문제 정의

- 기존 운전자 행동 인식 시스템은 **대량의 레이블링 데이터**를 요구하거나 **정해진 규칙 기반**으로 동작하여 일반화에 한계가 존재함
- **스마트폰 센서** (가속도, 자이로 등)는 운전 중 변화 감지에 유용하지만 이를 효과적으로 활용한 연구는 부족
- 특히 **운전 이벤트의 변화점 탐지(change point detection)** 와 **패턴 분류**를 동시에 고려한 시스템은 드뭄

---

## 🛠️ 사용 기술 및 접근법

- **센서 데이터 처리**: STFT(Short-Time Fourier Transform)를 이용한 시간-주파수 변환
- **1단계 - 변화점 탐지**:
  - `Time-Consistency` 기반 대조학습 학습 구조
  - InfoNCE Loss + Cosine Similarity 사용
- **2단계 - 분류기 학습**:
  - 사전학습된 딥러닝 모델의 feature를 기반으로 SVC 분류기 학습
  - 소량의 레이블로도 높은 성능 확보

---

## 📱 경량 모델 적용 및 비교

| Model        | Params | GFLOPs |
|--------------|--------|--------|
| SqueezeNet   | 1.2M   | 0.35   |
| ShuffleNet   | 1.4M   | 0.04   |
| RegNet       | 2.5M   | 0.06   |
| MobileNet    | 3.5M   | 0.3    |
| EfficientNet | 5.3M   | 0.39   |
| MnasNet      | 2.2M   | 0.10   |

→ **SqueezeNet**, **RegNet** 모델은 변화점 탐지 및 분류 모두에서 매우 우수한 성능

---

## 🧪 실험 결과 요약

- 변화점 탐지 정확도 (Precision 기준):  
  - RegNet: 0.92  
  - ShuffleNet: 0.73  
  - EfficientNet: 0.67
- 운전 패턴 분류 정확도:  
  - SqueezeNet: **Precision = 1.00, Recall = 1.00**
  - ShuffleNet, RegNet: Precision/Recall > 0.85

→ 특히 클래스당 30개 미만의 매우 소량 레이블 데이터만으로도 뛰어난 성능을 달성함

---

## 📌 기여 및 역할: 김준희 (Inha University, Dept. of Statistics)

본 프로젝트에서 저는 **데이터셋 설계, 모델링, 실험을 담당하여 수행**하였으며, 다음과 같은 기술적 기여를 했습니다:

### 🧩 대조학습 기반 변화점 탐지 모델 설계
- **Time Consistency** 기반의 Positive/Negative Pair 구성 전략 수립
- **InfoNCE Loss + Cosine Similarity** 를 활용한 대조학습 프레임워크 구현
- 임계값(0.0, 0.4, 0.6)에 따른 변화점 탐지 실험 및 정밀도 분석 수행

### 🌀 STFT 기반 데이터 전처리 및 입력 구조 설계
- **STFT 변환**을 통해 시간-주파수 정보를 추출하는 파이프라인 설계
- 비정규 간격 센서 데이터를 스플라인 보간하여 정규화 처리

### ⚙️ 경량화 모델 적용 및 하이퍼파라미터 튜닝
- 6종 경량 모델(SqueezeNet, ShuffleNet 등)에 대해 특징 추출기 구조 조정
- 전체 학습 설정 튜닝 (Epoch 200, Batch 32, FFT 32, Adam Optimizer)

### 🧠 분류기 설계 및 전이학습 적용
- 사전학습된 특징 추출기 + **SVC(Support Vector Classifier)** 기반 분류 구조 구현
- 학습/테스트 차량 분리 실험을 통해 **모델 일반화 성능 평가**

### 📊 실험 결과 분석 및 논문 보고서 작성
- 각 모델별 변화점 탐지 및 분류 결과 시각화 및 정량 평가
- SqueezeNet이 정밀도/재현율 1.0을 달성한 실험 해석 및 고찰 정리

---

## 📄 참고자료

- 📘 논문: [한국 ITS 학회, Vol.23 No.1 (2024)]  
- 🔗 DOI: [https://doi.org/10.12815/kits.2024.23.1.182](https://doi.org/10.12815/kits.2024.23.1.182)

---


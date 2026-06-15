---
title: "도메인 지식을 반영한 합성 커널 가우시안 프로세스 분류기를 이용한 분자 독성 예측과 원리 기반 불확실성 정량화"
excerpt: "분자 구조 정보와 도메인 지식을 반영한 합성 커널로 독성 분류와 불확실성 추정을 수행한 연구."
date: 2026-05-21
collection: projects
layout: single
order: 6
classes:
  - wide
tags:
  - Gaussian Process
  - Kernel Methods
  - Molecular Toxicity
  - Uncertainty Quantification
  - Bayesian ML
---

## 프로젝트 요약

- 개요: SMILES 기반 분자 구조와 화학 도메인 특징을 합성 커널로 결합해 독성 확률과 불확실성을 함께 예측한 연구
- 기간: 2026.04-
- 데이터: SMILES, Morgan fingerprint, RDKit descriptor, PCA 기반 연속 특징, 독성 라벨
- 기술 스택: RDKit, Gaussian Process Classifier, Tanimoto/RBF/Matern Kernel, Composite Kernel, Calibration Analysis
- 성과(성능): Hybrid GP 기준 `F1 0.803`, `AUC 0.877`, `Brier 0.147`로 비교 모델 대비 가장 균형 잡힌 성능 달성

## 문제 정의

분자 독성 예측은 단순히 독성 여부를 맞히는 문제로 끝나지 않습니다. 실제 후보 물질 선별에서는 모델이 얼마나 확신하는지, 학습 데이터와 구조적으로 먼 분자에서 예측이 불안정한지, 화학적 직관과 어긋나는 판단이 있는지를 함께 봐야 합니다.

이 프로젝트는 블랙박스 분류기 하나로 독성 라벨을 맞히는 대신, 분자 구조 유사도와 도메인 기반 연속 특징을 커널 수준에서 결합했습니다. 이를 Gaussian Process Classifier에 연결해 독성 확률뿐 아니라 예측 불확실성까지 함께 제공하는 구조로 설계했습니다.

<figure class="project-figure project-figure--medium">
  <img src="/assets/img/projects/domain-informed-molecular-toxicity-gpc/arch.png" alt="도메인 지식 기반 분자 독성 예측 합성 커널 아키텍처">
  <figcaption>SMILES에서 Morgan fingerprint와 RDKit descriptor를 추출하고, 구조 유사도와 연속 특징 커널을 합성해 Gaussian Process Classifier에 입력하는 전체 구조입니다.</figcaption>
</figure>

## 데이터와 EDA

데이터는 분자의 SMILES 표현과 독성 라벨을 기본 단위로 두고, 구조 기반 이진 fingerprint와 연속 descriptor를 함께 사용했습니다. EDA에서는 독성/비독성 클래스 분포, Morgan fingerprint 기반 구조 유사도, RDKit descriptor의 스케일과 상관 구조, PCA 이후 연속 특징 공간에서의 분리 가능성을 중심으로 확인했습니다.

특히 독성 예측에서는 학습 데이터와 구조적으로 멀리 떨어진 분자에 대해 모델이 과도하게 확신하면 위험합니다. 따라서 모델 성능은 F1이나 AUC만으로 보지 않고, Brier score와 커널 조합별 calibration 흐름까지 같이 보도록 구성했습니다.

## 접근 방법

핵심 접근은 하나의 범용 커널에 의존하지 않고, 서로 다른 관점의 유사도를 합성하는 것입니다. Tanimoto kernel은 Morgan fingerprint 기반 구조 유사도를 반영하고, RBF/Matern kernel은 RDKit descriptor와 PCA 기반 연속 특징 공간의 거리 구조를 반영합니다.

1. SMILES에서 Morgan fingerprint를 추출해 분자 구조의 부분 구조 유사도를 표현합니다.
2. RDKit descriptor를 계산하고 PCA로 압축해 연속적인 물성 기반 특징 공간을 구성합니다.
3. Tanimoto, RBF, Matern kernel을 조합해 구조 유사도와 도메인 특징을 동시에 반영합니다.
4. 합성 커널을 Gaussian Process Classifier에 연결해 독성 확률과 posterior uncertainty를 함께 추정합니다.

<figure class="project-figure project-figure--medium">
  <img src="/assets/img/projects/domain-informed-molecular-toxicity-gpc/kernels.png" alt="분자 독성 예측 합성 커널 조합별 F1, Brier, AUC 비교">
  <figcaption>커널 조합을 비교한 결과, Tanimoto + RBF + Matern 조합이 F1, AUC, Brier score의 균형에서 가장 좋은 결과를 보였습니다.</figcaption>
</figure>

## 성과(성능)

- 제안한 Hybrid GP는 `F1 0.803`, `AUC 0.877`, `Brier 0.147`을 기록했습니다.
- 단일 Tanimoto kernel 기반 GP는 `F1 0.763`, `AUC 0.863`, `Brier 0.165` 수준이었고, RBF 단일 커널은 `F1 0.709`, `AUC 0.798`, `Brier 0.231`로 상대적으로 낮았습니다.
- LDA, QDA, Naive Bayes 같은 전통적 분류기와 비교해도 Hybrid GP가 F1 기준 가장 높은 성능을 보였습니다.
- 성능 개선의 핵심은 fingerprint 구조 유사도와 descriptor 기반 연속 특징을 하나의 확률적 커널 모델 안에서 함께 다룬 점입니다.

<div class="project-figure-grid">
  <figure class="project-figure">
    <img src="/assets/img/projects/domain-informed-molecular-toxicity-gpc/AUC.png" alt="커널 조합별 AUC 비교">
    <figcaption>독성 분류의 ranking 성능은 Tanimoto 기반 조합에서 전반적으로 높게 나타났고, 세 커널을 모두 결합한 모델이 가장 높은 AUC를 기록했습니다.</figcaption>
  </figure>
  <figure class="project-figure">
    <img src="/assets/img/projects/domain-informed-molecular-toxicity-gpc/f1_model.png" alt="모델별 F1 성능 비교">
    <figcaption>Hybrid GP는 전통적 분류기와 단일 커널 GP를 모두 넘어서며, 독성/비독성 분류 균형을 가장 잘 맞췄습니다.</figcaption>
  </figure>
</div>

## 느낀점

이 프로젝트를 하면서 가장 크게 느낀 점은, 분자 독성 예측에서는 모델의 종류보다 `분자를 어떤 관점에서 비슷하다고 볼 것인가`가 훨씬 중요하다는 점이었습니다. 처음에는 Gaussian Process Classifier라는 모델 자체에 더 관심이 갔지만, 실험을 진행할수록 성능 차이는 커널을 어떻게 설계하고 어떤 분자 표현을 결합하느냐에서 크게 갈렸습니다.

특히 Tanimoto kernel은 분자 fingerprint의 구조적 유사성을 잘 반영했지만, 그것만으로는 설명되지 않는 연속적인 물성 정보도 분명히 있었습니다. 그래서 RDKit descriptor와 PCA 기반 특징을 RBF, Matern kernel로 함께 다루는 방식이 필요했고, 이 조합이 단일 커널보다 더 나은 결과를 만들었습니다.

또 하나 배운 점은 독성 예측처럼 실제 의사결정과 연결될 수 있는 문제에서는 정확도만 높아도 충분하지 않다는 점입니다. 모델이 어떤 샘플에서 확신하는지, 어떤 샘플에서 불확실한지, 그리고 그 확률이 얼마나 잘 보정되어 있는지를 같이 봐야 했습니다. 이 과정에서 성능 지표와 calibration, 불확실성 해석을 함께 놓고 모델을 평가하는 습관이 중요하다는 것을 다시 느꼈습니다.

## 논문/자료

- 원문 PDF: [Domain-informed Molecular Toxicity GPC PDF 열기](/assets/papers/domain-informed-molecular-toxicity-gpc.pdf)

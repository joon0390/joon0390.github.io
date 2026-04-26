---
title: "베이지안 딥커널 머신을 이용한 양식 넙치 성장 예측"
excerpt: "수온, 용존산소, 사료량을 바탕으로 넙치 성장량을 예측하고 BDKMR로 기존 커널 모델보다 더 낮은 오차를 달성한 연구."
date: 2026-04-18
collection: projects
layout: single
order: 4
header:
  teaser: /assets/img/projects/flounder/pipeline.svg
tags:
  - Bayesian Machine Learning
  - Deep Kernel Learning
  - Gaussian Process
  - Aquaculture
  - Growth Prediction
---

<div class="project-paper-layout">
<div class="project-paper-layout__content" markdown="1">

## 프로젝트 개요

국내 양식 넙치(olive flounder)의 성장 예측을 위해, 가우시안 프로세스 회귀와 신경망 기반 표현 학습을 결합한 Bayesian Deep Kernel Machine Regression (BDKMR) 모델을 제안한 연구입니다. 완도 2개 양식장과 제주 3개 양식장, 총 7개 수조에서 2023년 3월부터 2024년 7월까지 수집한 종단 데이터를 바탕으로 수온, 용존산소, 개체당 사료량이 성장에 미치는 비선형 관계를 모델링했습니다.

## 시각 자료

프로젝트 상단에는 연구 범위, 입력 데이터, 모델 파이프라인, 핵심 성능을 한 장으로 요약한 도식을 배치했습니다. 아래 차트는 논문에 보고된 LOOCV 성능 비교를 시각적으로 정리한 것입니다.

{% include figure image_path="/assets/img/projects/flounder/pipeline.svg" alt="Study overview for flatfish growth prediction with BDKMR" class="project-figure project-figure--wide" caption="연구 범위, 입력 변수, 모델 파이프라인, 핵심 결과를 한 장으로 정리한 요약 도식." %}

{% include figure image_path="/assets/img/projects/flounder/performance.svg" alt="LOOCV performance comparison across KRR, BKMR, BDKMR Equal, and BDKMR" class="project-figure project-figure--medium" caption="LOOCV 기준 성능 비교. BDKMR이 KRR, BKMR, BDKMR(Equal)보다 더 낮은 MAE와 MSE를 기록했습니다." %}

## 핵심 내용

- 수온과 용존산소는 1분 단위 센서 데이터, 사료량은 일 단위 기록, 체중은 월 단위 측정으로 수집되었으며, 이를 동일한 성장 관측 구간에 맞춰 정렬해 분석용 데이터셋을 구성했습니다.
- 월별 체중 측정에서는 농가별로 무작위 50마리를 표본 추출해 로그 평균 체중을 반응변수로 사용했고, 개체 수와 측정 변동성을 반영하기 위해 `Var(y_i) = \sigma^2 / n_i` 형태의 이분산 구조를 적용했습니다.
- BKMR의 불확실성 정량화 장점과 ANN의 표현 학습 능력을 결합한 BDKMR을 설계해, 환경 변수 간 복잡한 비선형 상호작용을 더 유연하게 학습하도록 구성했습니다.
- 추론은 MAP 추정과 Laplace approximation을 기반으로 수행해 베이지안 구조를 유지하면서도 실제 예측 문제에 적용 가능한 계산 효율을 확보했습니다.

## 데이터 및 실험 설계

- 대상 데이터: 완도 2개 양식장, 제주 3개 양식장, 총 7개 수조
- 수집 기간: 2023년 3월부터 2024년 7월까지
- 입력 변수: 수온, 용존산소, 개체당 사료량, 초기 로그 체중
- 반응 변수: 월별 로그 평균 체중
- 비교 모델: KRR, BKMR, BDKMR(Equal), BDKMR
- 평가 지표: Leave-One-Out Cross-Validation (LOOCV), MAE, MSE

## 주요 결과

- 제안한 BDKMR은 `MAE 0.1895`, `MSE 0.0629`를 기록해 비교 모델 중 가장 낮은 예측 오차를 보였습니다.
- 동분산 가정의 `BDKMR(Equal)`보다 이분산 구조를 반영한 BDKMR이 더 우수해, 관측별 정밀도 차이를 고려하는 것이 실제 양식 데이터 예측에 중요함을 확인했습니다.
- 기준 모델인 KRR은 `MAE 1.1141`, `MSE 3.5665`, BKMR은 `MAE 0.6977`, `MSE 0.9447`로 나타나, 딥커널 기반 표현 학습이 기존 커널 모델 대비 뚜렷한 성능 개선을 제공했습니다.

## 사용 기술

- Bayesian Deep Kernel Machine Regression
- Gaussian Process Regression
- Artificial Neural Networks
- Heteroscedastic Modeling
- Leave-One-Out Cross-Validation

## 프로젝트 의의

양식 데이터는 변수별 측정 주기가 다르고 환경 스트레스에 따라 변동성이 크게 달라지는 특성이 있습니다. 이 연구는 딥러닝의 표현 학습과 베이지안 커널 모델의 해석 가능성 및 불확실성 추정을 결합해, 급이 전략, 출하 시점, 환경 제어와 같은 실제 양식 운영 의사결정에 활용할 수 있는 성장 예측 프레임워크를 제시했다는 점에서 의미가 있습니다.

## 논문 정보

- Junhee Kim, Seung-Won Seo, Ho-Jin Jung, Hyun-Seok Jang, Han-Kyu Lim, Seongil Jo, "Predicting Flatfish Growth in Aquaculture Using Bayesian Deep Kernel Machines", *Applied Sciences*, 2025.
- DOI: [10.3390/app15179487](https://doi.org/10.3390/app15179487)

</div>

<aside class="project-paper-layout__viewer">
  <div class="project-paper-layout__viewer-inner">
    <p class="project-paper-layout__eyebrow">Paper Viewer</p>
    <p class="project-paper-layout__title">Applied Sciences 2025 Full Paper</p>
    <div class="project-paper-layout__frame">
      <iframe
        src="{{ '/assets/papers/flounder-bdkmr-applsci-2025.pdf' | relative_url }}#view=FitH"
        title="Predicting Flatfish Growth in Aquaculture Using Bayesian Deep Kernel Machines PDF"
      ></iframe>
    </div>
    <p class="project-paper-layout__note">
      브라우저에서 PDF 임베드가 지원되지 않으면
      <a href="{{ '/assets/papers/flounder-bdkmr-applsci-2025.pdf' | relative_url }}" target="_blank" rel="noopener">새 탭에서 논문 열기</a>
    </p>
  </div>
</aside>
</div>

---
title: "베이지안 최적화를 이용한 One-Class Classification"
excerpt: "불균형 데이터에서 정상/이상 패턴을 구분하기 위해 OC-SVM과 Deep SVDD의 하이퍼파라미터를 Grid Search, Random Search, Bayesian Optimization으로 비교한 이상 탐지 연구."
date: 2026-04-30
collection: projects
layout: single
order: 8
period: "2024.06-2026.04"
header:
  teaser: /assets/img/projects/one-class-classification-bayesian-optimization/OC-SVM.png
teaser_alt: "OC-SVM 기반 정상 영역 경계 학습 개념"
tags:
  - One-Class Classification
  - Bayesian Optimization
  - Anomaly Detection
  - OC-SVM
  - Deep SVDD
  - Gaussian Process
---

## 프로젝트 요약

- 개요: 정상 데이터가 대부분이고 이상 데이터가 적은 상황에서 One-Class Classification 모델의 하이퍼파라미터 탐색 방법을 비교한 프로젝트
- 기간: 2024.06-2026.04
- 데이터: synthetic dataset, 포병 군사훈련 산불 위험 데이터, Extended Cohn-Kanade(CK+) 감정 데이터
- 기술 스택: Python, OC-SVM, Deep SVDD, Bayesian Optimization, Gaussian Process, Grid Search, Random Search
- 성과(성능): synthetic dataset 실험에서 Bayesian Optimization이 대부분의 불균형 비율에서 F1과 AUC 기준 가장 안정적인 성능을 보임

## 문제 정의

One-Class Classification(OCC)은 정상 데이터는 충분하지만 이상 데이터는 적거나 불완전한 상황에서 중요한 이상 탐지 방법입니다. 실제 현장에서는 정상 상태 데이터만 많이 쌓이고, 고장, 위험, 비정상 행동 같은 이상 사례는 적게 관측되는 경우가 많습니다. 이때 일반적인 지도학습 분류기를 그대로 쓰면 클래스 불균형과 이상 데이터 부족 때문에 안정적인 모델을 만들기 어렵습니다.

이 프로젝트는 OCC 모델 자체보다 `하이퍼파라미터를 어떻게 찾을 것인가`에 초점을 둔 연구입니다. 특히 OC-SVM과 Deep SVDD는 모델 구조가 비교적 명확하더라도 `nu`, `gamma`, 네트워크 설정, 학습 조건 같은 하이퍼파라미터에 따라 성능 차이가 크게 나타날 수 있습니다. 따라서 Grid Search, Random Search, Bayesian Optimization을 비교해 어떤 탐색 전략이 불균형 이상 탐지 문제에서 더 효과적인지 확인했습니다.

<figure class="project-figure project-figure--narrow">
  <img src="/assets/img/projects/one-class-classification-bayesian-optimization/class_imbalance.png" alt="정상 데이터가 다수이고 이상 데이터가 소수인 클래스 불균형 예시">
  <figcaption>One-Class Classification에서 다루는 클래스 불균형 상황</figcaption>
</figure>

## 데이터와 EDA

실험은 synthetic dataset과 실제 데이터 응용을 함께 고려했습니다. synthetic dataset에서는 정상:이상 비율을 `95:5`, `90:10`, `80:20`, `70:30`, `60:40`으로 바꾸며 불균형 정도가 달라질 때 탐색 방법별 성능이 어떻게 변하는지 확인했습니다. train/test 비율은 `8:2`로 유지했습니다.

실제 응용 데이터로는 포병 군사훈련 상황에서의 산불 위험 데이터와 Extended Cohn-Kanade(CK+) 감정 데이터를 다뤘습니다. 두 데이터는 도메인은 다르지만, 정상 패턴과 드문 이상 패턴을 구분해야 한다는 점에서 OCC 문제로 해석할 수 있습니다.

EDA에서는 단순히 클래스 비율만 보는 것이 아니라, 이상 데이터가 정상 데이터 주변에 섞여 있는지, 특정 변수 공간에서 분리 가능한 구조가 있는지, 불균형 비율이 커질수록 recall과 precision 사이의 균형이 어떻게 흔들리는지를 확인하는 것이 중요했습니다.

## 접근 방법

비교 대상은 세 가지 하이퍼파라미터 탐색 전략입니다.

1. `Grid Search`: 미리 정한 후보 격자를 모두 탐색하는 방식
2. `Random Search`: 후보 공간에서 무작위로 조합을 선택해 탐색하는 방식
3. `Bayesian Optimization`: Gaussian Process 기반 surrogate model로 성능 함수를 근사하고, 다음 탐색 지점을 순차적으로 선택하는 방식

Grid Search는 해석이 쉽지만 후보 공간이 조금만 커져도 계산량이 빠르게 증가합니다. Random Search는 넓은 공간을 가볍게 훑을 수 있지만, 이전 실험 결과를 다음 탐색에 충분히 활용하지 못합니다. 반면 Bayesian Optimization은 이전 평가 결과를 바탕으로 성능이 좋을 가능성이 높은 영역과 아직 불확실한 영역을 균형 있게 탐색할 수 있습니다.

모델 관점에서는 OC-SVM과 Deep SVDD를 중심으로 비교했습니다. OC-SVM은 정상 데이터가 놓이는 경계를 kernel 기반으로 학습하고, Deep SVDD는 딥러닝 표현 공간에서 정상 데이터를 하나의 중심 주변으로 모으는 방식입니다. 두 방법 모두 정상 패턴의 경계를 어떻게 잡느냐가 핵심이며, 이 경계는 하이퍼파라미터 설정에 매우 민감합니다.

<figure class="project-figure project-figure--medium">
  <img src="/assets/img/projects/one-class-classification-bayesian-optimization/OC-SVM.png" alt="OC-SVM이 정상 데이터 영역의 경계를 학습하고 바깥쪽 샘플을 이상으로 판단하는 개념도">
  <figcaption>OC-SVM 기반 정상 영역 경계 학습 개념</figcaption>
</figure>

## 성과(성능)

synthetic dataset에서는 Bayesian Optimization이 F1과 AUC 기준으로 일관되게 강한 결과를 보였습니다.

| 정상:이상 비율 | F1 Grid | F1 Random | F1 BO | AUC Grid | AUC Random | AUC BO |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 95:5 | 0.7076 | 0.6814 | **0.7510** | 0.8567 | 0.8514 | **0.8620** |
| 90:10 | 0.7656 | 0.7488 | **0.7923** | 0.8536 | 0.8482 | **0.8588** |
| 80:20 | 0.8425 | 0.8302 | **0.8562** | 0.8795 | 0.8746 | **0.8821** |
| 70:30 | 0.8580 | 0.8499 | **0.8713** | 0.8735 | 0.8694 | **0.8807** |
| 60:40 | 0.8689 | 0.8614 | **0.8793** | 0.8754 | 0.8698 | **0.8832** |

특히 정상:이상 비율이 `95:5`처럼 더 불균형한 경우에도 Bayesian Optimization은 F1과 AUC에서 좋은 성능을 유지했습니다. 이는 이상 데이터가 적은 환경에서 단순한 격자 탐색이나 무작위 탐색보다, 이전 실험 정보를 활용하는 순차적 최적화가 더 효율적으로 작동할 수 있음을 보여줍니다.

다만 recall만 보면 일부 비율에서는 Grid Search나 Random Search가 더 높게 나타나는 경우도 있었습니다. 따라서 최종 판단은 recall 하나가 아니라 F1, AUC, 클래스 불균형 상황에서의 오탐/미탐 비용을 함께 고려해야 합니다.

## 느낀점

이 프로젝트를 진행하면서 가장 크게 느낀 점은, 이상 탐지에서는 모델 선택만큼이나 하이퍼파라미터 탐색 전략이 중요하다는 점이었습니다. 특히 OCC처럼 이상 데이터가 적은 문제에서는 작은 설정 차이로도 decision boundary가 크게 바뀌고, 그 결과 recall이나 F1이 눈에 띄게 흔들릴 수 있었습니다.

또한 Bayesian Optimization은 단순히 성능이 좋은 자동 튜닝 도구라기보다, 비싼 실험을 줄이면서도 의미 있는 후보를 순차적으로 찾아가는 방식이라는 점이 인상적이었습니다. Grid Search처럼 모든 조합을 기계적으로 확인하지 않아도 되고, Random Search처럼 이전 결과를 버리지도 않기 때문에, 제한된 실험 예산 안에서 더 합리적인 선택을 할 수 있었습니다.

개인적으로는 이 프로젝트를 통해 이상 탐지 문제를 평가할 때 accuracy보다 F1, AUC, recall의 의미를 더 조심스럽게 봐야 한다는 점을 배웠습니다. 이상 데이터가 적을수록 모델이 무엇을 놓치고 무엇을 과하게 잡아내는지가 중요해지기 때문에, 성능표를 해석할 때도 도메인에서의 비용 구조를 함께 생각해야 한다고 느꼈습니다.

## 참고자료

- GitHub: [joon0390/One-Class-Classification-Using-Bayesian-Optimization](https://github.com/joon0390/One-Class-Classification-Using-Bayesian-Optimization)

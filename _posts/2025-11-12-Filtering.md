---
layout: single
title: "Filtering & State Space Model"
date: 2025-11-12
permalink: /filtering/
categories:
  - Statistics
  - Bayesian
tags:
  - Filtering
  - State Space Model
  - Bayesian Filtering
  - Kalman Filter
toc: true
toc_sticky: true
comments: true
---

> 이 포스팅은 동적 시스템(Dynamic System)에서의 **필터링(Filtering)** 과 **상태공간모형(State-Space Model)** 에 대한 기본 개념을 정리한 글입니다.


## Introduction

현실 세계의 많은 문제는 **시간에 따라 변화하는 시스템**을 다루고 있으며, 이 시스템의 내부 상태는 직접적으로 관찰하기 어려운 경우가 많습니다.

예를 들어,

- GPS가 끊길 때 로봇의 **실제 위치**는 관측되지 않지만 지속적으로 추정해야 합니다.  
- 센서 노이즈로 인해 **실제 신호 값**을 직접적으로 보기 어렵습니다.  
- 금융 시계열에서는 시장의 **숨겨진 변동성(latent volatility)** 을 추정하는 것이 핵심입니다.

이처럼 **관찰할 수 없는 상태(hidden state)를 관측 가능한 데이터로부터 추정**하는 문제가 바로 **필터링(filtering)** 입니다.


## 필터링(Filtering)이란 무엇인가?

필터링은 다음 분포를 추정하는 문제입니다:

$$
p(x_t \mid y_{1:t})
$$

여기서

- $x_t$: 현재 시간의 숨겨진 상태(hidden state)  
- $y_{1:t}$: 과거부터 현재까지의 누적된 관측 데이터

즉, 지금까지 들어온 관측치들을 바탕으로 **현재의 은닉 상태가 어떤 값일지** 확률적으로 추정하는 것이 필터링입니다.

### ✔ Filtering, Prediction, Smoothing의 차이

| 문제 | 목표 | 조건부 분포 |
|------|--------|-------------|
| **Prediction** | 미래 상태 예측 | $ p(x_{t+1} \mid y_{1:t}) $ |
| **Filtering** | 현재 상태 추정 | $ p(x_t \mid y_{1:t}) $ |
| **Smoothing** | 과거 상태 재추정 | $ p(x_t \mid y_{1:T}) $ |

필터링은 “**online**” 문제이며, smoothing은 “**offline**” 문제입니다.


## State-Space Model (SSM)

필터링을 정의하려면, 시스템이 어떻게 변화하고 관측되는지 모델링해야 합니다.  
가장 일반적인 틀이 바로 **상태공간모형(State-Space Model)** 입니다.

SSM은 두 부분으로 구성됩니다.


### 1) 상태전이모형 (State Transition Model)

$$
x_t \sim p(x_t \mid x_{t-1})
$$

- 시스템이 **시간에 따라 어떻게 변화**하는지 나타냄  
- ex: 로봇의 위치 변화, 금융 시장의 변동성 변화 등


### 2) 관측모형 (Observation Model)

$$
y_t \sim p(y_t \mid x_t)
$$

- 은닉 상태 $x_t$가 어떻게 관측치 $y_t$를 만들어내는지 표현  
- ex: 센서 노이즈 모델, 측정 에러 등


### SSM의 직관적 이해

- **진짜 세계가 $x_t$** (하지만 우리는 못 본다)
- **관측되는 데이터는 $y_t$** (노이즈 포함)
- 우리는 $y_t$로부터 다시 $x_t$를 역추론해야 한다

필터링이 어떻게 이루어지는지 이해하기 위해 필요한 개념은 단 두 가지입니다.

### ✔ 1) Prediction (과거 posterior를 이용해 현재 prior 생성)

$$
p(x_t \mid y_{1:t-1})
=
\int p(x_t \mid x_{t-1})
\ p(x_{t-1} \mid y_{1:t-1})
\ dx_{t-1}
$$


### ✔ 2) Update (관측을 반영해 posterior 생성)

$$
p(x_t \mid y_{1:t})
\propto
p(y_t \mid x_t)\,
p(x_t \mid y_{1:t-1})
$$


이 두 식이 **모든 필터링 알고리즘의 기반**입니다.  
이 식을 **정확하게 계산할 수 있느냐 없느냐**에 따라 알고리즘이 크게 달라집니다.



## 해석적(analytic) 풀이가 가능한 경우: Kalman Filter

특수한 경우에 위 식들이 정확히(Closed-form solution이 존재) 계산 가능합니다.

### ✔ 선형 transition + 관측

$$
x_t = A x_{t-1} + \epsilon_t,\quad
y_t = C x_t + \eta_t
$$

### ✔ Gaussian 잡음

$$
\epsilon_t \sim \mathcal{N}(0,Q),
\quad
\eta_t \sim \mathcal{N}(0,R)
$$


### 그러면 무슨 일이 생기나?

- posterior가 항상 Gaussian 형태 유지  
- 평균·분산만 업데이트하면 됨  
- 필터링을 **closed-form**으로 계산 가능  

그래서 Kalman Filter는

- 항공/우주 항법  
- 자율주행  
- 센서 융합(핸드폰 IMU, GPS)  
- 공학 제어 시스템  

등에서 주로 사용됩니다.



## 하지만 현실 시스템은 대부분 더 복잡하다

실제 문제는 대체로:

- Nonlinear
- Non-Gaussian
- multi-modal
- Contain outlier

example:

- 로봇 이동에 삼각함수 포함  
- 금융 노이즈는 heavy-tailed  
- 뇌 latent dynamics는 복잡한 비선형 구조  
- 센서의 이상치는 가우시안 가정을 깨뜨림

이 경우 Kalman Filter는 더 이상 정확한 추정을 보장할 수 없습니다.


## EKF와 UKF: 비선형 근사 접근

비선형 시스템을 다루기 위해 다음 두 확장판이 도입되었습니다.


### 1) Extended Kalman Filter (EKF)

- 비선형 함수 Transition과 Observation $f, g$ 를 Jacobian을 이용하여 일차 테일러 근사를 통해 모델을 강제로 선형 근사합니다.
- 비교적 간단하지만, 강하게 비선형이면 부정확할 수 있습니다.


### 2) Unscented Kalman Filter (UKF)

- $f,g$를 선형화하는 것 대신, 분포를 선형화합니다. 
    - Unscented Transform(UT) 방법을사용하여 posterior 분포를 대표하는 sigma points를 생성하고 이 포인트들을 비선형 함수에 통과시켜 새로운 분포를 얻습니다. 
- EKF보다 안정적이며, 비선형에 더 강하고, 2차 정확도 수준의 근사 품질을 갖지만,
- **여전히 Gaussian 형태의 posterior를 가정**합니다.


## 자연스러운 결론: 더 일반화된 필터가 필요하다

여기까지 요약하면,

- Kalman 계열은 특정 상황(선형+정규)에선 매우 강력  
- 하지만 현실 데이터가 복잡하면 가정이 쉽게 깨짐  
- 그 결과 **posterior가 어떤 모양이든 유연하게 근사할 수 있는 방법**이 필요하다.

그것 중 하나가 **Particle Filter (Sequential Monte Carlo)** 입니다.


# 다음 편: Particle Filter (Sequential Monte Carlo)

다음 글에서는 다음 내용을 다룰 예정입니다.

- Particle Filter의 기본 아이디어  
- SIR (Sampling–Importance–Resampling) 알고리즘  
- Degeneracy & ESS  
- Auxiliary Particle Filter  
- Particle Learning  
- SMC²  

그리고 필터링의 Prediction–Update 구조가  
어떻게 **Monte Carlo 기반의 샘플 집합(particles)** 으로 구현되는지 설명합니다.


## Reference

- Sarkka, S. (2013). *Bayesian Filtering and Smoothing*.  
- Doucet, de Freitas, Gordon (2001). *Sequential Monte Carlo Methods in Practice*.  
- Ristic et al. (2004). *Beyond the Kalman Filter*.  
- Kevin P. Murphy (2023). *Probabilistic Machine Learning*.  

---
layout: single
title: "Particle Filter : Sequential Monte Carlo"
date: 2025-11-14
permalink: /particle-filter/
categories:
  - Statistics
  - Bayesian
  - Time Series
  - Machine Learning
tags:
  - Particle Filter
  - SMC
  - Sequential Monte Carlo
  - Importance Sampling
  - Filtering

toc: true
toc_sticky: true
comments: true
---

> 이 포스팅은 **Sequential Monte Carlo(SMC)** 방법의 대표 알고리즘인 **Particle Filter** 의 핵심 개념과 구조를 정리한 글입니다. 

> [“Filtering & State Space Model”](/filtering/) 에 이어지는 내용입니다.

---

## Introduction

앞선 글에서는 Filtering이 무엇인지,  
그리고 선형·정규성 가정하에서 Kalman Filter가 어떻게 closed-form 해를 제공하는지 정리했었습니다.

하지만 대부분의 실제 시스템은

- **비선형(nonlinear)**  
- **비가우시안(non-Gaussian)**  
- **multi-modal posterior**  
- **outlier 포함**  
- **복잡한 likelihood**

을 가지기 때문에 칼만 계열(EKF/UKF 포함)만으로는 정확한 추정이 어렵습니다.

이 문제를 해결하기 위한 가장 일반적인 방법이 바로 **Particle Filter(입자 필터)** 입니다.

Particle Filter는 **posterior 분포 자체를 Monte Carlo method로 직접 근사하는 방법**이며, 필터링 문제에 대한 **가장 유연한 범용 해법**이라고 할 수 있습니다.



## Particle Filter의 핵심 아이디어

Particle Filter는 posterior 분포 $p(x_t \mid y_{1:t})$를 **일련의 샘플(particles)** 과 **가중치(weights)** 의 조합으로 근사합니다.

즉,

$$
p(x_t \mid y_{1:t}) \approx
\sum_{i=1}^N w_t^{(i)} \, \delta(x_t - x_t^{(i)})
$$

여기서

- $x_t^{(i)}$: i번째 particle  
- $w_t^{(i)}$: 해당 particle의 중요도 가중치  
- $\delta(\cdot)$: Dirac delta (point mass)

즉, posterior라는 복잡한 분포를 “**여러 개의 Weighted Samples**” 로 나타내는 것입니다.

핵심은 다음과 같습니다.

> **비선형이든 비정규성이든 상관없이, 샘플만 잘 뿌리면 어떤 분포든 근사할 수 있다.**

이런 관점에서 PF는 매우 강력합니다.


## SIR Particle Filter (Bootstrap Filter)

가장 기본적이면서 널리 쓰이는 PF는 **SIR (Sampling–Importance–Resampling)** 입니다.

각 단계는 다음과 같습니다.


## 1) Initialization

초기 prior로부터 particle를 샘플링:

$$
x_0^{(i)} \sim p(x_0), \qquad w_0^{(i)} = \frac{1}{N}
$$


## 2) Prediction (Sampling)

각 particle을 transition model에 따라 다음 시점으로 이동시킴:

$$
x_t^{(i)} \sim p(x_t \mid x_{t-1}^{(i)})
$$

이는 filtering 공식의 **Prediction 단계**에 해당합니다.


## 3) Update (Importance Weighting)

새로운 관측값 $y_t$가 주어지면, likelihood로 가중치를 업데이트합니다.

$$
w_t^{(i)} \propto p(y_t \mid x_t^{(i)})
$$

정규화:

$$
w_t^{(i)} =
\frac{w_t^{(i)}}{\sum_{j=1}^N w_t^{(j)}}
$$

여기까지가 filtering 공식의 **Update 단계**입니다.


## 4) Resampling (필수 단계)

문제는, 시간이 지나면 대부분의 weight가 0이 되고  
몇 개의 particle만 큰 weight를 가지는 **degeneracy issue**가 발생합니다.

이를 해결하기 위해:

- weight가 큰 particle 중심으로 다시 샘플링하고  
- weight를 모두 동일하게 만들어줍니다.

Resampling의 주요 방식:

- Multinomial  
- Systematic 
- Stratified  


## 효율성 판단 기준: ESS (Effective Sample Size)

가중치가 얼마나 collapse 되었는지 확인하기 위해:

$$
\text{ESS} = \frac{1}{\sum_{i=1}^N (w_t^{(i)})^2}
$$

- ESS가 작아지면 → 대부분 weight가 0 → resampling 필요  
- 실전에서는 보통  
  $\text{ESS} < N/2$ 또는 $N/3$ 를 기준으로 사용


# PF는 Importance Sampling 기반 방법이다

PF는 사실상 **Sequential Importance Sampling(SIS)** 과 필수적으로 따라오는 **Resampling(SIR)** 의 조합입니다.

즉,

- proposal distribution: transition model  
- target distribution: filtering posterior  
- importance weight: likelihood  

이 구조는 MCMC와도 유사한 probabilistic intuition을 제공하기 때문에, PF는 Bayesian computation 영역에서 매우 중요한 알고리즘입니다.


# 왜 PF는 강력한가?

**비선형 + 비가우시안 + multimodal posterior를 그대로 다룰 수 있기 때문입니다.**

만약 posterior가 아래처럼 두 개의 모드를 가진다면:

- EKF: 망함 (선형화 + Gaussian 강제)  
- UKF: 망함 (Gaussian 강제)  
- PF: particle가 두 모드에 각각 분포하면 그대로 표현 가능

또한 이상치(outlier)가 등장해 likelihood가 heavy-tailed일 때도 PF는 Monte Carlo 방식이기 때문에 자연스럽게 적응합니다.


# 하지만 PF도 한계가 있다

특히:

### ✔ Sample degeneracy  
weight collapse → resampling 필요  
너무 자주 resampling하면 diversity 감소

### ✔ Curse of Dimensionality  
state dimension이 커지면 particle 수가 폭발적으로 필요

그래서 등장하는 여러 개선 알고리즘들이 있습니다.


# PF의 주요 변형들

## 1) Auxiliary Particle Filter (APF)

- 관측 $y_t$ 정보를 미리 사용해 더 나은 proposal을 구성  
- weight degeneracy를 줄이는 데 효과적


## 2) Rao–Blackwellized Particle Filter (RBPF)

PF로는 일부 state만 추적하고  
다른 state는 Kalman Filter로 처리하는 hybrid 방식.

예:  
SLAM에서 landmark map → KF  
robot pose → PF


## 3) Particle Learning (PL)

posterior의 sufficient statistics를 업데이트하여  
파라미터 learning까지 동시에 수행.


## 4) SMC²

parameter + latent state 둘 다 PF로 처리하는 확장.  
Bayesian inference 전체를 Sequential하게 수행.


# 예시

단순한 비선형 시스템:

$$
x_t = x_{t-1} + 0.5\sin(x_{t-1}) + \epsilon_t,
\quad
y_t = \frac{x_t^2}{20} + \eta_t
$$

- transition에 sine → 비선형  
- observation에 square → 매우 비선형  

EKF/UKF는 대부분 실패하는 경우라고 볼 수 있습니다.

그러나 PF는 particle들이 분포의 모양을 그대로 따라가기 때문에 잘 작동합니다.


# 다음 편: Particle Filter Code Lab

다음 글에서는 Particle Filter를 코드로 실험해보고 분석해볼 예정입니다. 감사합니다.
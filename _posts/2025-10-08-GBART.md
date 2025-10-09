---
layout: single
title: "[Paper Review] Generalized Bayesian Additive Regression Trees Models: Beyond Conditional Conjugacy"
date: 2025-10-08
permalink: /gbart/
categories:
  - Machine Learning
  - Bayesian
tags:
  - Bayesian Additive Regression Tree
  - Generalized Bayesian Additiv Regression Trees
toc: true
toc_sticky: true
comments: true
---

> 이 포스팅은 **Linero (2023)** 의 논문 [*Generalized Bayesian Additive Regression Trees Models: Beyond Conditional Conjugacy*](https://arxiv.org/abs/2202.09924) 을 읽고 정리한 글입니다.  

본 논문은 기존 **BART(Bayesian Additive Regression Trees)** 가 의존하던 **공액성(conditional conjugacy)** 조건을 완전히 제거하여,  로지스틱·감마·생존모형 등 일반화된 모델에서도 **튜닝 없이 베이지안 백피팅(Bayesian Backfitting)** 을 수행할 수 있게 한 방법론을 제시합니다.

---

## Background: Classical BART and Its Limitation 
BART는 관측 데이터 $(x_i, y_i)$를 다수의 얕은 회귀트리(weak learner)로 근사하는 모델입니다. BART에 대한 자세한 내용은 [BART](/bart/) 포스팅을 참고해주세요.
$$
y_i = \sum_{t=1}^{T} g(x_i; T_t, M_t) + \varepsilon_i, \quad
\varepsilon_i \sim \mathcal{N}(0,\sigma^2)
$$

여기서  
- $T_t$: 트리 구조 (split variable, split value 등),  
- $M_t = \{\mu_{t,\ell}\}$: 각 리프(leaf)의 예측값,  
- $g(x;T,M)$: 입력 $x$가 트리의 어느 리프에 속하는지에 따라 상수값 $\mu_\ell$ 반환.

즉, BART는 여러 회귀트리를 더해 **비선형 함수를 베이지안적으로 근사**하는 모델입니다.

---

## Why Conditional Conjugacy Was Essential

각 리프에 대해 데이터 $R_i^{(t)}$가 정규분포를 따른다고 가정하면,

$$
R_i^{(t)} \sim \mathcal{N}(\mu_\ell, \sigma^2)
$$

사전분포가 $\mu_\ell \sim \mathcal{N}(0, \sigma_\mu^2)$일 때, 사후분포는 Conjugacy 덕분에 닫힌 형태로 계산됩니다.

$$
p(\mu_\ell \mid R_i^{(t)}, T_t)
= \mathcal{N}(\bar{\mu}_\ell, V_\ell)
$$

이 덕분에 다음이 가능했습니다:
1. 각 리프의 $\mu_\ell$를 analytic하게 통합 (closed form marginal likelihood)
2. 트리 구조 Grow/Prune 제안을 **Metropolis–Hastings 없이** 수락/거절
3. 매우 효율적인 MCMC 백피팅 구현

→ 이것이 BART가 단순하면서도 강력했던 이유입니다.

---

### What about Non-Gaussian?

만약 $Y_i$가 정규분포가 아니라면

- $p(Y_i \mid \mu_\ell)$이 정규가 아니므로 사후분포가 닫힌 형태가 아님  
- 트리 제안 시 필요 우도비 계산이 어려움  
- 리프별 통합우도를 수치적으로 적분해야 함  

결과적으로, 기존 BART는 **Conditional Conjugacy** 없이는 작동하지 않았습니다.
모델마다 다음과 같은 “보조장치”가 필요했습니다.

| 모델 유형 | 필요한 보조기법 |
|------------|----------------|
| 로지스틱 회귀 | Polya–Gamma 보조변수 |
| 감마 회귀 | 근사 posterior 또는 grid 적분 |
| 생존모형 (AFT) | 특별한 잠재변수 도입 |

이것이 Linero가 해결하려 한 핵심 제약입니다.

---

## Key Idea: RJMCMC on Trees + Laplace Approximation

Linero(2023)는 공액성 제약을 완전히 제거하기 위해 **RJMCMC(Reversible Jump MCMC)** 와 **라플라스(Laplace) 근사**를 결합하였습니다.

핵심은 다음 두 단계입니다:

1. 트리 구조를 RJMCMC로 갱신  
2. 리프의 사후분포를 라플라스 정규근사로 대체  

이로써 어떤 분포 $f_\eta(y \mid \lambda)$를 사용하더라도,  
Bayesian Backfitting처럼 동작하는 일반화된 BART가 완성됩니다.

---

## RJMCMC on Trees

다른 트리들의 합을 고정하고 한 트리만 갱신합니다.

$$
\lambda_i = \sum_{k \neq t} g(X_i; T_k, M_k)
$$

그러면 현재 트리 $T_t, M_t$에 대해서는

$$
Y_i \sim f_\eta(y_i \mid \lambda_i + g(X_i; T_t, M_t))
$$

이때 리프별 likelihood는 다음과 같이 factorization 됩니다.

$$
L(T, M)
= \prod_{\ell \in L(T)} \prod_{i: X_i \in \ell}
f_\eta(Y_i \mid \lambda_i + \mu_\ell)
$$

> 이 식이 기존 BART의 “통합 우도” 역할을 대체합니다.

---

## Laplace Approximation for Leaf Posterior

각 리프의 사후로그밀도는 다음과 같습니다.

$$
\log p(\mu_\ell \mid Y_\ell) = \sum_{i \in \ell} \log f_\eta(Y_i \mid \lambda_i + \mu_\ell) + \log \pi_\mu(\mu_\ell)
$$

이를 최대점 $m_\ell$ 근처에서 2차 전개하면,

$$
\log p(\mu_\ell \mid Y_\ell) \approx \log p(m_\ell \mid Y_\ell) - \frac{(\mu_\ell - m_\ell)^2}{2 v_\ell^2}
$$

따라서 리프 사후분포를 **정규분포로 근사**할 수 있게 됩니다.

$$
p(\mu_\ell \mid Y_\ell) \approx \mathcal{N}(m_\ell, v_\ell^2)
$$

이 근사분포를 RJMCMC의 제안분포로 사용하면,Metropolis–Hastings의 수락률이 급격히 향상됩니다.

---

## What the User Provides

사용자가 모델별로 제공해야 하는 것은 단 세 가지이다:

1. 로그 가능도: $ \log f_\eta(y \mid \lambda) $  
2. 점수(Gradient): $ U = \partial_\lambda \log f_\eta $  
3. 피셔 정보(Fisher Information): $ I = -\partial_\lambda U $

이 세 가지만 정의하면,  라플라스 근사를 통해 모든 비공액 모델에서 RJMCMC가 가능합니다.

---

## RJ-Bayesian Backfitting Algorithm

1. 초기화  
   $\lambda_i \leftarrow \sum_t g(X_i; T_t, M_t)$
2. 트리 $t=1,\dots,T$에 대해  
   1. 현재 트리의 기여분을 $\lambda$에서 제거  
   2. Birth / Death / Change 제안 중 하나 선택  
   3. 제안된 $(T', M')$에 대해 Laplace 기반 정규 제안  
   4. MH 비율로 수락/거절  
   5. 슬라이스 샘플링 등으로 $M_t$ 보정  
   6. 새 기여분을 $\lambda$에 더함

---

## Why It Works So Well

- **공액성 불필요:**  
  모델별 보조변수 없이, 오직 log-likelihood / gradient / Fisher 정보만으로 구현 가능  
- **튜닝 프리:**  
  Laplace 기반 제안분포로 MH 수락률이 높아 별도 조정 불필요  
- **범용성:**  
  로지스틱, 감마, AFT 생존모형 등 모든 GLM류 적용 가능  
- **MCMC 안정성:**  
  사후 근사정규 분포를 사용하므로 mixing이 향상됨  

즉, **Generalized BART (GBART)** 라 부를 수 있는 새로운 범용 프레임워크를 제시합니다.

---

## Intuitive View: Laplace Approximation = Posterior Shape Matching

MCMC의 수락확률은 다음과 같습니다.

$$
\alpha = \min\!\left(1,
\frac{p(\text{new})\, q(\text{old}|\text{new})}{
p(\text{old})\, q(\text{new}|\text{old})}
\right)
$$

여기서 제안분포 $q$가 목표분포 $p$와 비슷할수록  
수락률이 높아집니다.

라플라스 근사는 $p(\mu_\ell | Y_\ell)$를 정규로 근사하므로  
제안분포가 $p$에 거의 일치 → 수락률 증가 → mixing 개선.  
결과적으로 tuning-free MCMC가 구현됩니다.

---

## Comparison: Classical BART vs Generalized BART

| 구분 | Classical BART | Generalized BART (Linero, 2023) |
|------|----------------|----------------------------------|
| 오차 가정 | 정규분포 | 임의의 분포 가능 |
| 공액성 필요 | Yes | **No (Laplace 근사)** |
| 트리 갱신 | Gibbs (공액) | **RJMCMC (비공액)** |
| Leaf 업데이트 | Analytical | **Laplace 근사 기반 제안** |
| 사용자 입력 | 없음 | **LogLik, Gradient, Fisher Info** |
| 적용 모델 | 회귀(정규) | 로지스틱, 감마, 생존 등 |
| 튜닝 | 필요 | **튜닝 프리** |

---

## Summary

- 기존 BART는 Conditional Conjugacy 덕분에 효율적이었지만, 정규모형에만 한정되는 한계가 있었습니다.  
- Linero(2023)는 RJMCMC와 Laplace Approximation을 결합해 이 제약을 제거하였습니다.  
- 이제 **Likelihood + Gradient + Fisher Info** 세 가지 정보만으로 어떤 분포형 데이터에서도 베이지안 백피팅이 가능합니다.  
- Generalized BART(GBART)는 기존 BART보다 훨씬 광범위한 모델링을 튜닝 없이 수행할 수 있는 새로운 베이지안 트리 프레임워크입니다.

---

## References

- Linero, A. (2023). [*Generalized Bayesian Additive Regression Trees Models: Beyond Conditional Conjugacy*](https://arxiv.org/abs/2202.09924). *arXiv preprint arXiv:2202.09924*.  
- Chipman, H. A., George, E. I., & McCulloch, R. E. (2010). [*BART: Bayesian Additive Regression Trees*](https://doi.org/10.1214/09-AOAS285). *Annals of Applied Statistics*, 4(1), 266–298.  

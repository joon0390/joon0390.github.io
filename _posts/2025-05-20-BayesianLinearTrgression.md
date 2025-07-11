---
layout: single
title: "Bayesian Linear Regression"
date: 2025-05-20
permalink: /BLR/
categories:
  - Statistics
  - Machine Learning

tags :
   - Bayesian
   - Regression
   - Linear Regression

toc: true
toc_sticky: true
---

> 이 포스팅은 Bayesian Linear Regression에 대한 개념을 소개하는 글입니다.

---

## Introduction

선형 회귀(Linear Regression)는 관측된 입력 변수 $X\in\mathbb{R}^{n\times p}$와 응답 변수 $y\in\mathbb{R}^n$ 사이의 관계를 직선(또는 초평면)으로 모델링하는 기법입니다. 전통적인 빈도주의 최소제곱(LSE) 접근법은 파라미터를 고정된 값으로 추정하지만, 베이지안 관점에서는 파라미터를 확률변수로 취급하여 불확실성을 자연스럽게 반영할 수 있습니다.

---

## Linear Regression

전통적 선형 회귀 모델은

$$y = X\beta + \varepsilon$$

여기서

- $X = [x_1,\,x_2,\dots,\,x_n]^T\in\mathbb{R}^{n\times p}$는 $p$개의 설명변수를 가진 $n$개 샘플  
- $\beta\in\mathbb{R}^p$는 회귀 계수 벡터  
- $\varepsilon\sim N(0,\,\sigma^2 I)$는 가우시안 오차  

를 가정합니다. 빈도주의적 해석에서는 잔차 제곱합(Residual Sum of Squares, RSS)을 최소화하여

$$\hat{\beta}_{\mathrm{OLS}} = (X^T X)^{-1} X^T y$$

를 얻습니다. 하지만 이 추정치는 과적합(overfitting)에 취약하고, 불확실성을 정량화하지 못합니다.

---

## Bayesian's Profit

베이지안 회귀에서는 회귀 계수 $\beta$와 노이즈 분산 $\sigma^2$를 확률변수로 취급합니다. 즉, 이들에 대해 사전분포(prior)를 설정하고, 관측된 데이터 $y$를 반영하여 사후분포(posterior)를 계산합니다.

- **사전분포** $p(\beta,\sigma^2)$  
- **가능도(likelihood)** $p(y\mid X,\beta,\sigma^2)$  
- **사후분포** $p(\beta,\sigma^2\mid X,y) \propto p(y\mid X,\beta,\sigma^2)\,p(\beta,\sigma^2)$

사후분포는 데이터와 사전정보를 결합한 분포로, 회귀 계수에 대한 불확실성을 정량적으로 표현합니다.

---

## Model

1. **가능도(likelihood)**  
   $$
   p(y\mid X,\beta,\sigma^2) = (2\pi\sigma^2)^{-\frac{n}{2}} \exp\Bigl(-\tfrac{1}{2\sigma^2}(y - X\beta)^T(y - X\beta)\Bigr).
   $$

2. **사전분포(prior)**  
   - 회귀 계수: $\beta\sim N(m_0,\,V_0)$  
   - 분산: $\sigma^2\sim \mathrm{Inv}\text{-}\Gamma(a_0,\,b_0)$

   여기서 $m_0\in\mathbb{R}^p$, $V_0\in\mathbb{R}^{p\times p}$는 사전 평균과 공분산 행렬, $a_0,b_0$는 Inverse-Gamma 분포의 shape과 scale 파라미터입니다.

---

## Posterior Distribution

사후분포 $p(\beta,\sigma^2\mid X,y)$는 다음과 같이 쓸 수 있습니다:

$$
p(\beta,\sigma^2\mid X,y) \propto p(y\mid X,\beta,\sigma^2)\;p(\beta\mid\sigma^2)\;p(\sigma^2).
$$

각각의 조건부 사후분포를 유도하면,

### Conditional Posterior Distribution of $\beta$ 

사전 $\beta\sim N(m_0,V_0)$, 우도 $y\sim N(X\beta,\sigma^2 I)$ 이므로, 가우시안-가우시안 결합 결과로

$$
\beta\mid y, X, \sigma^2 \sim N(m_n, \, V_n),
$$

여기서

$$
V_n = \bigl(V_0^{-1} + \tfrac{1}{\sigma^2} X^T X\bigr)^{-1},\quad
m_n = V_n \bigl(V_0^{-1} m_0 + \tfrac{1}{\sigma^2} X^T y\bigr).
$$

### Conditional Posterior Distribution of $\sigma^2$ 

사전 $\sigma^2\sim \mathrm{Inv}\text{-}\Gamma(a_0,b_0)$, 우도 기여를 합치면,

$$
\sigma^2\mid y,X \sim \mathrm{Inv}\text{-}\Gamma\bigl(a_n,\,b_n\bigr),
$$

여기서

$$
a_n = a_0 + \tfrac{n}{2},\quad
b_n = b_0 + \tfrac{1}{2}\bigl(y^T y + m_0^T V_0^{-1} m_0 - m_n^T V_0^{-1} m_n\bigr).
$$

---

## Predictive Distribution

새 입력 $x_*$에 대한 예측값 $y_*$의 사후 예측분포는,

$$
p(y_*\mid x_*,X,y) = \int p(y_*\mid x_*,\beta,\sigma^2)\;p(\beta,\sigma^2\mid X,y)\,d\beta\,d\sigma^2.
$$

이 적분 결과는 Student-t 분포 형태를 띠며,

$$
y_*\mid x_*,X,y \sim t_{2a_n}\bigl(x_*^T m_n,\, \tfrac{b_n}{a_n}\bigl(1 + x_*^T V_n x_*\bigr)\bigr).
$$

- 자유도: $2a_n$  
- 위치: $x_*^T m_n$  
- 스케일: $\sqrt{\tfrac{b_n}{a_n}\,(1 + x_*^T V_n x_*)}$  

---

## Python Code

```python
import numpy as np
from scipy.stats import invgamma, t

# 1) 데이터 생성
d, n = 3, 50
np.random.seed(0)
X = np.hstack([np.ones((n,1)), np.random.randn(n,d-1)])
beta_true = np.array([1.0, 2.0, -1.0])
sigma2_true = 0.5
y = X.dot(beta_true) + np.sqrt(sigma2_true)*np.random.randn(n)

# 2) 사전 설정
m0 = np.zeros(d)
V0 = np.eye(d) * 10
a0, b0 = 2.0, 1.0

# 3) 사후 파라미터 계산
Vn_inv = np.linalg.inv(V0) + (X.T @ X) / sigma2_true
Vn = np.linalg.inv(Vn_inv)
mn = Vn @ (np.linalg.inv(V0) @ m0 + X.T @ y / sigma2_true)
a_n = a0 + n/2
b_n = b0 + 0.5*(y.T@y + m0.T @ np.linalg.inv(V0) @ m0 - mn.T @ Vn_inv @ mn)

# 4) 예측
x_star = np.array([1.0, 0.5, -0.5])
df = 2*a_n
loc = x_star.dot(mn)
scale = np.sqrt(b_n/a_n * (1 + x_star.dot(Vn).dot(x_star)))
y_pred = t(df, loc=loc, scale=scale)
print(f"Predictive mean: {loc:.3f}")
print(f"Predictive 95% CI: {y_pred.ppf(0.025):.3f}, {y_pred.ppf(0.975):.3f}")
```

---

## Conclusion
Bayesian Linear Regression은 선형 모델에 확률론적 불확실성 해석을 더한 기법으로, 사전분포를 통해 과적합을 방지하고, 사후분포로 예측 신뢰도를 정량적으로 평가할 수 있습니다. Conjugate 모형을 사용하면 분석적 방식으로 빠르게 적합할 수 있고, 복잡한 Non-Conjugate 모형에서는 MCMC나 Variational Inference(VI) 등의 근사 기법을 통해 확장 가능합니다. 

감사합니다.

---

## Reference

- [Gelman, A., Carlin, J. B., Stern, H. S., Dunson, D. B., Vehtari, A., & Rubin, D. B. (2013). Bayesian Data Analysis (3rd ed.). CRC Press.](https://www.stat.columbia.edu/~gelman/book/)

- [Murphy, K. P. (2012). Machine Learning: A Probabilistic Perspective. MIT Press.](https://mitpress.mit.edu/9780262018029/machine-learning-a-probabilistic-perspective/)

- [Wikipedia : Bayeisna Linear Regression](https://en.wikipedia.org/wiki/Bayesian_linear_regression)

- [norman3.github.io/prml](https://norman3.github.io/prml/docs/chapter03/3.html)
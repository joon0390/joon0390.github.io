---
layout: single
title: "Linear Regression"
date: 2025-10-09
permalink: /linear-regression/
categories:
  - Statistics
  - Machine Learning
tags:
  - Linear Regression
  - Ordinary Least Squares
  - OLS
  - Regression
toc: true
toc_sticky: true
comments: true
---

> Linear Regressio은 **연속형 반응 변수**와 **설명 변수**의 선형 관계를 모델링하는 가장 기본적인 통계 모델입니다. 이 글에서는 OLS(Ordinary Least Squares)의 수식적 구조, 가정, 해석을 중심으로 정리하고, 간단한 파이썬 실습을 해보려고 합니다.

---

## 1. Model Definition

관측값 $i = 1, \ldots, n$에 대해 $p$개의 설명 변수를 갖는 다중선형회귀 모델은 다음과 같이 표현됩니다.

$$
y_i = \beta_0 + \beta_1 x_{i1} + \cdots + \beta_p x_{ip} + \varepsilon_i, \quad \varepsilon_i \sim \mathcal{N}(0, \sigma^2)
$$

행렬 형태로 작성하면

$$
\mathbf{y} = \mathbf{X}\boldsymbol{\beta} + \boldsymbol{\varepsilon},
$$

여기서

- $\mathbf{y} \in \mathbb{R}^n$ : 반응 벡터  
- $\mathbf{X} \in \mathbb{R}^{n \times (p+1)}$ : 설계행렬(design matrix), 첫 번째 열은 1  
- $\boldsymbol{\beta} \in \mathbb{R}^{p+1}$ : 회귀계수  
- $\boldsymbol{\varepsilon} \in \mathbb{R}^n$ : 오차항

---

## 2. OLS Estimator

OLS는 잔차제곱합(Residual Sum of Squares, RSS)을 최소화합니다.

$$
\text{RSS}(\boldsymbol{\beta}) = \Vert \mathbf{y} - \mathbf{X}\boldsymbol{\beta} \Vert_2^2
$$

이를 $\boldsymbol{\beta}$에 대해 미분하여 0으로 놓으면

$$
\frac{\partial \text{RSS}}{\partial \boldsymbol{\beta}} = -2\mathbf{X}^\top(\mathbf{y} - \mathbf{X}\boldsymbol{\beta}) = \mathbf{0}
$$

따라서 **정규방정식(normal equation)** 은

$$
\mathbf{X}^\top \mathbf{X} \hat{\boldsymbol{\beta}} = \mathbf{X}^\top \mathbf{y}
$$

이고, $\mathbf{X}^\top \mathbf{X}$가 가역(invertible)일 때 OLS 해는

$$
\hat{\boldsymbol{\beta}} = (\mathbf{X}^\top \mathbf{X})^{-1}\mathbf{X}^\top \mathbf{y}
$$

입니다.

---

## 3. Geometric Interpretation

OLS는 $\mathbf{y}$를 $\mathbf{X}$의 열공간(column space)에 정사영(project)하는 문제로 이해할 수 있습니다. 즉, 추정된 반응 $\hat{\mathbf{y}} = \mathbf{X}\hat{\boldsymbol{\beta}}$는 $\mathbf{X}$의 열공간에 속하며, 잔차 $\hat{\boldsymbol{\varepsilon}} = \mathbf{y} - \hat{\mathbf{y}}$는 $\mathbf{X}$의 열공간과 직교합니다.

$$
\mathbf{X}^\top \hat{\boldsymbol{\varepsilon}} = \mathbf{X}^\top(\mathbf{y} - \mathbf{X}\hat{\boldsymbol{\beta}}) = \mathbf{0}
$$

이 직교조건이 정규방정식과 동일합니다.

---

## 4. Basic Assumption of Linear Regression

1. **선형성(linearity)**: $\mathbb{E}[y|\mathbf{x}] = \mathbf{x}^\top \boldsymbol{\beta}$  
2. **독립성(independence)**: 관측치 간 오차가 독립  
3. **등분산성(homoscedasticity)**: $\operatorname{Var}(\varepsilon_i) = \sigma^2$ 일정  
4. **비다중공선성(no multicollinearity)**: $\mathbf{X}^\top \mathbf{X}$는 가역  
5. **정규성(normality)** *(추론 목적)*: $\boldsymbol{\varepsilon} \sim \mathcal{N}(\mathbf{0}, \sigma^2 \mathbf{I})$

가정을 만족하면 OLS 추정량은 Gauss–Markov 정리에 따라 BLUE(Best Linear Unbiased Estimator)가 됩니다.

---

## 5. Inference: Variance and Confidence Intervals

OLS 추정량의 분산은

$$
\operatorname{Var}(\hat{\boldsymbol{\beta}}) = \sigma^2 (\mathbf{X}^\top \mathbf{X})^{-1}
$$

이며, $\sigma^2$는 잔차제곱합으로 추정합니다.

$$
\hat{\sigma}^2 = \frac{1}{n - p - 1} \Vert \mathbf{y} - \mathbf{X}\hat{\boldsymbol{\beta}} \Vert_2^2
$$

각 계수의 표준오차는 $\sqrt{\operatorname{diag}(\hat{\sigma}^2 (\mathbf{X}^\top \mathbf{X})^{-1})}$로 계산하고, $t$-통계량을 통해 유의성 검정 및 신뢰구간을 구성합니다.

---

## 6. Worked Example: OLS in Python

간단한 2차원 설명 변수 예제를 통해 OLS 절차를 직접 구현해 보겠습니다.

1. 모의 데이터를 생성하고  
2. 정규방정식을 이용해 수작업으로 추정량을 계산한 뒤  
3. `statsmodels.OLS`와 결과를 비교합니다.

```python
import numpy as np
import statsmodels.api as sm

rs = np.random.default_rng(42)
n = 200

# 데이터 생성
beta_true = np.array([1.5, -2.0, 0.8])
X = rs.normal(size=(n, 2))
X_with_intercept = np.column_stack([np.ones(n), X])
noise = rs.normal(scale=0.5, size=n)
y = X_with_intercept @ beta_true + noise

# OLS 수작업
beta_hat = np.linalg.inv(X_with_intercept.T @ X_with_intercept) @ X_with_intercept.T @ y
residuals = y - X_with_intercept @ beta_hat
sigma2_hat = (residuals @ residuals) / (n - X_with_intercept.shape[1])

print(\"β̂ (manual):\", beta_hat)
print(\"σ̂²:\", sigma2_hat)

# statsmodels 비교
model = sm.OLS(y, X_with_intercept)
result = model.fit()
print(result.summary())
```

> 공선성이 존재해 $\mathbf{X}^\top\mathbf{X}$가 역행렬을 갖지 않는다면 `np.linalg.pinv(X_with_intercept) @ y`처럼 **Moore–Penrose 역행렬**을 사용하면 최소노름 해를 얻을 수 있습니다.

---

## 7. Model Diagnostics

모델 적합 후 다음과 같은 진단 플롯과 통계를 확인해야 합니다.

- **잔차 vs 적합값**: 패턴이 있으면 비선형 구조 또는 이분산을 의심  
- **QQ Plot**: 잔차 정규성 판단  
- **VIF (Variance Inflation Factor)**: 다중공선성 점검  
- **Cook's Distance**: 영향력(observation influence) 탐지

---

### References

- Statsmodels Documentation: [https://www.statsmodels.org/stable/regression.html](https://www.statsmodels.org/stable/regression.html)

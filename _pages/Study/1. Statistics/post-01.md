---
title: "Example Post: no thumbnail image"
date: "2023-12-01"
---

# BKMR

<script>MathJax = { tex: { inlineMath: [['$', '$'], ['\\(', '\\)']] }, svg: { fontCache: 'global' } };</script><script type="text/javascript" src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-svg.js"></script>

---

[Bayesian kernel machine regression for estimating the health effects of multi-pollutant mixtures](https://academic.oup.com/biostatistics/article/16/3/493/269719)

## 1\. BKMR이란?

BKMR은 여러 개의 노출 변수(Exposure)가 비선형적으로, 그리고 상호작용하면서 어떤 결과 변수(Outcome)에 영향을 줄 때 사용되는 회귀 모델입니다.

---

### 💡등장배경

환경 역학이나 독성학에서는 여러 화학물질이 복잡적으로 건강에 미치는 영향을 알아보고자 합니다.

하지만.

-   변수 들 간의 상호작용이 존재. → 선형 회귀는 어려움.
-   모든 조합을 일일이 고려할 수 없음.
-   불확실성까지 보고싶음.

👉 이런 문제를 해결하고자 BKMR이 등장하였습니다.

---

## 2\. 수학적 모델 구조

BKMR의 기본 모델은 다음과 같습니다.

$$  
y\_i = h(z\_i) + x\_i^T\\beta + \\epsilon\_i, \\epsilon\_i \\sim N(0,\\sigma^2)  
$$

-   $y\_i$ : 종속변수 (ex. 건강 지표)
-   $z\_i$ : 관심 변수들 (ex. 화학물질 노출)
-   $x\_i$ : 조정 공변량 (ex. 나이, 성별, 흡연 여부 등)
-   $h(\\cdot)$ : 비선형 커널 함수로 표현되는 효과

👉 이 모델은 $h(z)$를 **`Gaussian Process`** 로 설정함으로써 매우 유연한 비선형 함수로 추정 가능합니다.

---

## 3\. GPR과의 관계

BKMR은 두 아이디어의 결합방법입니다.

1.  $h(\\cdot) \\sim GP(0,K)$ : Gaussian Process 기반의 비선형 함수
2.  나머지 부분은 선형회귀

**즉, GPR을 부분적으로 사용하여 일부 변수들만 비선형적으로 다루는 회귀 모델이라고 볼 수 있습니다.**

---

## 4\. 커널 함수의 역할과 선택

BKMR의 핵심은 커널함수 $k(z\_i, z\_j)$를 어떻게 정의하느냐에 달려있습니다.

보통는 RBF Kernel을 많이 사용합니다:

$$  
k(z\_i,z\_j) = \\exp\\bigg(-\\dfrac{1}{\\rho^2}|z\_i-z\_j|^2\\bigg)  
$$

-   $\\rho$ : Bandwidth parameter → MCMC로 샘플링
-   $K$ : 이 커널을 모든 데이터 쌍에 대해서 계산하면 $n\\times n$ 행렬

---

## 5\. 공변량 조정(Covariate adjustment)

$h(z)$는 커널로 모델링 되지만, $x$는 일반 선형 회귀처럼 모델링됩니다.

이 구조는 다음의 장점을 갖습니다.

-   해석 가능성 증가
-   선형 조정 효과 분리

👉 복잡한 비선형 효과는 $h$, 명확한 선형 효과는 $\\beta$

---

## 6\. 추론 방식 : MCMC로 Posterior 추정

BKMR은 완전한 베이지안 모델이라서, 다음을 샘플링합니다.

-   $h(\\cdot)$ : GP로 표현된 Latent function
-   $\\rho, \\sigma^2, \\beta$ 등의 하이퍼파라미터

MCMC 방법은 Gibbs sampling과 Metropoliss-Hastings를 조합해서 추정합니다.

👉 계산량은 크지만, **Posterior Distribution 전체**를 얻을 수 있습니다!

---

## 7\. 변수 중요도와 상호작용 해석

BKMR이 강력한 이유는, 단순 예측이 아니라:

-   Variable Importance(PIP : Posterior Inclusion Probabilty)
-   Interaction Effect(2개 변수의 결합 영향 시각화)
-   Overall Effect(노출 전체의 집합적 영향)

같은 해석을 가능하게 하기 때문입니다.

예를 들어

```
OverallRiskSummary(fit)
SingVarRiskSummary(fit, sel = 1)
```

등의 함수를 통해서 아래와 같은 결과를 확인할 수 있습니다.

-   “변수 A는 건강에 얼마나 영향을 미치는 가”
-   “A와 B는 시너지 효과를 내는가(상호작용이 있는가)”
-   “모든 노출이 다 증가했을 때의 전체 효과는?”

---

## 8\. BKMR의 장단점

| 장점 | 단점 |
| --- | --- |
| ✅ 비선형 관계 추정 | ❌ 계산량 매우 큼(MCMC) |
| ✅ 변수 간의 상호작용 가능 | ❌ 매우 큰 데이터에 부적합 |
| ✅ 베이지안 해석 가능 | ❌ Python 지원 X(R only) |
| ✅ 변수 중요도, 불확실성 추정 |   |

---

## 9\. 예시

```
library(bkmr)

# outcome: Y (예: 건강지표)
# Z: 관심 변수들 (노출물질들)
# X: 공변량 (나이, 성별 등)

fit <- kmbayes(y = Y, Z = Z_matrix, X = X_matrix, iter = 10000)

summary(fit)

# 변수 영향력
SingVarRiskSummary(fit)

# 전체 효과
OverallRiskSummary(fit)

# 변수 간 상호작용
PredictorResponseBivar(fit, min.plot.dist = 0.2)
```

## 10\. 관련 자료

-   [Introduction to BKMR](https://jenfb.github.io/bkmr/overview.html)
-   [bkmr R packages](https://cran.r-project.org/web/packages/bkmr/index.html)
-   [BKMR Guide](https://bkmr-guide-iab-env-epi-c1e9f1201284eb8158cc30169fbc7e2f9058900a.gricad-pages.univ-grenoble-alpes.fr/)

 [Bayesian Kernel Machine Regression (BKMR) Guide

Presentation of the model Bayesian Kernel Machine Regression (BKMR) is a statistical method designed to model the complex relationships between a specific outcome (referred to as Y) and multiple set of predictor variables. This extensive set of variables i

bkmr-guide-iab-env-epi-c1e9f1201284eb8158cc30169fbc7e2f9058900a.gricad-pages.univ-grenoble-alpes.fr](https://bkmr-guide-iab-env-epi-c1e9f1201284eb8158cc30169fbc7e2f9058900a.gricad-pages.univ-grenoble-alpes.fr/)

 [bkmr: Bayesian Kernel Machine Regression

Implementation of a statistical approach for estimating the joint health effects of multiple concurrent exposures, as described in Bobb et al (2015) <<a href="https://doi.org/10.1093%2Fbiostatistics%2Fkxu058" target="\_top">doi:10.1093/biostatistics/kxu058<

cran.r-project.org](https://cran.r-project.org/web/packages/bkmr/index.html)

 [Introduction to Bayesian kernel machine regression and the bkmr R package

jenfb.github.io](https://jenfb.github.io/bkmr/overview.html)
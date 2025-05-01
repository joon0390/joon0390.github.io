---
title: "Bayesian Kernel Machine Regression (BKMR)"
tag:
  - Statistics
  - Bayesian
  - BKMR
  - Bayesian Kernel Machine Regression
  - Regression
  - Kernel Method
  - Gaussian Process
date: "2023-12-01"
---

# BKMR
> 본 포스팅에서는 Bayesian Kernel Machine Regression (BKMR)의 이론적 배경을 정리하고 간단한 실습을 진행합니다.  

---

## Introduction

환경 역학 연구에서는 다수의 오염 물질이 결합하여 미치는 건강 영향을 분석할 때, 변수 간의 비선형성·상호작용을 명확히 파악하기 어려운 한계가 있습니다.  
BKMR은 이러한 과제를 해결하기 위해 개발된 베이지안 회귀 기법으로, 다음과 같은 장점을 제공합니다:

1. **다변량 노출 변수 동시 모델링**  
   여러 노출 요인의 개별 및 결합 효과를 통합적으로 평가합니다.  
2. **비선형 반응 곡선 학습**  
   임계점(threshold), 정상화(plateau) 등의 비선형 노출–반응 관계를 유연하게 포착합니다.  
3. **상호작용 탐지**  
   변수 간 상승작용(synergy) 또는 길항작용(antagonism)을 식별합니다.  
4. **불확실성 정량화**  
   MCMC 기반 사후분포 추론을 통해 예측값과 함께 95% credible interval을 제공합니다.

---

## BKMR Model

본 절에서는 BKMR의 수학적 구조와 추론 절차를 요약합니다.

### 1) Model Structure

관측치 \(i\)의 결과 \(y_i\)는 다음과 같이 정의됩니다:

\[
y_i = h(\mathbf{x}_i) + \mathbf{z}_i^\mathsf{T}\boldsymbol{\beta} + \epsilon_i,
\quad \epsilon_i \sim \mathcal{N}(0,\sigma^2),
\]

- \(\mathbf{x}_i=(x_{i1},\dots,x_{ip})^\mathsf{T}\): 주요 노출 변수 벡터  
- \(\mathbf{z}_i\): 나이·성별 등 고정효과로 처리되는 보조 변수  
- \(\boldsymbol{\beta}\): 고정효과 선형 계수  
- \(h(\cdot)\): 노출 변수의 비선형·상호작용 효과를 모델링하는 함수  

### 2) Gaussian Process Prior

함수 \(h\)에는 Gaussian Process 사전분포를 적용합니다:

\[
h(\mathbf{x}) \sim \mathcal{GP}\bigl(0,\,K(\mathbf{x},\mathbf{x}')\bigr),
\]

주로 사용되는 커널:

- **RBF 커널**  
  \[
  K_{\mathrm{RBF}}(\mathbf{x},\mathbf{x}') 
  = \sigma_f^2 \exp\!\Bigl(-\tfrac{1}{2\ell^2}\|\mathbf{x}-\mathbf{x}'\|^2\Bigr)
  \]
  - \(\ell\): 길이 척도(length-scale)  
  - \(\sigma_f^2\): 함수 진폭 분산  
- **Matern 커널** (\(\nu=1.5\) 또는 \(2.5\) 권장)  
  \[
  K_{\nu}(r)
  = \sigma_f^2 \frac{2^{1-\nu}}{\Gamma(\nu)}
    \Bigl(\sqrt{2\nu}\tfrac{r}{\ell}\Bigr)^{\nu}
    K_{\nu}\Bigl(\sqrt{2\nu}\tfrac{r}{\ell}\Bigr),
    \quad r=\|\mathbf{x}-\mathbf{x}'\|.
  \]

### 3) Bayeisian Inference

1. **사전분포 설정**  
   - \(\boldsymbol{\beta} \sim \mathcal{N}(0,\tau^2I)\)  
   - \(\sigma^2 \sim \mathrm{Inv}\text{-}\Gamma(a_0,b_0)\)  
   - 커널 하이퍼파라미터 \(\ell,\sigma_f\): Half-Cauchy 분포  

2. **우도(Likelihood)**  
   \[
   \mathbf{y}\mid h,\boldsymbol{\beta},\sigma^2
   \sim \mathcal{N}\bigl(h(\mathbf{X}) + Z\boldsymbol{\beta},\,\sigma^2I\bigr)
   \]

3. **사후분포 (Posterior)**  
   \[
   p(h,\boldsymbol{\beta},\sigma^2,\theta\mid\mathbf{y})
   \;\propto\;
   p(\mathbf{y}\mid h,\boldsymbol{\beta},\sigma^2)\,
   p(h\mid\theta)\,p(\boldsymbol{\beta})\,p(\sigma^2)\,p(\theta)
   \]

4. **MCMC 샘플링**  
   - \(\boldsymbol{\beta},\sigma^2\): Gibbs 샘플링  
   - \(\theta=(\ell,\sigma_f)\), \(h(\mathbf{X})\): Metropolis–Hastings 또는 Elliptical Slice Sampling  
   - 수렴 진단: trace plot, \(\hat{R}\), effective sample size(ESS)

---

## Lab

다음 예제에서는 시뮬레이션 데이터를 생성하여 BKMR 모델을 적합하고, 결과를 요약·시각화해 봅니다.

```r
library(bkmr)    # devtools::install_github("jenfb/bkmr") 
library(dplyr)

set.seed(123)
dat <- SimData(n = 80, M = 3)    # n: 관측치 수, M: 노출 변수 개수
y <- dat$y
Z <- dat$Z                       # p차원 노출 변수 행렬
X <- dat$X                       # 보조 변수(예: 나이, 성별) 행렬

# BKMR 모델 적합
fitkm <- kmbayes(
  y       = y,
  Z       = Z,
  X       = X,
  iter    = 500,    
  varsel  = TRUE, 
  verbose = FALSE
)

print(fitkm)        
summary(fitkm)      
```
![summary](joon0390.github.io/_pages/Study/1.Statistics/img/summary.png)

```r
# 변수 선택 지표(PIP) 추출 및 확인
pips <- ExtractPIPs(fitkm)
print(pips)

# 단일 변수 노출–반응 함수 계산
pred.univar <- PredictorResponseUnivar(
  fit     = fitkm,
  method  = "approx",
  ngrid   = 50,
  q.fixed = 0.5
)

library(ggplot2)

ggplot(pred.univar, aes(x = z, y = est)) +
  geom_ribbon(aes(ymin = est - 1.96 * se,
                  ymax = est + 1.96 * se),
              alpha = 0.3) +
  geom_line() +
  facet_wrap(~ variable, scales = "free_x") +
  labs(
    x     = "노출 변수 값 (z)",
    y     = "예측된 h(z)",
    title = "단일 변수 노출–반응 함수"
  ) +
  theme_minimal()
```
![결과 시각화](joon0390.github.io/_pages/Study/1.Statistics/img/beta.png)
```r
risks <- OverallRiskSummaries(
  fit     = fitkm,
  qs      = seq(0.25, 0.75, by = 0.1),
  q.fixed = 0.5,
  method  = "approx"
)
print(risks)

```
![Risk 시각화](joon0390.github.io/_pages/Study/1.Statistics/img/risk.png)

---

## Reference

1. [Bobb, J. F., Valeri, L., Claus Henn, B., et al. (2015). *Bayesian kernel machine regression for estimating the health effects of multi-pollutant mixtures*. Biostatistics, 16(3), 493–508.](https://doi.org/10.1093/biostatistics/kxu058)  
2. [R Package : bkmr: Bayesian Kernel Machine Regression
](https://cran.r-project.org/web/packages/bkmr/index.html)  

---
layout: single
title: "Bayesian ICA"
date: 2025-10-02
permalink: /bica/
categories:
  - Statistics
  - Bayesian
tags:
  - ICA
  - Variational Inference
  - Latent Variable Model
  
toc: true
toc_sticky: true
comments: true
---

> 이 포스팅은 **Bayesian Independent Component Analysis (Bayesian ICA)** 에 대해 공부하며 정리한 글입니다.  

---

## Idea of ICA

ICA의 기본 모델은 다음과 같습니다. 

$$
\boldsymbol{x} = \boldsymbol{A}\boldsymbol{s} + \boldsymbol{\epsilon}
$$

- $\boldsymbol{x} \in \mathbb{R}^{D}$ : 관측 데이터 (observed data)  
- $\boldsymbol{s} \in \mathbb{R}^{K}$ : **독립된 잠재 변수 (independent sources)**  
- $\boldsymbol{A} \in \mathbb{R}^{D \times K}$ : Mixing Matrix  
- $\boldsymbol{\epsilon}$ : 노이즈 (보통 $\mathcal{N}(0, \sigma^2I)$)

ICA의 목표는 $A$와 $\boldsymbol{s}$를 추정하는 것인데, 이때 $\boldsymbol{s}$의 각 성분이 **통계적으로 독립적(independent)** 이라고 가정합니다.

즉, 관측된 데이터는 서로 독립적인 잠재 변수들의 **선형 혼합(linear mixture)** 이라는 것이 핵심 가정입니다.

$$
p(\boldsymbol{s}) = \prod_{k=1}^K p(s_k)
$$

> 전통적인 ICA는 Maximum Likelihood 기반으로 파라미터를 점 추정(point estimate)하지만,  
> **Bayesian ICA**는 파라미터 전체를 확률변수로 두어 **사후분포(posterior distribution)** 를 추정합니다.

---

## Bayesian ICA Model

Bayesian ICA의 전체 확률 모형은 다음과 같습니다.

$$
p(\boldsymbol{x}, \boldsymbol{s}, \boldsymbol{A}, \sigma^2)
= p(\boldsymbol{x} | \boldsymbol{A}, \boldsymbol{s}, \sigma^2)\,
  p(\boldsymbol{s})\, p(\boldsymbol{A})\, p(\sigma^2)
$$

- **Likelihood**
  $$
  p(\boldsymbol{x} | \boldsymbol{A}, \boldsymbol{s}, \sigma^2)
  = \mathcal{N}(\boldsymbol{x} | \boldsymbol{A}\boldsymbol{s}, \sigma^2 I)
  $$

- **Prior on sources** (독립 비가우시안 분포; e.g., Laplace Prior)
  $$
  p(s_k) = \frac{1}{Z}\exp(-|s_k|)
  $$
  여기서 $Z$는 정규화 상수이며, 이 형태는 Laplace(0,1) 분포에 해당합니다.  
  이는 대부분의 $s_k$가 0 근처(희소성)이고, 가끔 큰 값을 가질 수 있는 형태로  
  **비가우시안성(non-Gaussianity)** 과 **sparsity**를 동시에 부여합니다.  

  <br>

  > 만약 $p(s_k)$가 Gaussian이라면 ICA는 식별 불가능합니다.  
  > 즉, 독립 원천을 구분할 수 없기 때문에, 반드시 비가우시안 prior가 필요합니다.

- **Prior on Mixing Matrix**
  $$
  p(\boldsymbol{A}) = \prod_{i,j}\mathcal{N}(A_{ij} | 0, \tau^2)
  $$

- **Noise Prior**
  $$
  p(\sigma^2) \sim \text{Inverse-Gamma}(\alpha, \beta)
  $$

---

## Inference

ICA는 잠재 변수 $\boldsymbol{s}$가 비가우시안(non-Gaussian)을 따르기 때문에,  
Posterior $p(\boldsymbol{s}, \boldsymbol{A} | \boldsymbol{x})$는 닫힌 형태로 계산되지 않습니다.  
따라서 **근사추론(Approximate Inference)** 기법을 사용합니다.  

---

### (1) Variational Bayesian ICA (VB-ICA)

#### Factorized Posterior 가정
$$
q(\boldsymbol{s}, \boldsymbol{A}) = q(\boldsymbol{A}) \prod_n q(\boldsymbol{s}_n)
$$

#### ELBO (Evidence Lower Bound)
$$
\mathcal{L} = 
\mathbb{E}_q[\log p(\boldsymbol{X}, \boldsymbol{S}, \boldsymbol{A}) - \log q(\boldsymbol{S}, \boldsymbol{A})]
$$

이 하한(ELBO)을 최대화하면, 실제 로그 사후가능도(log-evidence)를 근사하게 됩니다.

#### VB-EM 형태의 업데이트
- **E-step:** $q(\boldsymbol{s}_n)$ 업데이트  
  $$
  q(\boldsymbol{s}_n) \propto 
  \exp\left(
  \mathbb{E}_{q(\boldsymbol{A})}[\log p(\boldsymbol{x}_n | \boldsymbol{A}, \boldsymbol{s}_n)]+ \log p(\boldsymbol{s}_n) \right)$$

- **M-step:** $q(\boldsymbol{A})$, $\sigma^2$, 하이퍼파라미터 업데이트

#### 수렴 후
Posterior 평균을 추정치로 사용:
$$
\hat{\boldsymbol{S}} = \mathbb{E}_q[\boldsymbol{S}], \quad
\hat{\boldsymbol{A}} = \mathbb{E}_q[\boldsymbol{A}]
$$

---

### (2) MCMC 기반 Bayesian ICA

- Gibbs sampling 혹은 Hamiltonian Monte Carlo(HMC)를 이용해  
  $p(\boldsymbol{A}, \boldsymbol{S} | \boldsymbol{X})$에서 직접 샘플링합니다.
- 이는 Posterior 불확실성을 완전하게 반영 가능합니다. 
- 하지만 계산량이 매우 크다는 단점도 존재합니다.

---

##  Variational Bayesian ICA Summary

| 단계 | 설명 |
|------|------|
| 1️⃣ 초기화 | $q(\boldsymbol{A}) = \mathcal{N}(\boldsymbol{A}_0, \Sigma_A)$ |
| 2️⃣ 반복 | 각 데이터 $x_n$에 대해 $q(\boldsymbol{s}_n)$ 업데이트 |
| 3️⃣ 갱신 | $\boldsymbol{A}$의 Posterior 및 하이퍼파라미터 업데이트 |
| 4️⃣ 수렴 | Posterior Mean을 추정값으로 사용 |

---

## References

- Attias, H. (1999). [*Independent Factor Analysis*. Neural Computation.](https://psycnet.apa.org/record/1999-13540-001)
- [Wikipedia ICA](https://en.wikipedia.org/wiki/Independent_component_analysis)
- Alaa, T. (2020). [Independent component analysis: An introduction](https://www.emerald.com/aci/article/17/2/222/6032/Independent-component-analysis-An-introduction)
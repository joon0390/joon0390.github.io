---
layout: single  
title: "Natural Gradient for Variational Inference"  
date: 2025-09-23 
permalink: /natural-gradient/  
categories:  
  - Statistics
  - Machine Learning
  - Bayesian
tags:  
  - Natural Gradient
  - Variational Inference
  - Information Geometry

toc: true  
toc_sticky: true  
---

> 이 포스팅은 [Amari (1998)](https://ieeexplore.ieee.org/document/6790500)의 **Natural Gradient** 개념과  
> Variational Inference에서의 응용을 정리한 글입니다.  

---

## Introduction

일반적인 경사하강법(Gradient Descent)은 파라미터 공간을 **유클리드 공간**으로 간주합니다.  
하지만 확률분포 최적화 문제에서는, 동일한 분포라도 **파라미터화(parameterization)** 방식에 따라 경사 방향이 달라지는 문제가 발생합니다.  

이를 해결하기 위해 Amari (1998)는 **자연 그래디언트(Natural Gradient)** 를 제안했습니다.  
이는 확률분포 공간의 **정보 기하학적 구조(Fisher metric)** 를 반영한, Parameterization에 불변한 최적화 방향입니다.  

---

## Ordinary Gradient vs. Natural Gradient

- **일반 Gradient 업데이트**
  $$
  \theta \leftarrow \theta - \eta \nabla_\theta L(\theta)
  $$

- **Natural Gradient 업데이트**
  $$
  \theta \leftarrow \theta - \eta \, F(\theta)^{-1}\nabla_\theta L(\theta)
  $$

여기서 $F(\theta)$는 **Fisher Information Matrix**입니다:
$$
F(\theta) = \mathbb{E}\big[ \nabla_\theta \log p(y|\theta)\, \nabla_\theta \log p(y|\theta)^\top \big].
$$

---

## Why Natural Gradient?

- **좌표 불변성**: 파라미터화 방식이 달라져도 동일한 업데이트 방향  
- **스케일 적응성**: feature마다 다른 스케일을 자동으로 보정  
- **빠른 수렴**: Variational Bayes, 강화학습, Bayesian NN에서 안정적인 학습  

---

### Derivation via KL Constraint

자연 그래디언트는 “KL 반경을 일정하게 유지하면서 ELBO를 가장 빠르게 증가시키는 방향”으로 유도됩니다.  
아래는 이를 한 줄 한 줄 증명한 과정입니다.

---

#### 1. 국소 최적화 문제 설정

- 근사 분포 $q_\lambda(\theta)$가 있고,  
- 최적화 목표는 $\mathcal L(\lambda)$ (예: ELBO).  
- 변화량을 $\mathrm d\lambda$, 기울기를 $g := \nabla_\lambda \mathcal L(\lambda)$라 둡니다.

1. $\mathcal L$의 1차 테일러 근사:
$$
\mathcal L(\lambda + \mathrm d\lambda) \;\approx\; \mathcal L(\lambda) + g^\top \mathrm d\lambda.
$$

2. KL 발산의 2차 근사:
$$
\mathrm{KL}\!\big(q_\lambda \,\|\, q_{\lambda+\mathrm d\lambda}\big)
\;\approx\; \tfrac12\, \mathrm d\lambda^\top F(\lambda)\, \mathrm d\lambda,
$$
여기서
$$
F(\lambda) = \mathbb E_{q_\lambda}\!\big[\nabla_\lambda \log q_\lambda(\theta)\, \nabla_\lambda \log q_\lambda(\theta)^\top\big]
$$
는 Fisher 정보 행렬입니다.

3. 따라서 신뢰영역(trust region) 문제는:
$$
\max_{\mathrm d\lambda}\; g^\top \mathrm d\lambda
\quad \text{s.t.}\quad
\tfrac12\,\mathrm d\lambda^\top F(\lambda)\,\mathrm d\lambda \le \varepsilon.
\tag{*}
$$


#### 2. 라그랑주 승수법 (KKT 조건)

4. 라그랑지안 정의:
$$
\mathcal J(\mathrm d\lambda,\alpha)
= g^\top \mathrm d\lambda - \alpha\!\left(\tfrac12\,\mathrm d\lambda^\top F\,\mathrm d\lambda - \varepsilon\right),
\quad \alpha \ge 0.
$$

5. 정지조건(stationarity):
$$
\nabla_{\mathrm d\lambda}\mathcal J
= g - \alpha F \mathrm d\lambda = 0
\;\;\Longrightarrow\;\;
F\,\mathrm d\lambda = \tfrac{1}{\alpha} g.
$$

6. 따라서,
$$
\mathrm d\lambda = \tfrac{1}{\alpha}\, F^{-1}\, g.
\tag{A}
$$

7. 제약 활성화(보완슬랙성):
$$
\tfrac12\,\mathrm d\lambda^\top F\,\mathrm d\lambda = \varepsilon.
$$

8. (A)를 대입:
$$
\tfrac12 \Big(\tfrac{1}{\alpha}F^{-1}g\Big)^\top F \Big(\tfrac{1}{\alpha}F^{-1}g\Big)
= \varepsilon.
$$

9. 정리하면:
$$
\frac{1}{2\alpha^2}\, g^\top F^{-1} g = \varepsilon
\quad\Longrightarrow\quad
\alpha = \sqrt{\frac{g^\top F^{-1} g}{2\varepsilon}}.
$$


#### 3. 최적 업데이트

따라서 최적 변화량은
$$
\mathrm d\lambda^\star = \kappa \, F^{-1} g,
$$
여기서 $\kappa = 1/\alpha$는 학습률에 해당하는 상수입니다.

즉, 자연 그래디언트는
$$
\tilde{\nabla}_\lambda \mathcal L = F(\lambda)^{-1}\nabla_\lambda \mathcal L.
$$

---

### 4. 핵심 포인트

- 일반 gradient는 유클리드 공간에서의 “좌표 의존적” 기울기.  
- 자연 gradient는 분포 공간의 **KL 기하학(정보 기하학)** 을 고려한 기울기.  
- 따라서 **파라미터화에 불변**하며, 동시에 feature 스케일 차이도 자동으로 보정.  

---

## Application to Variational Inference

Variational Inference(VI)에서는 근사 분포 $q_\lambda(\theta)$를 두고,  
Evidence Lower Bound(ELBO)를 최대화합니다.  

- **일반 gradient**: 좌표계와 스케일에 민감  
- **자연 gradient**: KL geometry 반영 → 훨씬 안정적  

예: **Gaussian Variational Approximation**
$$
q_\lambda(\theta) = \mathcal{N}(\mu, BB^\top + D^2),
$$
여기서 $\lambda = (\mu, B, D)$.  
이 경우 자연 그래디언트를 사용하면 $\mu, B, D$ 업데이트가 빠르고 안정적입니다.  

---

## Visualization Examples

### 1. Quadratic Loss (Elliptical Gaussian NLL)

![Quadratic Example](/assets/img/natural_gradient/output.png){: .align-center}

위 그림은 단순한 **이차함수(quadratic)** 에서 Ordinary Gradient(파란색)와 Natural Gradient(초록색)를 비교한 것입니다.  
- Ordinary Gradient는 **좌표축 기준**의 경사를 따라가기 때문에 곡선의 경로를 보입니다.  
- Natural Gradient는 Fisher metric을 반영하여 **등고선을 따라 곧장 최적점으로 향하는 방향**을 잡습니다.  

---

### 2. Mixture of Gaussians Negative Log-Likelihood

![Mixture of Gaussians Example](/assets/img/natural_gradient/output2.png){: .align-center}

여기서는 **두 개의 가우시안이 섞인 분포(Mixture of Gaussians)** 의 음의 로그 가능도(NLL)를 최적화하는 상황입니다.  
- Ordinary Gradient는 곡선의 경로를 가지고 최적점을 향하는 모습을 보입니다. 
- Natural Gradient는 Fisher Information을 기반으로 분포의 **지역적 곡률(local curvature)** 을 반영하여, 더 안정적인 최적화 방향을 보여줍니다.  

이 예시는 **복잡한 확률분포**에서도 Natural Gradient가 더 효율적임을 잘 보여줍니다.  

---

## Intuition

- 일반 gradient: 단순히 **좌표축 기준의 기울기**  
- 자연 gradient: **분포 공간에서의 진짜 경사 방향**  
- Fisher Information: “민감한 방향은 줄이고, 둔감한 방향은 키워주는 스케일링 역할”  

---

## Conclusion

- **Natural Gradient**는 Variational Inference, 강화학습, Bayesian Deep Learning 등에서 필수적인 도구  
- 분포의 기하학적 구조를 반영해 **더 효율적이고 안정적인 최적화**를 제공  
- DeepGLM/GLMM 같은 최신 Bayesian Deep Learning 연구에서도 핵심적으로 활용됨  

---

## Reference

- [Natural Gradient Works Efficiently in Learning
](https://ieeexplore.ieee.org/document/6790500)
- [Why Natural Gradient?](http://www.yaroslavvb.com/papers/amari-why.pdf)
- [Natural Gradient Methods: Perspectives,
Efficient-Scalable Approximations, and Analysis](https://arxiv.org/pdf/2303.05473)
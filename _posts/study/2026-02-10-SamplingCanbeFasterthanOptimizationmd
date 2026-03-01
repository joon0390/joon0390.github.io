---
layout: single
title: "[Paper Review] Sampling Can Be Faster Than Optimization"
date: 2026-02-10
permalink: /sampling-can-be-faster-than-optimization/
categories:
  - Machine Learning
  - Bayesian
  - Optimization
tags:
  - MCMC
  - Langevin Dynamics
  - ULA
  - MALA
  - Nonconvex Optimization

toc: true
toc_sticky: true
comments: true
---

> 이 포스팅은 **Ma, Chen, Jin, Flammarion, Jordan (2019)** 의 논문  
> [*Sampling Can Be Faster Than Optimization*](https://arxiv.org/pdf/1811.08413) 을 읽고 정리한 글입니다.

---

## 1. Bigger Picture: Two Tools for Inference

머신러닝에서 파라미터를 추정할 때 우리는 보통 두 가지 접근을 씁니다.

- **최적화(optimization)**: 손실을 최소화하는 한 점(point estimate)을 찾음
- **샘플링(sampling)**: 분포 전체(불확실성 포함)를 근사함

실무에서는 대개 "먼저 최적화로 빠르게 답을 찾고, 필요하면 샘플링으로 불확실성을 본다"는 흐름이 자연스럽습니다.  
그래서 암묵적으로 다음 통념이 자리 잡았습니다.

> "최적화는 빠르고, 샘플링은 정확하지만 느리다."

문제는 이 통념이 주로 **볼록(convex) / 로그볼록(log-concave)** 세계에서 정립되었다는 점입니다.  
이 논문은 한 단계 더 일반적인 상황, 즉 **국소적으로만 비볼록인 문제**로 질문을 확장합니다.

> "Local nonconvexity에서는 여전히 최적화가 샘플링보다 본질적으로 빠른가?"

논문의 핵심 답은 의외입니다. 특정 문제 클래스에서는 **샘플링은 차원 \(d\)** 에 대해 다항 시간 복잡도를 가지지만, 최적화는 **지수 시간 lower bound**를 가질 수 있습니다.

---

## 2. Setup: Local Nonconvexity

논문은 목적함수 \(U:\mathbb{R}^d \to \mathbb{R}\) 에 대해 아래를 가정합니다.

1. \(U\) 는 \(L\)-smooth (gradient Lipschitz)
2. 반지름 \(R\) 바깥에서는 \(m\)-strongly convex
3. \(\nabla U(0)=0\) (기술적 가정)

즉, 큰 영역에서는 잘 behaved하고, **안쪽 bounded region에서만 비볼록성**이 존재하는 구조입니다.

타깃 분포는

\[
p^*(x)\propto e^{-U(x)}
\]

이고, 조건수는 \(\kappa=L/m\) 로 둡니다.

---

## 3. Methods Compared: GD vs Langevin

가장 단순화해서 보면 업데이트는 다음과 같습니다.

- Gradient Descent:
\[
x_{k+1}=x_k-h_k\nabla U(x_k)
\]
- ULA:
\[
x_{k+1}=x_k-h_k\nabla U(x_k)+\xi_k,\quad \xi_k\sim\mathcal{N}(0,2h_k I)
\]
- MALA: ULA proposal + Metropolis accept/reject

핵심은 ULA가 GD와 거의 같은 형태인데, 잡음을 넣어 전역 구조를 탐색한다는 점입니다.

---

## 4. Key Result 1: Polynomial-Time Sampling

논문의 Theorem 1(ULA/MALA mixing time upper bound)을 요약하면,
\[
\tau_{\text{ULA}}(\varepsilon)=\tilde{O}\!\left(e^{32LR^2}\kappa^2\frac{d}{\varepsilon^2}\right),
\]
\[
\tau_{\text{MALA}}(\varepsilon)=\tilde{O}\!\left(e^{40LR^2}\kappa^{3/2}d^{1/2}(d\log\kappa+\log(1/\varepsilon))^{3/2}\right).
\]

\(LR^2=O(\log d)\)이면, 차원 \(d\)에 대해 **다항 시간**으로 제어됩니다.

논문의 이론 전개는
- 연속시간 확산과정의 KL 감소율 분석
- log-Sobolev constant 하한 도출
- 이산화(ULA/MALA) 오차 제어
순서로 진행됩니다.

---

## 5. Key Result 2: Exponential Lower Bound for Optimization

Theorem 2는 더 강합니다.

\[
K=\Omega\!\left(\left(\frac{LR^2}{\varepsilon}\right)^{d/2}\right)
\]

즉, 함수값/고계도함수 질의를 허용하는 매우 넓은 알고리즘 클래스에서도,  
\(\varepsilon\)-최적해를 찾는 데 필요한 반복 수가 차원에 대해 지수적으로 커질 수 있습니다.

직관은 "볼 안에 지수 개의 작은 basin을 packing할 수 있다"는 구성입니다.  
최적화는 "어느 basin이 진짜 global optimum인지"를 맞혀야 하므로 조합 폭발이 생깁니다.

---

## 6. Can Annealed Sampling Bypass Optimization?

논문은 \(q_\beta^*(x)\propto e^{-\beta U(x)}\) 를 크게 sharpen해서  
최적점 근처를 샘플링하는 아이디어(시뮬레이티드 어닐링 계열)도 함께 점검합니다.

결과적으로, 높은 확률로 \(\varepsilon\)-정확도를 얻으려면  
\(\beta=\tilde{\Omega}(d/\varepsilon)\)가 필요해져서 전체 계산량은 다시 지수 스케일이 됩니다.

즉, 이 설정에서는 "샘플링을 최적화 대용으로 쓰는 것"까지는 쉽지 않다는 메시지입니다.

---

## 7. Gaussian Mixture Model Case Study

논문은 GMM mean 추정에서 ULA/MALA와 EM을 비교합니다.

- 조건 \(MR^2=O(\log d)\) 하에서 ULA/MALA는 이론적으로 다항 복잡도
- 반면 EM은 초기화가 나쁘면 고차원에서 급격히 어려워짐

실험(Figure 2)에서도 차원이 커질수록
- ULA는 비교적 완만하게 증가
- EM은 필요한 gradient query가 급증
하는 경향을 보여줍니다.

---

## 8. Key Takeaways

1. 이 논문은 "샘플링 vs 최적화"를 정확히 비교하기 위해  
   **동일한 local nonconvex class** 위에서 복잡도 경계를 맞춰 제시합니다.

2. 샘플링 복잡도는 확률질량의 "전역 구조"에 더 민감하고,  
   최적화 복잡도는 "국소 basin 탐색"의 조합 복잡도에 민감하다는 대비가 명확합니다.

3. "샘플링이 항상 느리다"는 통념은 convex 세계에서는 맞지만,  
   nonconvex 세계에서는 반례가 충분히 생길 수 있음을 보여줍니다.

---

## 9. Limitations and Practical Interpretation

- 결과는 **worst-case 복잡도 경계**입니다. 모든 실제 문제에서 샘플링이 빠르다는 뜻은 아닙니다.
- 상수항 \(e^{O(LR^2)}\) 가 크면 이론적 다항성이 실무 시간으로 바로 이어지지 않을 수 있습니다.
- MCMC는 mixing/diagnostics가 필수라 구현 난이도는 여전히 있습니다.

---

## 10. Conclusion

이 논문의 가장 중요한 메시지는 아래 한 줄입니다.

> 비볼록 문제에서는, "최적화가 샘플링보다 무조건 빠르다"는 일반 명제가 성립하지 않는다.

특히 mixture-like landscape처럼 basin이 많은 구조에서는  
샘플링이 전역 구조를 따라 다항 시간으로 움직일 수 있고,  
최적화는 basin 식별 자체가 지수적으로 어려울 수 있다는 점이 인상적입니다.

---

## Reference

- Ma, Y.-A., Chen, Y., Jin, C., Flammarion, N., & Jordan, M. I. (2019).  
  *Sampling Can Be Faster Than Optimization*. arXiv:1811.08413.  
  https://arxiv.org/abs/1811.08413

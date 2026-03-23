---
layout: single
title: "Approximate Bayesian Computation Posterior"
date: 2026-03-23
permalink: /abc-posterior/
categories:
  - Statistics
  - Bayesian
  - Computation
tags:
  - Approximate Bayesian Computation
  - ABC
  - Likelihood-free Inference

toc: true
toc_sticky: true
comments: true
---

> 이 포스팅은 likelihood를 직접 계산하기 어려운 상황에서 사용되는 **Approximate Bayesian Computation (ABC)** 와, 그 결과 얻어지는 **ABC posterior**의 의미를 정리합니다.


---

## Introduction

베이지안 추론의 목표는 관측 데이터 $y_{obs}$가 주어졌을 때 파라미터 $\theta$의 사후분포

$$
p(\theta \mid y_{obs})
\propto
p(y_{obs}\mid\theta)\,p(\theta)
$$

를 계산하는 것입니다.

하지만 실제 문제에서는

- likelihood를 닫힌 형태로 쓸 수 없거나
- likelihood 계산이 지나치게 비싸거나
- 모델은 시뮬레이터 형태로만 주어지고 확률밀도는 직접 계산할 수 없는

경우가 자주 등장합니다.

예를 들어 population genetics, epidemiology, ecology, agent-based model 같은 분야에서는

> "주어진 $\theta$에서 데이터를 **생성(simulate)** 할 수는 있지만,  
> $p(y_{obs}\mid\theta)$를 **직접 평가(evaluate)** 하기는 어렵다."

는 상황이 흔합니다.

이럴 때 사용하는 대표적인 likelihood-free inference 방법이 바로 **Approximate Bayesian Computation (ABC)** 입니다.


---

## ABC의 핵심 아이디어

ABC의 아이디어는 단순합니다.

1. Prior에서 $\theta$를 뽑는다.
2. 그 $\theta$로부터 가짜 데이터 $y^\star$를 시뮬레이션한다.
3. $y^\star$가 실제 데이터 $y_{obs}$와 충분히 비슷하면 그 $\theta$를 accept한다.

즉, likelihood를 직접 계산하는 대신

> "이 파라미터가 실제 데이터를 **그럴듯하게 재현하는가?**"

를 기준으로 posterior를 구성합니다.

가장 단순한 형태의 rejection ABC는 아래와 같습니다.

```python
for t in range(T):
    theta = sample_prior()
    y_star = simulate(theta)

    if distance(S(y_star), S(y_obs)) <= epsilon:
        accept(theta)
```

여기서

- $S(\cdot)$ : summary statistic
- $\rho(\cdot,\cdot)$ : distance function
- $\epsilon > 0$ : tolerance

입니다.


---

## ABC Posterior의 정의

관측 데이터 전체를 그대로 비교하기보다, 일반적으로는 요약통계량

$$
s_{obs} = S(y_{obs})
$$

를 사용합니다.

이때 rejection ABC로 accept된 $\theta$들은 다음 분포를 따릅니다.

$$
\pi_\epsilon(\theta \mid s_{obs})
\propto
\pi(\theta)\,
\Pr_\theta\!\left(
\rho\big(S(Y), s_{obs}\big)\le \epsilon
\right)
$$

또는 적분 형태로

$$
\pi_\epsilon(\theta \mid s_{obs})
\propto
\pi(\theta)
\int
\mathbf{1}\!\left\{
\rho\big(S(y), s_{obs}\big)\le\epsilon
\right\}
p(y\mid\theta)\,dy
$$

로 쓸 수 있습니다.

이 분포가 바로 **ABC posterior** 입니다.

이 식의 의미는 명확합니다.

- Prior $\pi(\theta)$에서 시작하고
- 그 $\theta$가 만들어내는 데이터가 관측 요약통계량과 가까울수록
- 더 큰 가중치를 받아 posterior에서 더 자주 남게 됩니다.

즉 ABC posterior는

> **관측 데이터와 비슷한 시뮬레이션을 만들어내는 파라미터에 확률질량을 집중시키는 분포**

라고 이해할 수 있습니다.


---

## ABC Posterior는 무엇을 근사하는가?

이 부분이 가장 중요합니다.

ABC posterior는 일반적으로 원래의 정확한 posterior

$$
p(\theta\mid y_{obs})
$$

를 바로 주는 것이 아닙니다. 실제로는 두 단계의 근사가 들어갑니다.

첫 번째는 **summary statistic 근사**입니다.

ABC는 보통 $y_{obs}$ 전체가 아니라 $S(y_{obs})$만 사용하므로, $\epsilon \to 0$라고 해도 보통은

$$
\pi_\epsilon(\theta \mid s_{obs})
\to
p(\theta \mid S(y_{obs}))
$$

에 가까워집니다.

따라서 $S(\cdot)$가 sufficient statistic이 아닐 경우,

$$
p(\theta\mid S(y_{obs})) \neq p(\theta\mid y_{obs})
$$

가 되어 정보 손실이 발생합니다.

두 번째는 **tolerance 근사**입니다.

$\epsilon > 0$이면 exact match가 아니라 "충분히 가까움"만 요구하므로, 추가적인 smoothing이 들어갑니다. 즉

$$
\pi_\epsilon(\theta \mid s_{obs})
\approx
p(\theta \mid S(y_{obs}))
$$

이며, $\epsilon$이 클수록 근사는 거칠어집니다.

정리하면 ABC의 오차는 대체로 다음 두 가지에서 옵니다.

1. **summary statistic 선택으로 인한 정보 손실**
2. **finite tolerance $\epsilon$로 인한 smoothing bias**

따라서 ABC posterior를 해석할 때는

> "이것은 true posterior의 근사인가?"  
> 보다는  
> "어떤 summary와 어떤 tolerance 아래에서 정의된 posterior인가?"

를 먼저 보는 것이 중요합니다.


---

## Kernel ABC 관점

위의 rejection ABC는 indicator kernel

$$
\mathbf{1}\{\rho(S(y), s_{obs})\le \epsilon\}
$$

을 사용한 형태로 볼 수 있습니다.

보다 일반적으로는 kernel weight를 사용해

$$
\pi_\epsilon^K(\theta \mid s_{obs})
\propto
\pi(\theta)
\int
K_\epsilon\!\left(
\rho(S(y), s_{obs})
\right)
p(y\mid\theta)\,dy
$$

로 정의할 수 있습니다.

여기서 $K_\epsilon(\cdot)$는 거리가 작을수록 큰 값을 주는 kernel입니다.

이 관점의 장점은 acceptance/rejection만 하는 대신, 관측치에 더 가까운 시뮬레이션에 더 큰 가중치를 줄 수 있다는 점입니다. 그래서 ABC posterior는 단순한 "필터링 결과"라기보다,

> **관측 요약통계량 주변을 kernel로 부드럽게 감싼 posterior**

로도 해석할 수 있습니다.


---

## Summary Statistic의 역할

ABC에서 summary statistic은 사실상 모델링의 핵심입니다.

이상적으로는 low-dimensional이면서도 $\theta$에 대한 정보를 충분히 보존하는 $S(y)$가 필요합니다. 하지만 실제로는 두 요구가 충돌합니다.

- summary dimension이 너무 크면 관측치와 시뮬레이션이 가까워질 확률이 급격히 줄어듭니다.
- 너무 단순한 summary를 쓰면 중요한 정보가 사라져 posterior가 지나치게 넓어지거나 왜곡됩니다.

즉 ABC는 흔히

> **likelihood-free** 이지만,  
> **summary-design-free** 하지는 않습니다.

고차원 데이터에서 raw data 자체를 비교하면 acceptance rate가 거의 0에 가까워지는 경우가 많기 때문에, 평균, 분산, quantile, autocorrelation, regression coefficient 같은 domain-specific summary가 많이 사용됩니다.

최근에는 neural network를 이용해 informative summary를 학습하는 방향도 많이 연구되고 있습니다.


---

## Tolerance $\epsilon$의 역할

$\epsilon$은 계산 효율과 정확도 사이의 trade-off를 조절합니다.

- $\epsilon$이 작을수록 posterior는 더 정밀해지지만 accept되는 샘플 수가 줄어듭니다.
- $\epsilon$이 클수록 계산은 쉬워지지만 posterior bias가 커집니다.

연속형 데이터에서는 exact equality가 사실상 확률 0이기 때문에, $\epsilon$은 거의 항상 필요합니다.

실무에서는 보통

- 파일럿 시뮬레이션으로 distance 분포를 본 뒤
- 하위 몇 % quantile을 기준으로 $\epsilon$을 정하거나
- SMC-ABC에서 점차 $\epsilon$을 줄여가는

방식을 사용합니다.


---

## 대표적인 ABC 알고리즘

### Rejection ABC

가장 단순하고 직관적인 방법입니다. 구현이 쉽지만 acceptance rate가 매우 낮을 수 있습니다.

### MCMC-ABC

[MCMC](/mcmc/)의 proposal 구조를 이용하되, likelihood 대신 ABC acceptance criterion을 사용합니다. prior 샘플링보다 효율적일 수 있지만 mixing이 느려질 수 있습니다.

### SMC-ABC

큰 tolerance에서 시작하여 점차 작은 tolerance로 내려가면서 particle을 이동시킵니다. 현재는 실무에서 가장 널리 쓰이는 방식 중 하나입니다.

ABC posterior를 더 정교하게 근사하면서도 rejection ABC보다 훨씬 효율적입니다.


---

## ABC Posterior의 장점과 한계

### 장점

1. **Likelihood-free inference**  
   likelihood를 직접 계산할 수 없는 simulator-based model에도 적용 가능합니다.

2. **모델 유연성**  
   확률밀도를 수식으로 쓰지 못해도 시뮬레이터만 있으면 posterior inference를 시도할 수 있습니다.

3. **직관적인 해석**  
   "관측 데이터를 얼마나 잘 재현하는가"라는 기준으로 posterior를 구성하므로 이해가 비교적 쉽습니다.

### 한계

1. **Summary statistic 의존성**  
   어떤 summary를 쓰느냐에 따라 posterior가 크게 달라질 수 있습니다.

2. **Curse of dimensionality**  
   데이터나 summary 차원이 커지면 accept/reject 기반 접근이 급격히 비효율적이 됩니다.

3. **Approximation bias**  
   $\epsilon$과 summary choice 때문에 true posterior와 차이가 날 수 있습니다.

4. **시뮬레이션 비용**  
   하나의 posterior sample을 얻기 위해 매우 많은 시뮬레이션이 필요할 수 있습니다.


---

## Summary

- **ABC posterior**는 likelihood를 직접 계산하는 대신, 시뮬레이션 데이터가 관측 데이터와 얼마나 가까운지를 바탕으로 정의되는 사후분포입니다.
- rejection ABC에서 accept된 파라미터의 분포는

$$
\pi_\epsilon(\theta \mid s_{obs})
\propto
\pi(\theta)\,
\Pr_\theta\!\left(
\rho\big(S(Y), s_{obs}\big)\le\epsilon
\right)
$$

로 표현됩니다.

- $\epsilon \to 0$이면 보통 $p(\theta\mid S(y_{obs}))$에 가까워지며, $S(\cdot)$가 sufficient statistic일 때에만 true posterior에 수렴합니다.
- 따라서 ABC의 성능은 **simulator**, **summary statistic**, **distance**, **tolerance**의 네 요소에 의해 결정됩니다.

결국 ABC posterior는 단순히 "근사 posterior"라기보다,

> **요약된 데이터와 허용 오차 아래에서 정의된 likelihood-free posterior**

라고 보는 것이 가장 정확합니다.


---

## References

- [Beaumont, M. A., Zhang, W., & Balding, D. J. (2002). Approximate Bayesian Computation in Population Genetics.](https://www.genetics.org/content/162/4/2025)
- [Marin, J.-M., Pudlo, P., Robert, C. P., & Ryder, R. J. (2012). Approximate Bayesian Computational Methods.](https://doi.org/10.1214/11-STS378)
- [Sisson, S. A., Fan, Y., & Beaumont, M. (2018). Handbook of Approximate Bayesian Computation.](https://www.routledge.com/Handbook-of-Approximate-Bayesian-Computation/Sisson-Fan-Beaumont/p/book/9781138633215)
- [Lintusaari, J., Gutmann, M. U., Dutta, R., Kaski, S., & Corander, J. (2017). Fundamentals and Recent Developments in Approximate Bayesian Computation.](https://arxiv.org/abs/1707.01254)

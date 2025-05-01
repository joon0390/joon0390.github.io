---
title: "MCMC Sampling in Bayesian Models"
tag :
        - Bayesian
        - Sampling
        - MCMC
        - Gibbs Sampling
        - Metropolis-Hastings
        - Algorithm

permalink: /study/mcmc-sampling
date: "2025-04-23"
---

# MCMC Sampling in Bayesian Models
> 본 포스팅에서는 베이지안 추론에서 핵심이 되는 MCMC(Markov Chain Monte Carlo) 알고리즘의 원리와 구현 방식을 자세히 살펴봅니다.  


---

## Introduction

많은 베이지안 모델은 사후분포를 구하고 이를 이용해 원하는 값을 계산하는 것을 목표로 합니다.
<br/>

관측 데이터 $\mathbf{y}$가 주어졌을 때, 파라미터 $\theta$의 사후분포는 다음과 같이이 Bayes' rule에 의해 계산됩니다.
<br/>

$$
p(\theta \mid \mathbf{y})=\dfrac{p(\mathbf{y}\mid \theta)\,p(\theta)}{p(\mathbf{y})}
$$

<br/>

하지만 분모에 등장하는 정규화 상수 혹은 Evidence $p(\mathbf{y})$는 다음과 같은 고차원 적분으로 정의됩니다.

$$
p(\mathbf{y})=\int p(\mathbf{y}∣\theta)p(\theta)d\theta
$$

<br/>

따라서 대부분의 실제 모델에서는 이 적분을 closef form으로 계산하기가 거의 불가능합니다. 특히 파라미터 $\theta$의 차원이 커짐에 따라 차원의 저주로 인해 수치적분 방식은 계산의 비용이 급격하게 증가하게 됩니다.

##### > 그래서 직접적 계산이 아닌, 다른 방식의 접근이 필요합니다.

---

## Markov Chain

>[Markov Chain](https://en.wikipedia.org/wiki/Markov_chain)은 현재 상태가 오직 이전 상태에만 의존하는 확률 과정(stochastic process)입니다. 
<br/>

<figure style="text-align: center; margin: 2em 0;">
    <img
        src = '/assets/img/MCMC/markov_chain.png'
        alt = "markov_chain"
        width = "200"
    >
    <figcaption style="text-align: center;">
        Markov Chain
    </figcaption>
</figure>
<br/>

##### Markov Chain은 다음과 같은 구성 요소로 정의됩니다:

1. **Markov Property(무기억성)**  
   - 현재 상태 $X_t$만이 다음 상태 $X_{t+1}$에 영향을 미치며, 과거 상태 $X_{t-1},X_{t-2},…$에는 직접 의존하지 않습니다.  
     수식: $P(X_{t+1}=x'\mid X_t=x, …) = P(X_{t+1}=x'\mid X_t=x)$
<br/>

2. **State Space**  
   - 이산 또는 연속 공간이며, 각각의 $x$가 가능한 상태를 나타냅니다.
<br/>

3. **Transition Kernel(전이 커널)**  
   - $K(x\to x') = P(X_{t+1}=x'\mid X_t=x)$.  
   - n-step transition: $K^n(x\to x') = P(X_{t+n}=x'\mid X_t=x)$.  
   - Chapman–Kolmogorov: $K^{m+n}(x\to z) = \int K^m(x\to y)\,K^n(y\to z)\,dy$.
<br/>

4. **Stationary Distribution(정착 분포)**  
   - 분포 $\pi(x)$가 전이 커널의 고정점이 되려면  
     $\pi(x') = \int \pi(x)\,K(x\to x')\,dx$  
     를 만족해야 합니다.  
   - 유한 상태 체인의 경우 $\pi = \pi K$로 표현합니다.
<br/>

5. **Ergodicity**  
   - Irreducibility(귀환성), Aperiodicity(주기 없음), Positive recurrence(양의 재귀성)  
   - 이 세 조건이 만족되면  
     $\lim_{t\to\infty}P(X_t=x)=\pi(x)$가 성립합니다.
<br/>

6. **Detailed Balance**  
   - Metropolis–Hastings에서 자주 사용되는 조건으로  
     $\pi(x)\,K(x\to y)=\pi(y)\,K(y\to x)$  
     가 성립하면 $\pi$는 정착분포가 됩니다.



MCMC에서는 이 전이 커널을 목표 사후분포 $p(\theta\mid y)$가 Stationary distribution가 되도록 설계합니다. 
충분히 오랜시간 동안 반복하면, 상태 분포가 $p(\theta\mid y)$에 가까워지게 됩니다. 






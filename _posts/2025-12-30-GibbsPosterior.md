---
layout: single
title: "Gibbs Posteriors"
date: 2025-12-30
permalink: /gibbs-posteriors/
categories:
  - Statistics
  - Bayesian
tags:
  - Gibbs Posteriors
  - Generalized Bayes
  - PAC-Bayes

toc: true
toc_sticky: true
comments: true
---

> 이 포스팅은 모델 Misspecification에 강건한 **Gibbs Posteriors** (or **Generalized Posteriors**)에 대해 정리합니다.



## Introduction: Standard Bayes의 한계

이전 포스팅에서 다룬 [**Variational Inference**](/variational-inference/)를 포함한 대부분의 베이지안 추론은 "우리가 가정한 Likelihood 모델 $p(y \mid \theta)$가 실제 데이터 생성 과정을 포함하고 있다"고 가정합니다. 

하지만 실제 데이터는 다음과 같은 상황에 처할 때가 많습니다.

1.  **Model Misspecification**: 실제 분포가 가우시안이 아닌데 가우시안 Likelihood를 사용하는 경우.
2.  **Outliers**: 소수의 이상치가 Likelihood에 과도한 영향을 주어 Posterior를 왜곡하는 경우.
3.  **Complex Loss**: 단순히 데이터의 생성 확률을 높이는 것보다, 특정 Loss 함수(예: Hinge loss, MAE)를 최소화하는 파라미터 $\theta$를 찾고 싶은 경우.

이런 상황에서 Likelihood 대신 **임의의 Loss 함수를 직접 사용하여 Posterior를 정의**하는 방법이 바로 **Gibbs Posteriors**입니다.


</br>

## Gibbs Posterior의 정의

Standard Bayesian의 Posterior는 다음과 같습니다.
$$p(\theta \mid y) \propto \exp\left( \log p(y \mid \theta) \right) p(\theta)$$

여기서 $\log p(y \mid \theta)$를 일반적인 Loss 함수 $L_n(y, \theta)$로 치환하고, 학습의 속도를 조절하는 **Learning rate $\eta$** 를 도입하면 **Gibbs Posterior**가 정의됩니다.

$$
\pi_n(\theta) \propto \exp\left( -\eta \cdot L_n(y, \theta) \right) \pi_0(\theta)
$$

- $L_n(y, \theta)$: 데이터를 평가할 Loss 함수 (ex: $\sum (y_i - f_\theta(x_i))^2$)
- $\pi_0(\theta)$: Prior 
- $\eta > 0$: **Learning rate** (또는 Inverse Temperature). 데이터로부터 얼마나 적극적으로 정보를 수용할지 결정합니다.


</br>

## 왜 "Gibbs"인가?

이 명칭은 통계 역학의 [**Gibbs Distribution** or Boltzmann Distribution](https://en.wikipedia.org/wiki/Boltzmann_distribution)에서 유래했습니다. 에너지(Loss)가 낮은 상태에 더 높은 확률을 부여하는 구조가 동일하기 때문입니다.

- **Energy** $\leftrightarrow$ **Loss Function**
- **Temperature** $\leftrightarrow$ **$1/\eta$**

$\eta$가 커질수록(온도가 낮아질수록) Posterior는 Loss를 최소화하는 지점 주변으로 아주 좁게 집중됩니다.



</br>

## Variational Gibbs Inference (VGI)

Gibbs Posterior 역시 정규화 상수를 구하기 어렵기 때문에, VI를 통해 근사할 수 있습니다. Gibbs Posterior를 타겟으로 하는 VI의 목적 함수(Generalized ELBO)는 다음과 같습니다.

$$
\mathcal{L}_{Gibbs}(q) = \mathbb{E}_q[-\eta L_n(y, \theta)] - \text{KL}(q(\theta) \,\|\, \pi_0(\theta))
$$

이를 최대화하는 것은 결국 다음의 두 항을 최적화하는 것과 같습니다.
1.  **$\mathbb{E}_q[L_n(y, \theta)]$ 최소화**: 기대 손실을 줄여 데이터에 적합시킴.
2.  **$\text{KL}(q \| \pi_0)$ 최소화**: Prior에서 너무 멀어지지 않도록 규제(Regularization).

이 수식은 **PAC-Bayesian Bound**와도 직접적으로 연결됩니다. 특정 조건하에서 이 목적 함수를 최적화하여 얻은 $q$는 본 적 없는 데이터에 대한 Generalization Error의 상한을 최소화하는 분포임이 증명되어 있습니다.

</br>

## 핵심 파라미터: $\eta$ 의 선택

Gibbs Posterior에서 가장 까다로운 부분은 $\eta$의 값을 결정하는 것입니다.

- $\eta$가 너무 크면: 모델이 데이터의 노이즈나 이상치에 과적합(Overfitting)됩니다.
- $\eta$가 너무 작으면: 데이터로부터 충분히 배우지 못하고 Prior에 머물게 됩니다(Underfitting).

최근 연구들(SafeBayes 등)은 데이터의 증거(evidence)를 기반으로 최적의 $\eta$를 자동으로 선택하는 알고리즘을 제안하고 있습니다.

</br>


## Gibbs Posteriors의 장점

1.  **Robustness**: 이상치에 민감한 Squared error 대신 Huber loss나 Absolute error를 사용하여 더 강건한 추론이 가능합니다.
2.  **Flexibility**: 모델의 확률적 구조(Likelihood)를 엄밀하게 설계하지 않아도, 목적에 맞는 Loss만 있다면 베이지안 추론의 틀을 사용할 수 있습니다.
3.  **Bayesian Deep Learning과의 연관성**: 신경망에서 사용하는 Cross-entropy 등을 Loss로 직접 사용하여 Bayesian Neural Networks를 구성할 때 이론적 근거를 제공합니다.

</br>

## Summary

- **Gibbs Posterior**는 Likelihood 대신 Loss 함수를 지수화하여 Posterior를 구성하는 방식이다.
- 모델 오설정이나 이상치가 존재하는 실제 환경에서 Standard Bayes보다 강건한 성능을 보인다.

</br>

## Reference

- [Bissiri, P. G., et al. (2016). A General Framework for Updating Belief Distributions.](https://academic.oup.com/jrsssb/article/78/5/1103/7036237)
- [Gneiting, T., & Raftery, A. E. (2007). Strictly Proper Scoring Rules, Prediction, and Estimation.](https://www.tandfonline.com/doi/abs/10.1198/016214506000001437)
- [Alquier, P. (2021). User's Guide to PAC-Bayes.](https://arxiv.org/abs/2110.11216)
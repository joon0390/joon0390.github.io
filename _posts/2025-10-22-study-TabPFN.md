---
layout: single
title: "TabPFN: Foundation Models for Tabular Data"
date: 2025-10-22
categories: 
    - Bayesian
    - Deep Learning
    - Machine Learning
tags: 
    - TabPFN
    - Transformer
    - Meta-learning
---

## Introduction: The Last Frontier of Tabular Data

> 딥러닝은 이미지·텍스트·음성 분야에서는 이미 인류를 능가했지만, 표형 데이터(tabular data)에서는 여전히 XGBoost, LightGBM, CatBoost의 부스팅 삼총사가 자리를 지키고 있습니다.  
> TabPFN은 이 오래된 문제를 ‘**prior를 학습한 사전 확률 모델(Prior-Founded Network)**’이라는 관점으로 새롭게 접근합니다.

---

## Core Idea: A Model That Can Infer "Without Training"

- PFN(Pretrained Foundation Network)은 작은 표형 데이터셋을 **meta-learning** 방식으로 학습한 거대한 사전분포(prior distribution)입니다.  
- 훈련 과정에서 수천 개의 인공 데이터셋을 생성하고, Bayesian posterior를 근사하도록 Transformer를 학습시킵니다.  
- 새로운 데이터셋이 주어지면 TabPFN은 단 한 번의 forward pass로 예측합니다 — “Training-Free Inference.”

---

## Architecture and Training Procedure

- 구조적으로는 **Transformer Encoder**를 사용하지만, 각 sample을 **token**, 각 feature를 **dimension**으로 취급합니다.
- self-attention은 feature 간 관계가 아니라 **sample 간 관계**를 포착합니다.  
- Meta-training data는 Bayesian linear regression, GP, decision tree 등 다양한 synthetic generator로 만듭니다.
- 손실은 posterior predictive likelihood를 근사하는 방향으로 설정합니다.

---

## Comparison with Existing Models

| 모델 | 학습 패러다임 | 훈련 비용 | 데이터 의존성 | 해석력 |
|------|----------------|------------|----------------|--------|
| XGBoost | supervised | 낮음 | domain-specific | 높음 |
| FT-Transformer | deep supervised | 높음 | 실제 데이터 | 중간 |
| **TabPFN** | meta-learning (Bayesian prior) | 매우 높음 | synthetic meta-data | 중간~낮음 |

> TabPFN의 강점은 “새로운 데이터셋에서도 즉시 예측이 가능하다”는 점이지만,  
> meta-train 분포와 실제 데이터 분포가 다를 경우(**prior mismatch**) 정확도가 급격히 떨어지게 됩니다.

---

## A Philosophical Detour: Empirical Bayes and the Collective Prior

Ryan Giordano (2025)은 자신의 글  [*“TabPFN and the Long Empirical Bayes of Supervised Learning”*](https://rgiordan.github.io/posts/2025-08-25-tabpfn.html)에서 TabPFN을 베이지안 관점에서 매우 흥미롭게 해석했습니다. 그는 TabPFN을 **“SBI on SBI”**, 즉 *감독학습 과제들의 공간 위에서 또 한 번의 SBI를 수행하는 모델*로 봅니다.

Giordan의 핵심 주장은 다음과 같습니다.  

"TabPFN이 사용하는 사전분포는 **무작위로 생성된 결정 트리, 다층 퍼셉트론, 특징 변환들**로 구성된다. 그 이유는 지난 수십 년간의 감독학습 연구가 “이러한 함수 형태들이 실제 세상의 관계를 표현하는 데 유용하다”는 사실을 경험적으로 보여줬기 때문이다."

따라서 그는 이렇게 말합니다.
 
> “TabPFN is basically SBI on SBI for univariate supervised learning posterior predictive targets.”

즉, 우리가 지금까지 다양한 모델과 데이터셋으로 쌓아온 감독학습의 역사는 결국 “**함수 형태의 공간을 탐색하며 프라이어를 학습해 온 거대한 경험적 베이지안 과정**”이었다는 것입니다. 수많은 벤치마크 결과, Kaggle 대회, 튜토리얼, 블로그 팁들이 결국 현실적 데이터 생성 구조를 가장 잘 설명하는 함수 형태를 향한 작은 노이즈 가중 경사 상승(noisy hill climb)이었다는 해석입니다.

이 관점에서 보면 TabPFN은 그 집단적 탐색의 정점에 있는 **경험적 베이지안 사전의 압축물**이라 할 수 있습니다.  
하지만 Giordan은 동시에 경고합니다.  
> “If we now rely too much on TabPFN, we run the risk of terminating the empirical Bayes learning process too early.”

즉, 우리가 이제 이 ‘동결된 메타 프라이어(meta-prior)’에 너무 의존한다면, 그동안 커뮤니티가 함께 쌓아 온 탐색 과정을 조기에 멈춰 버릴 수 있다는 것입니다. TabPFN의 성공은 놀랍지만, 그만큼 “탐색의 다양성”을 유지하려는 노력이 필요하다는 철학적 성찰을 남깁니다.

---

## Critical Discussion

- Bayesian 관점에서 TabPFN은 **approximate posterior sampler**로 볼 수 있습니다.
- 하지만 meta-training에서 사용하는 prior는 “암묵적”이며, posterior uncertainty가 완전히 보존되지 않습니다.  
- 실제로 UCI benchmark 외 도메인(예: sensor, biomedical)에서는 성능이 일관되지 않습니다.
- 즉, “훈련 없는 추론”은 가능하지만 “불확실성 추론”은 여전히 어렵습니다.

---

## Future Directions

- GP-style prior나 Flow-based meta-training을 결합한 확률적 PFN  
- FT-Transformer와의 hybrid (feature-wise embedding + meta-prior)  
- Bayesian TabPFN: uncertainty-aware output layer 설계

---

## Conclusion

> TabPFN은 “tabular data에 대한 foundation model”이라는 개념을 처음으로 실현한 사례입니다.  
> 그러나 진정한 foundation model이 되려면, meta-prior의 다양화와 uncertainty quantification의 통합이 필요합니다.

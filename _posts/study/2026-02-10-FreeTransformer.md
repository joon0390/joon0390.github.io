---
layout: single
title: "[Paper Review] The Free Transformer"
date: 2026-02-10
permalink: /freetransformer/
categories:
  - Deep Learning
  - Bayesian
tags:
  - Transformer
  - VAE
  - Latent Variable
  - Autoregressive Modeling
  - The Free Transformer

toc: true
toc_sticky: true
comments: true
---

> 이 포스팅은 François Fleuret의 논문 **The Free Transformer (arXiv:2510.17558)** 를 읽고 정리한 글입니다.

---

## 1. 왜 이 논문이 중요한가?

Decoder-only Transformer는 다음 토큰 확률을 잘 모델링합니다.

$$
p(S)=\prod_{t=1}^{T}p(s_t \mid s_{<t})
$$

이 구조는 강력하지만, 논문은 한 가지 구조적 한계를 지적합니다.

- 전역적인 결정(예: 문장 전체의 감정, 답안의 전체 계획)도
- 결국 토큰 히스토리 안에서 **뒤늦게** 복원해야 한다.

즉, 이론적으로는 충분해도 실무적으로는 비효율이 생길 수 있다는 문제의식입니다.

---

## 2. 핵심 아이디어: Conditional Latent Variable

논문은 시퀀스 생성을 잠재변수 $Z$에 조건부로 둡니다.

$$
p(S)=\sum_Z p(Z)\prod_{t=1}^{T}p(s_t \mid s_{<t}, Z_{\le t})
$$

직관은 단순합니다.

- $Z$가 “전역 계획” 역할을 맡고
- decoder는 해당 계획을 따라 토큰을 생성합니다.

즉, 토큰 단계와 계획 단계를 분리해서 학습하게 만듭니다.

---

## 3. 학습 방법: cVAE + Free Bits

학습 시에는 비인과(non-causal) encoder를 추가하여 $Q(Z\mid S)$를 만듭니다.

- **학습**: $Z \sim Q(Z\mid S)$를 샘플링해 decoder 학습
- **추론**: encoder 없이 prior $P(Z)$에서 $Z$ 샘플링 후 생성

목표함수는 재구성 손실 + KL 제약입니다.

$$ \mathcal{L} = \frac{1}{T}\sum_{t=1}^{T} \Big(\mathrm{CE}_t - \kappa_t \Big) $$

$$ \kappa_t = \max\left(0,\ \mathrm{KL}\left(Q(Z_t\mid S)\,\|\,P(Z_t)\right)-\kappa\right) $$

핵심은 token-wise free bits입니다.

- KL이 너무 작아 latent를 안 쓰는 문제를 막고
- KL이 너무 커서 정답을 통째로 latent에 복사하는 문제도 제어합니다.

논문에서는 $\kappa$를 통해 토큰당 허용 정보량(bit/token)을 제어합니다.

---

## 4. 아키텍처 관점에서의 변경점

논문의 장점 중 하나는 “큰 구조 변경 없이” 적용된다는 점입니다.

- decoder 중간층에 $Z$를 주입
- 학습 시에만 비인과 encoder 1개 블록 추가
- 추론에서는 encoder 제거

보고된 학습 오버헤드는 대략 3% 수준(모델 크기별로 약간 차이)입니다.

---

## 5. 실험 결과 요약

논문에서 보고한 대표 수치(요약):

### 5.1 1.5B / 47B tokens

- HumanEval+: **0.055 → 0.085**
- MBPP: **0.112 → 0.152**
- GSM8K: **0.025 → 0.033**

코드/수학 관련 지표에서 개선이 보입니다.

### 5.2 8B / 200B tokens

- HumanEval+: **0.159 → 0.189**
- MMLU: **0.359 → 0.398**
- CSQA: **0.356 → 0.450**

모델 크기를 키워도 이득이 유지되는 경향이 관찰됩니다.

### 5.3 8B / 1T tokens

- HumanEval+: **0.268 → 0.299**
- MMLU: **0.592 → 0.623**
- CSQA: **0.707 → 0.748**

논문은 고 $\kappa$ 설정(예: 4 bit/token)에서는 학습 붕괴 위험도 함께 보고합니다.

---

## 6. 내가 이해한 포인트

이 논문의 메시지는 다음 한 줄로 정리됩니다.

> "Transformer의 표현력 문제가 아니라, 생성 과정의 유도편향(inductive bias)을 바꾸자."

제가 특히 중요하다고 본 점:

1. 성능 향상 자체보다, **왜 향상되는지 설명 가능한 구조**를 제시했다.
2. 추론 비용을 크게 늘리지 않으면서 latent planning을 도입했다.
3. 모든 벤치마크를 올리는 만능 해법은 아니며, 태스크별 편차를 솔직히 보여줬다.

---

## 7. 짧은 결론

The Free Transformer는 “Autoregressive를 버리자”가 아니라,
**Autoregressive 위에 latent planning을 얹자**는 실용적 제안입니다.

구조 변경은 작지만, 코드/수학 추론 등 일부 영역에서는 의미 있는 개선을 보여줍니다.
앞으로 중요한 질문은 성능의 절대치보다,
이 latent planning이 어떤 문제 클래스에서 가장 강한지 이론적으로 밝히는 일입니다.

---

## Reference

- Fleuret, F. (2025). *The Free Transformer*. arXiv:2510.17558.  
  https://arxiv.org/abs/2510.17558

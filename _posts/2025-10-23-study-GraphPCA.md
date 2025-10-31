---
layout: single
title: "Graph PCA"
date: 2025-10-23
permalink: /graphpca/
categories:
  - Graph
  - Statistics
  - Bayesian
tags:
  - Dimensionality Reduction
  - GraphPCA
  - Manifold Learning
toc: true
toc_sticky: true
comments: true
---

> 이 포스팅은 [“GraphPCA: a fast and interpretable dimension reduction algorithm” (Yang et al., 2024)](https://pmc.ncbi.nlm.nih.gov/articles/PMC11545739/) 논문을 기반으로,  
> **일반적인 데이터와 그래프 구조를 동시에 반영하는 차원축소 방법** 으로서의 Graph PCA를 정리한 글입니다.

---

## Introduction
- Graph PCA는 **데이터의 저차원 표현**을 학습하면서,  
  동시에 **그래프 인접성(유사도)** 을 보존하도록 정규화 항을 추가한 **PCA 변형 기법**입니다.
- 즉, 표준 PCA가 재구성 오차만 최소화하는 반면,  
  Graph PCA는 인접한 샘플들의 저차원 임베딩이 **가깝게 유지**되도록 합니다.
- 논문에서는 이 문제를 닫힌형(closed-form) 해법으로 효율적으로 풀 수 있음을 보입니다.

---

## Model & Objective Function
데이터 행렬을 $X \in \mathbb{R}^{n \times d}$,  
그래프의 라플라시안 행렬을 $L = D - W$라 하면,  
Graph PCA는 다음 문제를 풉니다:

$$
\min_{Z,\,U} \; \|X - ZU^\top\|_F^2 + \lambda\,\mathrm{tr}(Z^\top L Z)
\quad \text{s.t. } U^\top U = I
$$

- $Z \in \mathbb{R}^{n\times q}$: 저차원 임베딩  
- $U \in \mathbb{R}^{d\times q}$: 로딩(주성분 축)  
- $\lambda \ge 0$: 그래프 제약 강도  
- $L$: 그래프 라플라시안, $D$는 차수행렬, $W$는 인접 가중치 행렬

---

## Optimization Derivation

### 1. $Z$-step  (고정 $U$)
$$
J(Z) = \|X - ZU^\top\|_F^2 + \lambda\,\mathrm{tr}(Z^\top L Z)
$$

이를 $Z$에 대해 미분하면:
$$
\frac{\partial J}{\partial Z} = -2XU + 2Z + 2\lambda LZ = 0
$$

따라서
$$
\boxed{(I + \lambda L)Z = XU \quad\Rightarrow\quad Z = (I + \lambda L)^{-1}XU}
$$

즉, 그래프 라플라시안이 저차원 표현 $Z$를 **스무딩(smoothing)** 하도록 작용합니다.

---

### 2. $U$-step  (고정 $Z$)
$$
\min_{U^\top U = I} \|X - ZU^\top\|_F^2
$$

이는 Orthogonal Procrustes Problem 형태로,  
$X^\top Z = P\Sigma Q^\top$ (SVD)라 하면

$$
\boxed{U = PQ^\top}
$$

이로써 $U$는 $X^\top Z$를 최적으로 정렬(회전)하는 직교행렬이 됩니다.

---

### 3. 전체 알고리즘
| 단계 | 내용 |
|------|------|
| 초기화 | 표준 PCA의 상위 q개 축으로 $U$ 초기화 |
| $Z$-step | $Z = (I + \lambda L)^{-1}XU$ |
| $U$-step | $U = \text{Procrustes}(X^\top Z)$ |
| 반복 | 두 단계를 교대로 수행해 수렴 시까지 |

논문에서는 이 과정을 반복 없이 한 번에 계산할 수 있는  
**닫힌형(closed-form) 솔루션**도 제시합니다.

---

## Graph Construction
- $W$: k-NN 그래프 또는 거리/유사도 기반 가중치 행렬  
- $L = D - W$: 기본 라플라시안  
- 정규화 버전 $L_{\text{sym}} = I - D^{-1/2}WD^{-1/2}$ 도 사용 가능  
- 입력 $X$는 특성 스케일 차이를 줄이기 위해 표준화(z-score) 권장

---

## Hyperparameter λ
- $\lambda$가 커질수록 그래프 매끄러움(이웃 보존)은 강화되고,  
  데이터 재구성 정확도는 감소합니다.  
- 일반적으로 작은 로그스케일 그리드에서 탐색:  
  $\lambda \in \{10^{-3}, 10^{-2}, 10^{-1}, 1, 10\}$
- 다운스트림 성능(클러스터링 / 분류 정확도 등)으로 선택 가능

---

## Related Methods
| 방법 | 특징 |
|------|------|
| PCA | 그래프 구조 고려 X |
| Laplacian Eigenmaps | 비선형 임베딩, 재구성 항 없음 |
| Graph PCA | PCA + 그래프 정규화 절충형 |
| LLE / Isomap | 거리 보존 기반 비선형 매니폴드 학습 |


---

## Reference
- **Yang J., Wang L., Liu L., Zheng X.**  
  *GraphPCA: a fast and interpretable dimension reduction algorithm.*  
  Genome Biology 2024, 25(1):287. [DOI: 10.1186/s13059-024-03429-x](https://doi.org/10.1186/s13059-024-03429-x)  
  [PMC11545739](https://pmc.ncbi.nlm.nih.gov/articles/PMC11545739/)

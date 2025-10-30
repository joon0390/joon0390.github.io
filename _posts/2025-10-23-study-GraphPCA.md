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

toc: true
toc_sticky: true
comments: true
---

## 개요
- 데이터의 저차원 표현을 학습하면서, 그래프 인접성(유사도)을 보존하도록 정규화 항을 추가한 PCA 변형입니다.
- 핵심은 그래프 라플라시안 $L$을 이용한 매끄러움(smoothness) 패널티 $λ tr(Z^T L Z)$ 입니다. 이 항은 그래프에서 이웃한 노드들의 임베딩이 가깝도록 유도합니다.

---

## 모델과 목적함수
- 데이터 행렬: $X ∈ R^{N×d}$, 임베딩: $Z ∈ R^{N×q}$, 로딩(주성분 축): $U ∈ R^{d×q}$ with $U^T U = I$.
- 그래프: 가중치 행렬 $W$, 차수행렬 $D$, 라플라시안 $L = D - W$ (또는 정규화 라플라시안 사용 가능).

목적함수 (Graph-regularized PCA):
$$
\min_{Z, U} \; \|X - Z U^\top\|_F^2 
\; + \; \lambda\,\mathrm{tr}(Z^\top L Z)
\quad\text{s.t. } U^\top U = I.
$$

해석:
- 첫 항은 재구성 오차를 최소화하는 PCA 항, 둘째 항은 그래프 매끄러움 정규화입니다.
- $tr(Z^T L Z) = 1/2 \sum_{i,j} W_{ij} \|z_i - z_j\|^2$ 이므로 이웃 임베딩 차이를 줄입니다.

---

## 최적화 (간단한 교대최적화)
1) $Z$-스텝 (고정 $U$):
   - 일차조건으로 $2(Z - XU) + 2\lambda LZ = 0$ →
     $$(I + \lambda L)Z = XU.$$
   - 선형계 풀이로 $Z = (I + \lambda L)^{-1} XU$.

2) $U$-스텝 (고정 $Z$):
   - $U^T U = I$ 제약하의 Procrustes 문제.
   - $X^T Z = P \Sigma Q^T$ (SVD)라 하면 $U = P Q^T$.

실행 요약:
- 초기화: 표준 PCA의 상위 $q$개 주성분으로 $U$ 초기화.
- 반복: $Z$-스텝 → $U$-스텝 수회 반복 후 수렴.

---

## 그래프 구성 팁
- $W$: k-NN 그래프(코사인/가우시안 커널), 또는 ε-이웃 그래프 사용.
- $L$: 기본 $L = D - W$ 또는 정규화 $L_{sym} = I - D^{-1/2} W D^{-1/2}$.
- 특성 스케일에 민감하므로, $X$는 표준화(각 특성 z-정규화) 권장.

---

## λ 선택 가이드
- $λ$가 클수록 그래프 매끄러움 강조(지역구조 보존 ↑, 데이터 재구성 ↓).
- 작은 그리드 검색 예: ${1e-3, 1e-2, 1e-1, 1, 10}$ 등.
- 실무에선 검증 성능(다운스트림 분류/회귀, 클러스터 지표)로 선택.

---

## 관련 기법과 관계
- $λ→0$이면 표준 PCA에 수렴.
- 재구성 항 없이 $min_Z tr(Z^T L Z)$와 적절한 제약을 두면 Laplacian Eigenmaps와 유사.
- LLE/Isomap은 비선형 매니폴드 학습 계열, 본 방법은 선형 재구성+그래프 매끄러움의 절충.

---

## Original PCA와 비교
- 목적함수: PCA는 $min_{Z,U} ||X - ZU^T||_F^2$(또는 $max$ 분산)만 고려. Graph PCA는 여기에 $+ λ tr(Z^T L Z)$를 추가.
- 해법: PCA는 $X$의 SVD 한 번으로 닫힌형 해. Graph PCA는 $Z$-스텝 선형계 풀이 + $U$-스텝 Procrustes를 몇 차례 교대(여전히 간단함).
- 복잡도: $L$이 희소(sparse)이면 $Z$-스텝은 CG 등으로 빠르게 풀림. $λ→0$이면 PCA와 동일 비용.
- 해석: PCA는 전역 분산 보존, Graph PCA는 “이웃 보존(매끄러움)”을 가미한 분산–매끄러움 절충.


---

## Reference
- Jolliffe, I. T. “Principal Component Analysis.”
- Belkin, M., Niyogi, P. “Laplacian Eigenmaps.”

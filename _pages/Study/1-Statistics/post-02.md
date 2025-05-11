---
title: "[Paper Review] BART : Bayesian Additive Regression Tree"
tag : 
    - Statistics
    - Bayesian
    - BART
    - Bayesian Additive Regression Tree
    - Regression Tree
    - Classification

permalink: /Study/1-Statistics/BART
thumbnail: "/assets/img/bart/bart.jpg"
date: "2025-04-28"
---

# BART
>이 포스팅은 Chipman et al. (2010)의 논문 ["BART: Bayesian Additive Regression Trees"](https://arxiv.org/pdf/0806.3286)를 읽고 정리한 글입니다.

---
## Introduction
예측(Prediction) 문제는 결국 어떤 input $x$ 를 입력받아 output $Y$를 출력하는 함수 Unknown function $f$에 대한 추론으로 볼 수 있습니다. 

$$Y  = f(x)$$

우리가 원하는 함수 $f$를 보다 정확히 추론하기 위해 많은 방식이 연구되어왔습니다. 
Regression Model부터  Neural Network까지 모두 함수 $f$를 어떻게 추론(or 근사)할 것인가에 대한 다양한 방법론이라고 생각할 수 있습니다. 

이번에 제가 소개할 방법론은 여러 개의 **Regression Tree**의 합으로 $f(x) = \mathbb{E}[Y\mid x]$ 를 근사하는 방법입니다. 수식으로 간단히 표현해보면 다음과 같습니다.

$$f(x) \approx h(x)  = \sum_{j=1}^mg_j(x), \;\;g_j : \text{regression tree}$$

따라서 우리는 **a sum-of-trees (트리합)** 모델로 근사를 할 수 있습니다. 

$$Y = h(x)  + \epsilon, \;\;\epsilon \sim N(0,\sigma^2)$$

---
Sum-of-trees 모델은 근본적으로 다변수에 대한 가법(Additive)모델입니다. 
>“가법(Additive)“이란 여러 개의 개별 함수의 결과를 독립적으로 계산한 뒤, 이들의 값을 합산하여 결과를 얻는 것을 의미합니다. 

저차원 Smoother의 합으로 볼 수 있는 GAM(Generalized Additive Model, 일반화 가법 모형)과 비교했을 때, 이 다변수 가법 모형은 훨씬 더 자연스럽게 변수 간의 상호작용을 반영합니다. 
또한 단일 Tree 모델에 비해 Sum-of-trees 모델은 더 쉽게 addtive 효과를 다룰 수 있습니다.

다양한 Set-of-trees 모델들(*Ensemble 이라고 불리는*)은 많은 연구자들의 이목을 끌었습니다. 대표적으로 Boosting, Bagging, Random Forest 는 저마다의 방식을 통해 Tree들의 Linear Combination을 Fitting하는 방식으로 볼 수 있습니다.

Boosting은 이전 트리가 설명하지 못한 부분을 순차적으로 다음 트리가 학습하는 방식이고, bagging과 random forest는 무작위성을 도입해 다수의 독립된 트리를 만들고 예측을 평균내어 분산을 줄이는 방식입니다.

또 다른 접근 방식은 베이지안(Bayesian) 프레임워크를 이용하여 개별 Tree 모델들의 사후분포(Posterior distribution)를 구한 뒤, 이를 바탕으로 각 Tree의 예측값을 평균화(Bayesian Model Averaging)하는 방법입니다. 이때, 각 Tree가 데이터에 얼마나 잘 부합하는지에 따라 사후확률(Posterior probabilities)을 가중치로 사용하여 최종 예측값을 계산하게 됩니다.

---
#### 이제 본격적으로 BART(Bayesian Additive Regression Tree)에 대한 이야기를 해보겠습니다.

Chipman et al. (2010) 에서 저자는 Sum-of-trees 모델을 사용하여 $f(x)=\mathbb{E}[Y\mid x]$ 를 근사하는 방법론인 Bayesian Additive Regression Model을 제안했습니다. 

주요 아이디어는 다음과 같습니다.

##### $\rightarrow$ "각 Tree의 기여도를 제한하기 위해, 정규화(Regularization)된 Prior를 설정하여 Sum-of-trees 모델의 구조를 안정화시키기"

논문의 표현 방식을 빌려오면, 각각의 Regression Tree $g_j$들을 **weak Learners**로 만드는 것입니다. 각 $g_j$의 effects를 약화시킴으로써, BART는 각각의 약한 트리들이 함수 $f$의 small \& different portion을 설명하는 모델이 됩니다. 

Sum-of-trees 모델을 fitting하기 위해 BART는 [Bayesian backfitting MCMC](https://projecteuclid.org/journals/statistical-science/volume-15/issue-3/Bayesian-backfitting-with-comments-and-a-rejoinder-by-the-authors/10.1214/ss/1009212815.full)의 살짝 변형된 방식을 사용합니다. 간단히 말하면, **각 트리를 하나씩 순서대로 업데이트**하면서, **현재까지 남은 오차(residual)** 를 다음 트리가 학습하도록 반복하는 절차입니다. 즉, 전체 모델이 설명하지 못한 잔차를 순차적으로 각 트리들이 개선해나가도록 설계되어 있습니다.

이는 Gradient Boosting과 굉장히 유사해보입니다.
잔차를 학습한다는 측면에서 굉장히 닮아있습니다. 하지만 두 가지 차이가 존재합니다.
1. 개별 트리의 영향을 **Prior** 를 부여함으로써 약화시킨 것.
   > BART는 개별 트리의 약화를 위해 명시적인 손실 함수(loss function)를 최적화하는 대신, **사전 분포(prior distribution)** 를 사용하여 각 트리의 영향을 제한합니다.  
2. 고정된 개수의 Tree들에 대해 Bayesian Backfitting을 통해 적합한 것.
   > 또한 Boosting이 매 반복마다 새로운 트리를 추가하는 것과 달리, BART는 **고정된 개수의 트리(fixed number of trees)** 를 유지하면서 이들에 대해 반복적으로 베이지안 백피팅(Bayesian backfitting)을 수행합니다.

BART는 **강력한 사전 분포를 통해 복잡한(parameter rich) 모델을 제어하면서 학습하는 베이지안 비모수 방법(Bayesian nonparametric approach)** 으로 이해할 수 있습니다.


---

## The BART model
BART 모델은 두 부분으로 구성되는데, 이는 **a-sum-of-trees 모델** 과 모델의 파라미터에 부여되는 **Regularization 사전분포** 입니다.


### A-sum-of-trees model
   - Notation
      - $T$ : 내부 노드, 결정규칙, 말단 노드 집합으로 구성되는 Binary Tree
      - $M = \{\mu_1,\mu_2,\dots,\mu_b\}$ : $T$ 의 $b$ 개의 말단 노드에 주어진 파라미터 집합

여기서 결정 규칙은 $x=(x_1,x_2,\dots,x_p)$ 가 입력으로 주어질 때, Binary Splits으로 볼 수 있습니다. 에를 들어 $\{x \in A\}\; vs\;\{x \notin A\} $ 같이 주어질 수 있습니다. 각각의 입력 $x$ 는 연속적인 결정규칙에 의해 하나의 말단 노드에 배정받게 됩니다. 그리고 해당 말단 노드의 value$(\mu_i)$ 를 갖게 됩니다. 주어진 $T$와 $M$에 대해 입력 $x$를 적절한 말단 노드의 값 $\mu_i$에 할당하는 함수를 $g(x;T,M)$으로 정의하겠습니다.
이제 트리 모합 모델을 다음과 같이 표기할 수 있습니다.

$$Y = \sum_{j=1}^mg(x;T,M) + \epsilon,\;\;\epsilon\sim N(0,\sigma^2) $$

이제 주어진 $x$에 대한 출력의 기댓값 $\mathbb{E}[Y\mid x]$ 는 $g(x;T_j,M_j)$ 에 의해 $x$ 에 할당된 $\mu_{ij}$ 들의 합으로 표현될 수 있습니다. 여기서 중요한 것은 단변수의 상황에는 각각의 $\mu_{ij}$ 가 Main effect를 의미하며, 다변수의 상황에서는 변수간의 Interaction effect를 의미한다는 것입니다. 따라서 자연스레, sum-of-trees 모델은 main effect와 interaction effect를 모두 포착할 수 있게 됩니다.

쉽게 말해 노드에서의 결정 규칙이 일변수에서 다변수로 됨에 따라 상호작용을 고려할 수 있다는 것입니다. 예를 들어
단일 변수의 경우, 

$$ x_1  < 0.5  $$

의 결정 규칙을 가지고 있었다면, 다변수에서는  

$$x_1 < 0.5 \;\;\&\;\; x_2 > 0.3 $$

처럼 확장되어 상호작용을 고려할 수 있게 됩니다. 

---
### A regularization prior

**Prior**를 부여할 대상인 Sum-of-trees 모델의 파라미터는 다음과 같습니다.

1. $T_j$ : 트리 구조 파라미터
2. $M_j = \{\mu_{ij}\}$ : 말단 노드의 출력값 집합
3. $\sigma$ : 오차 분산 파라미터

Regularization prior의 설정은, 우리가 특정 종류의 사전 분포에만 관심을 제한함으로써 훨씬 간단해집니다.  
구체적으로는 다음과 같이 전체 사전분포를 **독립적인 구조로 factorize**합니다:

$$
p\left((T_1, M_1), \dots, (T_m, M_m), \sigma\right) = \left[\prod_{j=1}^m p(M_j \mid T_j) p(T_j)\right] p(\sigma)
$$

그리고 리프 노드의 값들도 서로 독립이라는 가정을 통해:

$$
p(M_j \mid T_j) = \prod_i p(\mu_{ij} \mid T_j)
$$

이러한 구성은 사전 분포를 단순하고 모듈화된 방식으로 구성할 수 있게 해주며, BART 모델의 계산적 효율성을 높여줍니다.

---

#### Tree 구조에 대한 prior

트리 구조 $T_j$에 대한 prior는 다음과 같이 설정합니다.  
깊이 $d$의 노드가 분기할 확률은 다음과 같은 함수로 정의됩니다:

$$
\text{Pr}(\text{split at depth } d) = \alpha (1 + d)^{-\beta}
$$

여기서 $\alpha \in (0, 1)$, $\beta \geq 0$는 하이퍼파라미터로, 일반적으로 $(\alpha, \beta) = (0.95, 2)$ 를 논문에서는 추천하고 있습니다.  
이는 트리가 깊어질수록 분기 확률이 감소하게 만들어 **얕은 트리를 유도**하고, 과적합을 방지하는 역할을 합니다.

또한, 어떤 노드가 분기(split)하기로 결정된 경우,  
사용할 **분할 변수(split variable)** 와 **분할 기준값(split value)** 은 다음과 같이 선택됩니다:

- 분할 변수는 현재 노드에서 사용 가능한 $p$개의 변수 중 균등 분포로 선택됩니다.
- 분할 기준값은 선택된 변수의 값 범위 내에서 균등 분포로 선택됩니다.


---

#### 리프 노드 값에 대한 prior

각 트리 $T_j$의 말단 노드 값 $\mu_{ij}$ 에 대해서는 다음과 같은 정규분포 prior를 가정합니다:

$$
\mu_{ij} \sim \mathcal{N}(0, \sigma_\mu^2)
$$

이 prior는 likelihood인 $Y_i \mid \mu_{ij} \sim \mathcal{N}(\mu_{ij}, \sigma^2)$ 와 conjugate 관계를 이룹니다.  
이로 인해 posterior 또한 정규분포로 닫힌 형태(closed-form)를 가지게 되어,  
Gibbs sampling 시 각 $\mu_{ij}$를 직접 샘플링(direct draw)할 수 있습니다:

$$
\mu_{ij} \mid \text{data} \sim \mathcal{N}(\tilde{\mu}, \tilde{\sigma}^2)
$$

뿐만 아니라, conjugate 구조를 활용하면 $\mu_{ij}$를 MCMC 과정에서 **marginalization** (적분으로 제거)할 수도 있습니다.  
즉, $\mu_{ij}$를 샘플링하지 않고 **사후 분포 내에서 주변화하여 수치적으로 통합**할 수 있으며,  
이는 계산 효율을 크게 높이는 방식 중 하나입니다.


---

#### 오차 분산 $\sigma^2$에 대한 prior

오차 항에 대한 분산 파라미터 $\sigma^2$는 다음과 같은 scaled-inverse-chi-squared 분포를 따릅니다:

$$
\sigma^2 \sim \text{Inv-}\chi^2(\nu, \lambda)
$$

여기서 $\nu$는 자유도(degrees of freedom), $\lambda$는 스케일(scale) 파라미터입니다.  
이 prior는 정규 likelihood와 conjugate 관계를 이루므로, posterior도 scaled-inverse-chi-squared 형태를 따르게 됩니다.

> 즉, $\sigma^2$ 역시 conjugate prior를 사용하여 **Gibbs sampling 단계에서 직접 샘플링이 가능**합니다.

---


#### 정리

BART에서는 prior를 단순히 무정보적(noninformative) 하게 설정하는 것이 아니라, 실제 데이터 분산을 기반으로 사전 정보를 반영하는 **data-informed prior approach**를 사용합니다. 이로 인해 데이터가 사전분포에 반영되어서는 안된다는 베이지안 원칙에는 어긋나지만, 데이터와 사전분포의 range가 충돌하지 않기 위해  compromise합니다. 
| 파라미터 | Prior 형태 | 목적 |
|:--|:--|:--|
| $T_j$ | $\text{Pr}(\text{split at depth } d) = \alpha (1 + d)^{-\beta}$ | 얕은 트리 유도 |
| $\mu_{ij}$ | $\mathcal{N}(0, \sigma_\mu^2)$ | 트리당 기여를 작게 제한 |
| $\sigma^2$ | $\text{Inv-}\chi^2(\nu, \lambda)$ | 오차 분산의 정규화 |

이러한 prior 설정은 BART가 개별 트리의 영향력을 제한하면서도, 여러 개의 약한 트리들이 모여 복잡한 함수 $f(x)$를 근사할 수 있도록 도와줍니다.

---

---

## Bayesian Backfitting MCMC

> 이제 어떻게 sum-of-tress 모델을 fitting 하는 지 알아보겠습니다다.

BART는 sum-of-trees 모델의 posterior를 추정하기 위해  
**Bayesian backfitting MCMC**를 사용합니다. 이는 CGM98 (Chipman, George, McCulloch, 1998)에서 제안한 backfitting 전략을 기반으로 하며,  
각각의 트리와 리프 노드 값을 **full conditional posterior**에서 샘플링하는 형태의 **Gibbs sampler**입니다.

### 목적

우리가 추정하고자 하는 전체 posterior는 다음과 같습니다:

$$
p\left( (T_1, M_1), \dots, (T_m, M_m), \sigma \mid Y \right)
$$

여기서,
- $T_j$: $j$번째 트리의 구조 (내부 노드, split 변수, split 값 등)
- $M_j = \{\mu_{ij}\}$: 말단 노드의 파라미터 집합
- $\sigma$: 오차 

---

### Backfitting의 핵심 아이디어

**전체 함수 $f(x)$를 한 번에 적합하지 않고**,  
각각의 트리 $g_j(x) = g(x; T_j, M_j)$가 담당하는 **잔차(residual)** 를 반복적으로 업데이트합니다.

즉, 나머지 모든 트리를 고정한 상태에서,  
하나의 트리 $g_j$만 선택하여 다음 조건부 posterior에서 샘플링합니다:

$$
p(T_j, M_j \mid R_j, \sigma)
$$

여기서 $R_j$는 현재 $j$번째 트리가 담당해야 할 잔차:

$$
R_j = Y - \sum_{k \ne j} g_k(x)
$$

---

## Gibbs Sampling

각 MCMC iteration에서 다음 과정을 반복합니다:

### (1) 각 트리에 대해 다음을 순차적으로 수행

#### (a) 트리 구조 $T_j$의 샘플링

- 고정된 $R_j$를 이용해 $T_j$의 posterior 분포로부터 샘플링.
- 가능한 트리 구조 후보들과 사전 확률 $p(T_j)$, likelihood를 곱해 MH(Metropolis-Hastings) 방식으로 수행.
- 트리 구조 변화는 보통 아래 네 가지 제안(move) 중 하나:
  - grow
  - prune
  - change
  - swap

#### (b) 리프 노드 파라미터 $M_j = \{\mu_{ij}\}$의 샘플링

- 각 $\mu_{ij}$는 해당 노드에 속한 잔차 값들과 conjugate 정규 prior를 통해
- 다음과 같은 조건부 posterior에서 직접 샘플링:

$$
\mu_{ij} \mid \text{data}, T_j, \sigma \sim \mathcal{N}(\tilde{\mu}_{ij}, \tilde{\sigma}^2)
$$

- $\tilde{\mu}_{ij}$, $\tilde{\sigma}^2$는 해당 노드에 속한 데이터의 평균, $\sigma^2$, prior variance로 구성됨.

---

### (2) 오차 분산 $\sigma^2$ 샘플링

- 모든 트리 구조 및 파라미터가 주어졌을 때, 잔차 $Y - \hat{Y}$를 기준으로 다음 분포에서 샘플링:

$$
\sigma^2 \mid Y, T_{1:m}, M_{1:m} \sim \text{Inv-}\chi^2(\nu^*, \lambda^*)
$$

- $\nu^* = \nu + n$, $\lambda^*$는 prior의 scale과 squared residual sum으로 계산됨.

---
### 정리

- 전체 함수를 한꺼번에 업데이트하지 않고, 각 트리 하나씩 번갈아 가며 잔차를 설명하는 방식.
- 이는 **Gradient Boosting처럼 residual을 순차적으로 줄이는 방식**과 구조적으로 유사하지만,  
  **확률 모델 기반의 MCMC를 통해 샘플링**한다는 점에서 베이지안적 특성이 강조됩니다.
- 파라미터가 conjugate 구조도록 설정하여 closed-form conditional posterior를 통해 효율적 샘플링이 가능합니다.

---

## Variable Selection in BART

BART는 명시적으로 변수 선택(variable selection)을 수행하진 않지만,  
**모델 내 각 변수의 중요도**를 추정할 수 있는 **변수 포함 비율 (variable inclusion proportions)** 개념을 제공합니다.

### 변수 선택 메커니즘

- BART의 트리 분할 과정에서, 각 split은 입력 변수 중 하나를 선택하여 수행됩니다.
- MCMC 샘플링 과정에서 **각 변수가 얼마나 자주 분할에 사용되었는지**를 기록할 수 있습니다.
- 이를 기반으로 **변수 포함 확률(variable inclusion probabilities)** 을 계산할 수 있습니다.

$$
\hat{p}_k = \frac{\text{Number of splits on variable } x_k}{\text{Total number of splits across all trees and iterations}}
$$

이 확률 $\hat{p}_k$는 변수 $x_k$가 모델에서 얼마나 중요하게 사용되고 있는지를 나타냅니다.

### 해석 방법

- $\hat{p}_k$ 값이 높을수록 변수 $x_k$는 모델의 다양한 트리에서 반복적으로 사용되며, 중요한 예측 변수로 간주됩니다.
- 반면 $\hat{p}_k$ 값이 매우 작거나 0에 가까우면, 해당 변수는 모델에서 거의 사용되지 않았으며 중요도가 낮다고 판단할 수 있습니다.

### 장점

- 기존의 결정 트리 기반 변수 중요도와 달리, **posterior 기반 확률적 추론**이므로  
  **불확실성까지 반영된 변수 중요도 해석**이 가능합니다.
- $\hat{p}_k$ 값의 분포(예: credible interval)를 추정하면, 변수 중요도에 대한 **통계적 신뢰**까지 제시할 수 있습니다.

> 이와 같은 변수 중요도 분석은 BART가 예측뿐만 아니라 **설명력 있는 모델링에도 유용**하다는 점을 보여줍니다.

---
## Conclusion
이번 포스팅에서는 BART(Bayesian Additive Regression Tree)의 핵심 개념, 수학적 구성 요소, 정규화된 prior 구조, Bayesian Backfitting MCMC, 그리고 변수 선택 메커니즘까지 상세히 살펴보았습니다.

BART는 복잡한 함수 $f(x)$를 표현하는 데 있어 **설명 가능성과 예측 정확도**, 그리고 **불확실성 정량화**까지 모두 갖춘 강력한 Bayesian 기법입니다.

다음 포스팅에서는 원 논문에서 제안한 BART의 실험 결과와 다른 머신러닝 모델들과의 성능 비교, 그리고 실제 데이터셋에 적용한 사례를 리뷰할 예정입니다. 긴 글 읽어주셔서 감사합니다 :)
## Reference
- [Bayesian Backfitting](https://www.jstor.org/stable/2676659?seq=1)
- [Bayesian Additive Regression Trees: A Review and Look Forward](https://www.annualreviews.org/content/journals/10.1146/annurev-statistics-031219-041110)
- [BART package in r](https://cran.r-project.org/web/packages/BART/index.html)
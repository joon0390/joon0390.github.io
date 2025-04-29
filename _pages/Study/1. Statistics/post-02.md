---
title: "BART : Bayesian Additive Regression Tree"
tag : 
    - Statistics
    - BART
    - Bayesian Additive Regression Tree
    - Regression Tree
    - Bayesian

thumbnail: "/assets/img/thumbnail/bart.jpg"
date: "2025-04-28"
---

# BART
이 포스팅은 Chipman et al. (2010)의 논문 ["BART: Bayesian Additive Regression Trees"](https://arxiv.org/pdf/0806.3286)를 읽고 정리한 글입니다.

---
## Introduction
예측(Prediction) 문제는 결국 어떤 input $x$ 를 입력받아 output $Y$를 출력하는 함수 Unknown function $f$에 대한 추론으로 볼 수 있습니다. 

$$Y  = f(x)$$

우리가 원하는 함수 $f$를 보다 정확히 추론하기 위해 많은 방식이 연구되어왔습니다. 
Regression Model부터  Neural Network까지 모두 함수 $f$를 어떻게 추론(or 근사)할 것인가에 대한 다양한 방법론이라고 생각할 수 있습니다. 

이번에 제가 소개할 방법론은 여러 개의 **Regression Tree**의 합으로 $f(x) = \mathbb{E}[Y\mid x]$ 를 근사하는 방법입니다. 수식으로 간단히 표현해보면 다음과 같습니다.

$$f(x) \approx h(x)  = \sum_{j=1}^mg_j(x), \;\;g_j : \text{regressoin tree}$$

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
>- 각각의 Tree 들의 **영향력을 작게 유지**하기 위해 **Regularize하는 Prior**를 부여하여 Sum-of-trees 모델을 다듬자!

논문의 표현 방식을 빌려오면, 각각의 Regression Tree $g_j$들을 **weak Learners**로 만드는 것입니다. 각 $g_j$의 effects를 약화시킴으로써, BART는 각각의 약한 트리들이 함수 $f$의 small \& different portion을 설명하는 모델이 됩니다. 

Sum-of-trees 모델을 fitting하기 위해 BART는 [Bayesian backfitting MCMC](https://projecteuclid.org/journals/statistical-science/volume-15/issue-3/Bayesian-backfitting-with-comments-and-a-rejoinder-by-the-authors/10.1214/ss/1009212815.full)의 살짝 변형된 방식을 사용합니다. 간단히 말하면, **각 트리를 하나씩 순서대로 업데이트**하면서, **현재까지 남은 오차(residual)** 를 다음 트리가 학습하도록 반복하는 절차입니다. 즉, 전체 모델이 설명하지 못한 잔차를 순차적으로 각 트리들이 개선해나가도록 설계되어 있습니다.

이는 Gradient Boosting과 굉장히 유사해보입니다.
잔차를 학습한다는 측면에서 굉장히 닮아있습니다. 하지만 두 가지 차이가 존재합니다.
1. 개별 트리의 영향을 **Prior** 를 부여함으로써 약화시킨 것.
   >- BART는 개별 트리의 약화를 위해 명시적인 손실 함수(loss function)를 최적화하는 대신, **사전 분포(prior distribution)** 를 사용하여 각 트리의 영향을 제한합니다.  
2. 고정된 개수의 Tree들에 대해 Bayesian Backfitting을 통해 적합한 것.
   >- 또한 Boosting이 매 반복마다 새로운 트리를 추가하는 것과 달리, BART는 **고정된 개수의 트리(fixed number of trees)** 를 유지하면서 이들에 대해 반복적으로 베이지안 백피팅(Bayesian backfitting)을 수행합니다.

BART는 **강력한 사전 분포를 통해 복잡한(parameter rich) 모델을 제어하면서 학습하는 베이지안 비모수 방법(Bayesian nonparametric approach)** 으로 이해할 수 있습니다.


---

## The BART model
언급했었던 대로 BART 모델은 두 부분으로 구성되는데, 이는 **a-sum-of-trees 모델** 과 모델의 파라미터에 부여되는 **Regularization 사전분포** 입니다.


### A-sum-of-trees model
   - Notation
      - $T$ : 내부 노드, 결정규칙, 말단 노드 집합으로 구성되는 Binary Tree
      - $M = \{\mu_1,\mu_2,\dots,\mu_b\}$ : $T$ 의 $b$ 개의 말단 노드에 주어진 파라미터 집합

여기서 결정 규칙은 $x=(x_1,x_2,\dots,x_p)$ 가 입력으로 주어질 때, Binary Splits으로 볼 수 있습니다. 에를 들어 $\{x \in A\}\; vs\;\{x \notin A\} $ 같이 주어질 수 있습니다. 각각의 입력 $x$ 는 연속적인 결정규칙에 의해 하나의 말단 노드에 배정받게 됩니다. 그리고 해당 말단 노드의 value$(\mu_i)$ 를 갖게 됩니다. 주어진 $T$ 와 $M$ 에 대해서 $x$ 를 $\mu_i \in M$ 으로 assign하는 함수를 이제 $g(x;T,M)$ 으로 표기하겠습니다.   
이제 트리 모합 모델을 다음과 같이 표기할 수 있습니다.

$$Y = \sum_{j=1}^mg(x;T,M) + \epsilon,\;\;\epsilon\sim N(0,\sigma^2) $$

이제 주어진 $x$에 대한 출력의 기댓값 $\mathbb{E}[Y\mid x]$ 는 $g(x;T_j,M_j)$ 에 의해 $x$ 에 할당된 $\mu_{ij}$ 들의 합으로 표현될 수 있습니다. 여기서 중요한 것은 단변수의 상황에는 각각의 $\mu_{ij}$ 가 Main effect를 의미하며, 다변수의 상황에서는 변수간의 Interaction effect를 의미한다는 것입니다. 따라서 자연스레, sum-of-trees 모델은 main effect와 interaction effect를 모두 포착할 수 있게 됩니다.

쉽게 말해 노드에서의 결정 규칙이 일변수에서 다변수로 됨에 따라 상호작용을 고려할 수 있다는 것입니다. 예를 들어
단일 변수의 경우, 

>$$ x_1  < 0.5  $$

의 결정 규칙을 가지고 있었다면, 다변수에서는  

>$$x_1 < 0.5 \;\;\&\;\; x_2 > 0.3 $$

처럼 확장되어 상호작용을 고려할 수 있게 됩니다. 

---
### A regularization prior


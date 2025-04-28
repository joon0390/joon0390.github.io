---
title: "BART : Bayesian Additive Regression Tree"
tag : 
    - Statistics
    - BART
    - Bayesian Additive Regression Tree
    - Regression Tree
    - Bayesian
date: "2025-04-29"
---

이 포스팅은 Chipman et al. (2010)의 논문 ["BART: Bayesian Additive Regression Trees"](https://arxiv.org/pdf/0806.3286)를 읽고 정리한 글입니다.

---
## Introduction
예측(Prediction) 문제는 결국 어떤 input $x$ 를 입력받아 output $Y$를 출력하는 함수 Unknown function $f$에 대한 추론으로 볼 수 있습니다. 

$$Y  = f(x)$$

우리가 원하는 함수 $f$를 보다 정확히 추론하기 위해 많은 방식이 연구되어왔습니다. 
Regression Model부터  Neural Network까지 모두 함수 $f$를 어떻게 추론(or 근사)할 것인가에 대한 다양한 방법론이라고 생각할 수 있습니다. 

이번에 제가 소개할 방법론은 여러 개의 **Regression Tree**의 합으로 $f(x)$를 근사하는 방법입니다. 수식으로 간단히 표현해보면 다음과 같습니다.
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
## BART
##### 이제 본격적으로 BART(Bayesian Additive Regression Tree)에 대한 이야기를 해보겠습니다.


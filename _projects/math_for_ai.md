---
title: "동물 이미지 분류 대회 (MTH3501)"
excerpt: "ConvNeXt, ResNet, ViT 등 다양한 모델을 실험하며 이미지 분류 정확도 향상을 달성한 실전 분류 과제"
collection: projects
layout: posts
date: 2024-06-07
tags:
  - Image Classification
  - Computer Vision
  - Deep Learning
  - ConvNeXt
  - ResNet
  - Vision Transformer
  - MTH3501
---

## 🧠 프로젝트 개요

본 프로젝트는 **「인공지능을 위한 수학 1 (MTH3501)」** 강좌에서 진행된 **이미지 분류 대회 기반 과제**입니다.  
총 11개의 동물 클래스를 포함한 96x64 RGB 이미지들에 대해 학습을 수행하고,  
**Test 데이터셋 600장에 대한 예측 결과 파일(csv 또는 txt)**을 제출하여 분류 정확도를 평가받는 형식입니다.

> 📁 학습용 데이터셋: 총 2,750장 (11 classes × 각 250장)  
> 📁 테스트 데이터셋: 총 600장 (레이블 비공개)  
> 🐾 클래스 목록:  
> bear(0), bird(1), fish(2), crocodile(3), horse(4), lizard(5), monkey(6), panthera(7), penguin(8), turtle(9), whale(10)

---

## 🔧 사용한 모델 및 전략

다양한 모델 구조를 활용하여 성능을 비교하며 실험을 진행했습니다.

- ✅ **ConvNeXt**  
  최신 비전 트랜스포머 구조를 CNN 스타일로 재해석한 모델로, 성능 최상

- ✅ **ResNet (ResNet18, ResNet50)**  
  안정적이고 빠른 수렴을 보여 baseline 성능 확보용으로 사용

- ✅ **Vision Transformer (ViT)**  
  Patch 분할 및 Self-Attention 기반 구조, 적절한 augmentation이 효과적

- ✅ **Data Augmentation**  
  RandomHorizontalFlip, ColorJitter, ResizeCrop 등의 다양한 변형 기법 적용

- ✅ **하이퍼파라미터 튜닝**  
  - Optimizer: AdamW, SGD  
  - Learning Rate: CosineAnnealing  
  - Epoch: 최대 80, Early Stopping 적용  
  - Batch Size: 32~128

---

## 🏆 리더보드 성적

> 다음은 제출된 결과 파일을 기반으로 주어진 테스트셋에서의 리더보드 기록입니다:

| 제출자 학번 | 제출 횟수 | Accuracy (%) |
|-------------|------------|--------------|
| **12191886** | 1 | **88.33** |
| **12191886** | 3 | **88.17** |
| **12191886** | 2 | **88.33** |
| ********    | 2 | 84.33 |
| ********    | 3 | 79.33 |

🎉 12191886 으로 상위권을 모두 휩쓸며 1~3위를 기록하였습니다.

---

## 📄 참고 자료

- [ConvNeXt: A ConvNet for the 2020s (CVPR 2022)](https://arxiv.org/abs/2201.03545)  
- [Vision Transformer (ViT)](https://arxiv.org/abs/2010.11929)  
- [ResNet: Deep Residual Learning](https://arxiv.org/abs/1512.03385)  
- [PyTorch Image Models (timm)](https://github.com/huggingface/pytorch-image-models)

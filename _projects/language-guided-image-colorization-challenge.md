---
title: "언어정보 기반 흑백 이미지 색채화"
excerpt: "자연어 지시문을 조건으로 흑백 이미지를 색채화하고, L-CAD 기반 생성 결과를 CATANet 초해상화와 후보정으로 개선한 2025 인하 인공지능 챌린지 최우수상 프로젝트."
date: 2025-07-01
collection: projects
layout: single
order: 10
period: "2025.07"
header:
  teaser: /assets/img/projects/language-guided-image-colorization-challenge/result.png
teaser_alt: "흑백 이미지 색채화와 후보정 결과 비교"
tags:
  - Image Colorization
  - Diffusion Model
  - Vision-Language
  - Super Resolution
  - Competition
---

## 프로젝트 요약

- 개요: 자연어 지시문을 기반으로 흑백 이미지를 의미에 맞게 채색하는 vision-language 생성 AI 프로젝트
- 기간: 2025.07
- 데이터: 512x512 흑백 이미지, 자연어 색채 지시문, 생성 결과 및 평가용 컬러 이미지
- 기술 스택: L-CAD, Stable Diffusion fine-tuning, CATANet, CLIP/HSV Score, image post-processing
- 성과(성능): Public score 0.686에서 0.703으로 개선, 2025 인하 인공지능 챌린지 최우수상

## 문제 정의

흑백 이미지 색채화는 단순히 회색조 위에 그럴듯한 색을 입히는 문제가 아니었습니다. 같은 흑백 이미지라도 자연어 지시문이 어떤 객체의 색을 명시하는지, 전체 분위기를 요구하는지에 따라 결과가 달라져야 했습니다. 동시에 원본 이미지의 구조와 윤곽은 유지해야 했기 때문에, 생성 모델의 자유도를 그대로 두면 색은 풍부하지만 객체 경계가 흐려지거나 지시문과 다른 색이 들어가는 문제가 생겼습니다.

대회에서는 텍스트와 이미지의 의미 일치도뿐 아니라 색상 품질도 함께 중요했습니다. 실제 실험 과정에서도 CLIP Score는 높지만 HSV Score가 낮은 결과가 자주 나왔습니다. 즉, 텍스트와는 맞아 보이지만 색감이 탁하거나 최종 이미지 품질이 평가 지표에 충분히 맞지 않는 경우가 있었고, 모델 출력 이후의 복원과 후보정까지 하나의 파이프라인으로 다뤄야 했습니다.

## 데이터와 EDA

입력은 512x512 흑백 이미지와 자연어 지시문이었습니다. 지시문은 단순한 분위기 수준의 문장부터 특정 객체의 색을 직접 지정하는 문장까지 다양했습니다. 그래서 EDA에서는 이미지 자체보다 이미지와 텍스트가 서로 어떤 정보를 나눠 갖는지를 먼저 확인했습니다.

특히 다음 항목을 중점적으로 봤습니다.

- 지시문이 전체 색감만 요구하는지, 특정 객체의 색을 요구하는지
- 흑백 이미지에서 객체 경계와 구조가 충분히 보존되어 있는지
- 생성 결과에서 색 번짐, 채도 부족, 배경 오염이 반복적으로 나타나는지
- 텍스트 일치도는 높지만 색상 유사도가 낮은 케이스가 어떤 패턴을 갖는지
- 모델 단계에서 해결할 문제와 후보정으로 개선할 문제를 분리할 수 있는지

## 접근 방법

<figure class="project-figure project-figure--medium">
  <img src="/assets/img/projects/language-guided-image-colorization-challenge/pipeline.png" alt="L-CAD, CATANet, 후보정으로 이어지는 언어정보 기반 흑백 이미지 색채화 파이프라인">
  <figcaption>L-CAD 색채화, CATANet 초해상화, 후보정으로 구성한 전체 파이프라인</figcaption>
</figure>

### Step 1. 색채화: L-CAD

색채화 단계에서는 Stable Diffusion을 색채화 태스크에 파인튜닝한 L-CAD(Language-based Colorization with Any-level Descriptions)를 사용했습니다. 핵심은 흑백 이미지의 구조를 유지하면서 텍스트 조건에 맞는 색상만 입히도록 모델을 제한하는 것이었습니다.

<figure class="project-figure project-figure--medium">
  <img src="/assets/img/projects/language-guided-image-colorization-challenge/l-cad.png" alt="L-CAD 모델 아키텍처">
  <figcaption>L-CAD 모델 아키텍처</figcaption>
</figure>

L-CAD는 흑백 이미지의 휘도 정보, 즉 L 채널을 Luminance-encoder로 인코딩해 UNet에 직접 주입합니다. 덕분에 모델이 장면의 윤곽과 배치를 새로 상상하기보다 기존 이미지의 구조를 따라가면서 색상 정보를 생성하도록 유도할 수 있었습니다.

또 하나의 중요한 특징은 Any-level Description이었습니다. "A colorful image"처럼 정보가 적은 scarce-level description부터 "A girl is wearing a purple t-shirt"처럼 특정 객체와 색을 함께 주는 partial-level description까지 다양한 수준의 텍스트 조건을 처리하도록 구성했습니다. 이 방식은 지시문이 항상 상세하지 않은 대회 환경에서 유용했습니다.

### Step 2. 초해상화: CATANet

L-CAD의 생성 결과는 256x256 해상도였기 때문에, 최종 제출 형식에 맞추기 위해 512x512로 복원해야 했습니다. 단순 보간으로 키우면 색은 유지되더라도 경계와 질감이 흐려질 수 있어 CATANet(Cross-Attention Token Aggregation Network) 기반 초해상화를 적용했습니다.

CATANet에서는 Token Aggregation Block을 통해 객체 경계와 영역 간 상호작용을 함께 학습합니다. 이 과정은 단순히 픽셀 수를 늘리는 작업이 아니라, 저해상도 생성 결과의 전역 색조를 유지하면서 세부 질감과 경계를 자연스럽게 복원하는 단계로 작동했습니다.

### Step 3. 후보정

마지막 단계에서는 모델 출력이 평가 지표와 시각적 품질 양쪽에서 안정적으로 보이도록 후보정을 적용했습니다. 특히 CLIP Score는 높지만 HSV Score가 낮은 결과를 줄이는 데 집중했습니다.

적용한 후보정은 감마 보정, CLAHE 기반 대비 향상, 채도 강화, 언샤프 마스크, 양방향 필터 등이었습니다. 생성 모델의 결과를 무리하게 바꾸기보다, 색감이 흐려지거나 경계가 약해지는 지점을 보정하는 방향으로 조정했습니다.

<figure class="project-figure project-figure--medium">
  <img src="/assets/img/projects/language-guided-image-colorization-challenge/result.png" alt="흑백 이미지에서 L-CAD 색채화, CATANet 초해상화, 후보정으로 개선되는 결과 예시">
  <figcaption>색채화, 초해상화, 후보정을 거친 결과 변화</figcaption>
</figure>

## 성과(성능)

- 후보정 단계를 거치며 Public score가 0.686에서 0.703으로 향상되었습니다.
- L-CAD의 구조 보존 능력, CATANet의 복원 품질, 세밀한 후보정 전략을 결합해 최우수상을 수상했습니다.
- 제한된 컴퓨팅 자원 안에서 사전 학습 모델을 활용하고, 모델링 이후의 후처리까지 평가 지표에 맞춰 조정하는 파이프라인을 구성했습니다.

## 느낀점

이 프로젝트에서 가장 크게 배운 점은 생성 모델 프로젝트에서도 마지막 품질은 모델 하나로만 결정되지 않는다는 것이었습니다. 처음에는 더 좋은 색을 만들기 위해 모델 구조나 파인튜닝 방식에만 집중했지만, 실제 점수는 해상도 복원과 후보정 단계에서 크게 달라졌습니다.

특히 텍스트 조건을 잘 반영한 결과와 평가 지표에서 높은 점수를 받는 결과가 항상 같지는 않았습니다. 사람이 보기에는 자연스러운 이미지라도 HSV 기반 평가에서는 불리할 수 있었고, 반대로 색상은 맞지만 전체 이미지가 탁해 보이는 경우도 있었습니다. 이 차이를 줄이기 위해 결과를 직접 비교하고, 어떤 보정이 어떤 지표에 영향을 주는지 반복적으로 확인한 과정이 의미 있었습니다.

제한된 자원 안에서 완전히 새로운 모델을 학습하기보다, 이미 강한 사전 학습 모델을 문제에 맞게 조정하고 후단 파이프라인을 정교하게 만드는 접근이 현실적인 선택이라는 것도 체감했습니다. 포트폴리오에 남길 만한 성과도 중요했지만, 실제로는 생성 AI 결과물을 평가 기준에 맞게 끝까지 다듬는 경험이 더 크게 남은 프로젝트였습니다.

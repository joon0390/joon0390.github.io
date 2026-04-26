---
title: "대조학습 방법을 이용한 주행 패턴 분석 기법 연구"
excerpt: "스마트폰 센서 기반 주행 시계열에 시간 일관성 대조학습을 적용해 변화점을 탐지하고, 소량의 레이블만으로도 패턴 분류를 수행한 연구."
date: 2026-04-18
collection: projects
layout: single
order: 5
classes:
  - wide
tags:
  - Contrastive Learning
  - Self-Supervised Learning
  - Driving Pattern Analysis
  - STFT
  - Change Point Detection
---

<style>
  #main {
    max-width: min(96vw, 1720px);
  }

  .layout--single .page {
    width: 100% !important;
    padding-inline-end: 0 !important;
  }

  .layout--single .page__inner-wrap {
    width: 100%;
  }

  #contrastive-project-layout {
    display: block;
  }

  #contrastive-project-main {
    min-width: 0;
    max-width: 56rem;
  }

  #contrastive-project-viewer {
    margin-top: 2rem;
  }

  .contrastive-copy {
    word-break: keep-all;
    overflow-wrap: break-word;
  }

  .contrastive-panel {
    border: 1px solid rgba(20, 54, 82, 0.12);
    border-radius: 1.25rem;
    background:
      radial-gradient(circle at top, rgba(255, 255, 255, 0.78), transparent 42%),
      linear-gradient(180deg, #f8fbfc 0%, #edf4f6 100%);
    box-shadow: 0 18px 40px rgba(17, 57, 91, 0.08);
    padding: 1.1rem;
  }

  .contrastive-eyebrow {
    margin: 0 0 0.35rem;
    color: #73899b;
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.14em;
    text-transform: uppercase;
  }

  .contrastive-title {
    margin: 0 0 0.35rem;
    color: #103754;
    font-size: 1.05rem;
    font-weight: 700;
    line-height: 1.4;
  }

  .contrastive-subtitle {
    margin: 0 0 0.9rem;
    color: #627888;
    font-size: 0.92rem;
    line-height: 1.65;
  }

  .contrastive-pdf-frame {
    overflow: hidden;
    border: 1px solid rgba(18, 57, 91, 0.1);
    border-radius: 1rem;
    background: #fff;
    height: 76vh;
    min-height: 52rem;
  }

  .contrastive-pdf-frame iframe {
    display: block;
    width: 100%;
    height: 100%;
    border: 0;
  }

  .contrastive-note {
    margin: 0.85rem 0 0;
    color: #627888;
    font-size: 0.9rem;
    line-height: 1.6;
  }

  .contrastive-flow {
    margin: 1.7rem 0 2.1rem;
  }

  .contrastive-flow__head {
    margin-bottom: 0.9rem;
  }

  .contrastive-flow__head h3 {
    margin: 0 0 0.25rem;
    color: #103754;
    font-size: 1.1rem;
  }

  .contrastive-flow__head p {
    margin: 0;
    color: #627888;
    font-size: 0.95rem;
  }

  .contrastive-flow-grid {
    display: grid;
    gap: 0.9rem;
  }

  .contrastive-flow-card {
    border: 1px solid rgba(18, 57, 91, 0.1);
    border-radius: 1rem;
    background: #fff;
    padding: 1rem;
  }

  .contrastive-flow-card h4 {
    margin: 0 0 0.5rem;
    color: #103754;
    font-size: 1rem;
  }

  .contrastive-flow-card p {
    margin: 0;
    color: #4f6678;
    font-size: 0.95rem;
    line-height: 1.65;
  }

  .contrastive-flow-arrow {
    display: none;
  }

  .contrastive-summary-grid {
    display: grid;
    gap: 0.9rem;
    margin: 1rem 0 1.8rem;
  }

  .contrastive-summary-box {
    border-radius: 1rem;
    padding: 1rem;
  }

  .contrastive-summary-box h4 {
    margin: 0 0 0.35rem;
    color: #103754;
    font-size: 1rem;
  }

  .contrastive-summary-box p {
    margin: 0;
    color: #4f6678;
    font-size: 0.95rem;
    line-height: 1.65;
  }

  .contrastive-summary-box--mint {
    background: #e4f6ef;
  }

  .contrastive-summary-box--blue {
    background: #e8f2fb;
  }

  @media (min-width: 980px) {
    #contrastive-project-layout {
      display: grid;
      grid-template-columns: minmax(0, 1.55fr) clamp(28rem, 30vw, 34rem);
      gap: clamp(1.75rem, 2.4vw, 3rem);
      align-items: start;
    }

    #contrastive-project-viewer {
      margin-top: 0;
      position: sticky;
      top: 1rem;
      width: 100%;
      max-width: 34rem;
      justify-self: end;
    }

    .contrastive-flow-grid {
      grid-template-columns: minmax(0, 1fr) 2rem minmax(0, 1fr) 2rem minmax(0, 1fr);
      align-items: stretch;
    }

    .contrastive-flow-arrow {
      display: flex;
      align-items: center;
      justify-content: center;
      color: #18839a;
      font-size: 1.6rem;
      font-weight: 700;
    }

    .contrastive-summary-grid {
      grid-template-columns: repeat(2, minmax(0, 1fr));
    }
  }

  @media (max-width: 979px) {
    .contrastive-pdf-frame {
      height: 72vh;
      min-height: 38rem;
    }
  }

  @media (max-width: 640px) {
    .contrastive-panel {
      padding: 0.9rem;
    }

    .contrastive-pdf-frame {
      height: 68vh;
      min-height: 32rem;
    }
  }
</style>

<div id="contrastive-project-layout">
<div id="contrastive-project-main" class="contrastive-copy" markdown="1">

## 프로젝트 개요

스마트폰 센서로 수집한 주행 데이터를 바탕으로, 레이블이 거의 없는 환경에서도 주행 패턴의 변화점을 탐지하고 소량의 레이블 데이터만으로 분류까지 확장할 수 있도록 설계한 연구입니다. 핵심은 지도학습에 의존하지 않고, 시간 축에서의 일관성을 기준으로 대조학습을 수행해 주행 데이터의 표현을 먼저 학습한 뒤 이를 변화점 탐지와 패턴 분류에 재사용하는 것입니다.

<section class="contrastive-flow">
  <div class="contrastive-flow__head">
    <h3>학습 파이프라인</h3>
    <p>레이블이 없는 데이터로 표현을 먼저 학습하고, 이후 소량의 레이블로 분류를 확장하는 2단계 구조입니다.</p>
  </div>
  <div class="contrastive-flow-grid">
    <div class="contrastive-flow-card">
      <h4>전처리</h4>
      <p>스플라인 보간, 이벤트 중심 슬라이싱, STFT 변환을 적용해 주행 시계열을 시간-주파수 표현으로 정리합니다.</p>
    </div>
    <div class="contrastive-flow-arrow" aria-hidden="true">→</div>
    <div class="contrastive-flow-card">
      <h4>대조학습</h4>
      <p>시간적으로 가까운 구간은 positive pair, 먼 구간은 negative pair로 두고 InfoNCE loss로 특징 표현을 학습합니다.</p>
    </div>
    <div class="contrastive-flow-arrow" aria-hidden="true">→</div>
    <div class="contrastive-flow-card">
      <h4>분류 확장</h4>
      <p>사전학습된 특징 추출기를 고정한 뒤, 32차원 특징 벡터에 SVC를 붙여 소량 레이블 조건에서 분류를 수행합니다.</p>
    </div>
  </div>
</section>

## 핵심 내용

- 전체 프레임워크는 `2단계`로 구성됩니다. 먼저 레이블 없는 데이터로 변화점 탐지를 위한 표현을 학습하고, 이후 소량의 레이블 데이터로 분류기를 학습합니다.
- 대조학습 단계에서는 시간적으로 가까운 구간을 `positive pair`, 먼 구간을 `negative pair`로 두는 `time consistency` 전략을 사용했습니다.
- 특징 벡터 간 유사도는 `cosine similarity`로 계산하고, 손실 함수는 `InfoNCE loss`를 사용해 유사한 구간은 가깝게, 다른 구간은 멀어지도록 학습했습니다.
- 변화점 탐지는 학습된 특징 공간에서 현재 구간과 과거 구간 사이의 코사인 유사도가 임계값 이하로 떨어질 때 발생한 것으로 판단했습니다.
- 분류 단계에서는 사전학습된 딥러닝 모델을 특징 추출기로 고정하고, 추출된 `32`차원 특징 벡터 위에 `SVC`를 학습하는 하이브리드 구조를 사용했습니다.

<div class="contrastive-summary-grid">
  <div class="contrastive-summary-box contrastive-summary-box--mint">
    <h4>표현 학습 관점</h4>
    <p>레이블 대신 시간 일관성을 학습 신호로 사용해, 데이터 자체에 들어 있는 주행 패턴 변화를 먼저 익히도록 설계했습니다.</p>
  </div>
  <div class="contrastive-summary-box contrastive-summary-box--blue">
    <h4>실전 적용 관점</h4>
    <p>경량 백본과 소량 레이블 조건을 함께 검증해, 스마트폰 기반 운전자 보조 시스템으로 확장 가능한 구성을 지향했습니다.</p>
  </div>
</div>

## 데이터 및 실험 설계

- 데이터셋은 `Ferreira et al. (2017)`의 공개 스마트폰 기반 운전 데이터셋을 사용했습니다.
- 사용 센서는 가속도, 선형가속도, 자기장, 각속도이며, 각 운전자는 `13분`씩 `4회` 주행하며 특정 이벤트를 수행했습니다.
- 전처리는 `스플라인 보간`, `이벤트 구간 중심 슬라이싱`, `STFT(Short Time Fourier Transform)` 순서로 진행해 시간-주파수 특성을 함께 반영했습니다.
- 대조학습 단계에서는 레이블을 제거한 전체 `913`개 세그먼트를 사용했고, 분류 단계에서는 `58`개 세그먼트로 학습, `61`개 세그먼트로 검증했습니다.
- 일반화 성능을 보기 위해 학습 차량과 검증 차량을 다르게 분리했습니다. 즉, 동일 차량 분포에만 맞춘 모델이 아니라 차량이 달라져도 유지되는 표현 학습 성능을 보려는 설계입니다.
- 경량 백본으로는 `SqueezeNet`, `ShuffleNet`, `RegNet`, `MobileNet`, `EfficientNet`, `MnasNet`의 `6개` 모델을 비교했습니다.

## 주요 결과

- 변화점 탐지에서는 대부분의 모델이 코사인 유사도 임계값 `0.4`에서 가장 좋은 정밀도를 보였습니다.
- 변화점 탐지 최고 성능은 `RegNet`이 기록했으며, 정밀도는 `0.92`였습니다.
- 분류에서는 `SqueezeNet`이 정밀도 `1.0`, 재현율 `1.0`으로 가장 높은 성능을 보였습니다.
- `RegNet`도 정밀도 `0.956`, 재현율 `0.951`로 높은 성능을 보였고, `ShuffleNet` 역시 정밀도 `0.886`, 재현율 `0.869`를 기록했습니다.
- 반면 `MobileNet`, `EfficientNet`, `MnasNet`은 상대적으로 성능이 낮았고, 특히 `MnasNet`은 본 태스크와 네트워크 최적화 방향의 차이로 인해 학습이 충분히 이뤄지지 않은 것으로 해석되었습니다.
- 클래스별 레이블 수가 `30개 내외`로 매우 적은 조건과, 학습/검증 차량이 서로 다른 조건에서도 유의미한 분류 성능을 보였다는 점이 중요한 결과입니다.

## 사용 기술

- Contrastive Learning
- Self-Supervised Learning
- Time Consistency Learning
- STFT
- Change Point Detection
- SVC
- Lightweight CNN Backbones

## 프로젝트 의의

이 연구의 강점은 라벨이 부족한 주행 데이터 문제를 단순 분류가 아니라 `표현 학습 + 변화점 탐지 + 소량 레이블 분류`의 구조로 다시 설계했다는 점입니다. 특히 변화점 탐지를 먼저 학습한 뒤 같은 표현을 분류에도 재활용함으로써, 실제 데이터 수집 환경에서 더 현실적인 접근을 제시했습니다.

또한 스마트폰 탑재를 고려한 경량 모델들을 비교했다는 점도 의미가 큽니다. 정확도만 보는 것이 아니라 메모리와 연산량을 함께 고려해, 이후 모바일 기반 운전자 보조 시스템이나 실시간 주행 분석 시스템으로 확장할 수 있는 기반을 만들었다고 볼 수 있습니다.

## 느낀점

이 프로젝트에서 특히 인상적이었던 부분은, 같은 대조학습 프레임워크를 유지하더라도 어떤 백본을 쓰느냐에 따라 변화점 탐지와 분류 성능이 꽤 크게 달라졌다는 점이었습니다. `SqueezeNet`, `ShuffleNet`, `RegNet`, `MobileNet`, `EfficientNet`, `MnasNet`처럼 여러 경량 모델을 직접 바꿔가며 실험해 보면서, 단순히 가벼운 모델이면 모두 비슷하게 동작하는 것이 아니라 각 아키텍처가 특징을 추출하는 방식과 태스크 적합도가 결과에 큰 영향을 준다는 점을 체감했습니다.

또한 이 과정을 통해 모델 하나의 성능만 보는 것이 아니라, 백본 선택 자체가 연구의 중요한 변수라는 점을 더 분명하게 배웠습니다. 어떤 모델은 매우 적은 레이블 환경에서도 강한 일반화 성능을 보였고, 어떤 모델은 같은 설정에서도 학습이 잘 되지 않았기 때문에, 실제 적용을 생각하면 정확도와 함께 안정성, 연산량, 도메인 적합성까지 같이 봐야 한다는 점이 남았습니다. 개인적으로는 여러 백본을 직접 바꿔가며 결과 차이를 확인한 경험이, 경량 모델 설계와 실전 배포 관점을 함께 이해하는 데 큰 도움이 되었습니다.

## 논문 정보

- 정회준, 김승하, 김준희, 권장우
- "대조학습 방법을 이용한 주행 패턴 분석 기법 연구"
- 한국ITS학회논문지, `23권 1호`, `182-196`, `2024`
- DOI: [10.12815/kits.2024.23.1.182](https://doi.org/10.12815/kits.2024.23.1.182)

</div>

<aside id="contrastive-project-viewer">
  <div class="contrastive-panel">
    <p class="contrastive-eyebrow">Paper Viewer</p>
    <h2 class="contrastive-title">Contrastive Driving Paper</h2>
    <p class="contrastive-subtitle">대조학습 기반 주행 패턴 분석 논문 원문을 같은 페이지에서 바로 확인할 수 있도록 붙였습니다.</p>
    <div class="contrastive-pdf-frame">
      <iframe src="/assets/papers/contrastive-driving-pattern-analysis-2024.pdf#view=FitH" title="대조학습 방법을 이용한 주행 패턴 분석 기법 연구 PDF"></iframe>
    </div>
    <p class="contrastive-note">브라우저 내장 PDF 뷰어에 따라 확대 비율과 인터페이스는 조금 다르게 보일 수 있습니다.</p>
  </div>
</aside>
</div>

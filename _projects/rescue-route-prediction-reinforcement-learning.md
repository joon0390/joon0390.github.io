---
title: "AI 기반 산악사고 요구조자 이동경로 예측 시스템"
excerpt: "공간정보 전처리와 DQN 기반 강화학습을 결합해 산악사고 요구조자의 이동 경로를 예측하고 수색 지역을 좁히는 특허 기반 프로젝트."
date: 2026-04-18
collection: projects
layout: single
order: 3
classes:
  - wide
tags:
  - Reinforcement Learning
  - DQN
  - Path Prediction
  - Geographic Information System
  - Patent
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

  #rescue-project-layout {
    display: block;
  }

  #rescue-project-main {
    min-width: 0;
    max-width: 56rem;
  }

  #rescue-project-viewer {
    margin-top: 2rem;
  }

  .rescue-copy {
    word-break: keep-all;
    overflow-wrap: break-word;
  }

  .rescue-panel {
    border: 1px solid rgba(20, 54, 82, 0.12);
    border-radius: 1.25rem;
    background:
      radial-gradient(circle at top, rgba(255, 255, 255, 0.78), transparent 42%),
      linear-gradient(180deg, #f8fbfc 0%, #edf4f6 100%);
    box-shadow: 0 18px 40px rgba(17, 57, 91, 0.08);
    padding: 1.1rem;
  }

  .rescue-eyebrow {
    margin: 0 0 0.35rem;
    color: #73899b;
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.14em;
    text-transform: uppercase;
  }

  .rescue-title {
    margin: 0 0 0.35rem;
    color: #103754;
    font-size: 1.05rem;
    font-weight: 700;
    line-height: 1.4;
  }

  .rescue-subtitle {
    margin: 0 0 0.9rem;
    color: #627888;
    font-size: 0.92rem;
    line-height: 1.65;
  }

  .rescue-pdf-frame {
    overflow: hidden;
    border: 1px solid rgba(18, 57, 91, 0.1);
    border-radius: 1rem;
    background: #fff;
    height: 76vh;
    min-height: 52rem;
  }

  .rescue-pdf-frame iframe {
    display: block;
    width: 100%;
    height: 100%;
    border: 0;
  }

  .rescue-note {
    margin: 0.85rem 0 0;
    color: #627888;
    font-size: 0.9rem;
    line-height: 1.6;
  }

  .rescue-flow {
    margin: 1.7rem 0 2.1rem;
  }

  .rescue-flow__head {
    margin-bottom: 0.9rem;
  }

  .rescue-flow__head h3 {
    margin: 0 0 0.25rem;
    color: #103754;
    font-size: 1.1rem;
  }

  .rescue-flow__head p {
    margin: 0;
    color: #627888;
    font-size: 0.95rem;
  }

  .rescue-flow-grid {
    display: grid;
    gap: 0.9rem;
  }

  .rescue-flow-card {
    border: 1px solid rgba(18, 57, 91, 0.1);
    border-radius: 1rem;
    background: #fff;
    padding: 1rem;
  }

  .rescue-flow-card h4 {
    margin: 0 0 0.5rem;
    color: #103754;
    font-size: 1rem;
  }

  .rescue-flow-card p {
    margin: 0;
    color: #4f6678;
    font-size: 0.95rem;
    line-height: 1.65;
  }

  .rescue-flow-arrow {
    display: none;
  }

  .rescue-summary-grid {
    display: grid;
    gap: 0.9rem;
    margin: 1rem 0 1.8rem;
  }

  .rescue-summary-box {
    border-radius: 1rem;
    padding: 1rem;
  }

  .rescue-summary-box h4 {
    margin: 0 0 0.35rem;
    color: #103754;
    font-size: 1rem;
  }

  .rescue-summary-box p {
    margin: 0;
    color: #4f6678;
    font-size: 0.95rem;
    line-height: 1.65;
  }

  .rescue-summary-box--mint {
    background: #e4f6ef;
  }

  .rescue-summary-box--blue {
    background: #e8f2fb;
  }

  @media (min-width: 980px) {
    #rescue-project-layout {
      display: grid;
      grid-template-columns: minmax(0, 1.55fr) clamp(28rem, 30vw, 34rem);
      gap: clamp(1.75rem, 2.4vw, 3rem);
      align-items: start;
    }

    #rescue-project-viewer {
      margin-top: 0;
      position: sticky;
      top: 1rem;
      width: 100%;
      max-width: 34rem;
      justify-self: end;
    }

    .rescue-flow-grid {
      grid-template-columns: minmax(0, 1fr) 2rem minmax(0, 1fr) 2rem minmax(0, 1fr);
      align-items: stretch;
    }

    .rescue-flow-arrow {
      display: flex;
      align-items: center;
      justify-content: center;
      color: #18839a;
      font-size: 1.6rem;
      font-weight: 700;
    }

    .rescue-summary-grid {
      grid-template-columns: repeat(2, minmax(0, 1fr));
    }
  }

  @media (max-width: 979px) {
    .rescue-pdf-frame {
      height: 72vh;
      min-height: 38rem;
    }
  }

  @media (max-width: 640px) {
    .rescue-panel {
      padding: 0.9rem;
    }

    .rescue-pdf-frame {
      height: 68vh;
      min-height: 32rem;
    }
  }
</style>

<div id="rescue-project-layout">
<div id="rescue-project-main" class="rescue-copy" markdown="1">

## 프로젝트 개요

산악사고 발생 시 요구조자의 마지막 위치만으로는 실제 이동 방향을 빠르게 좁히기 어렵습니다. 이 프로젝트는 산악 지형과 공간정보를 전처리한 뒤, 강화학습 기반 AI 모델로 요구조자의 이동 경로를 시뮬레이션해 수색 범위를 더 빠르게 압축하는 시스템을 설계한 특허 기반 작업입니다. 첨부한 등록특허공보(B1) 문서를 기준으로 정리했습니다.

<section class="rescue-flow">
  <div class="rescue-flow__head">
    <h3>시스템 구성</h3>
    <p>특허 문서에서 확인되는 3단 구조를 기준으로 정리했습니다.</p>
  </div>
  <div class="rescue-flow-grid">
    <div class="rescue-flow-card">
      <h4>공간정보 데이터 처리부</h4>
      <p>산악 지형, 경사도, 수계, 도로, 등산로, 유역 경계 같은 공간정보를 불러와 경로 예측용 입력으로 정리합니다.</p>
    </div>
    <div class="rescue-flow-arrow" aria-hidden="true">→</div>
    <div class="rescue-flow-card">
      <h4>이동경로 예측 AI 모델부</h4>
      <p>전처리된 공간정보와 보상 함수를 결합해 DQN 기반 강화학습으로 요구조자 이동 정책을 학습합니다.</p>
    </div>
    <div class="rescue-flow-arrow" aria-hidden="true">→</div>
    <div class="rescue-flow-card">
      <h4>이동경로 예측부</h4>
      <p>학습된 정책으로 다회 시뮬레이션을 수행하고, 이동 가능 경로를 시각화해 구조 판단에 활용할 수 있게 제공합니다.</p>
    </div>
  </div>
</section>

## 핵심 내용

- 입력 데이터로는 경사도와 함께 저수지, 강, 등산로, 도로, 유역 경계, 수로 등 산악 공간정보가 사용됩니다.
- 에이전트는 요구조자를 가정하며, 나이, 성별, 건강 상태, 탐험 비율 같은 속성을 반영해 행동 특성이 달라지도록 설계됩니다.
- 보상 함수는 단순 최단거리보다 실제 산악 이동 특성을 반영하도록 구성되며, 위치 기반 선호, 유역 경계 회피, 거리, 고도, 상태 변화 같은 요소를 포함합니다.
- 학습 방식은 `DQN` 기반이며, `epsilon-greedy` 전략으로 탐색과 활용의 균형을 맞추도록 설계됩니다.
- 최종 출력은 한 번의 단일 경로가 아니라, 다회 시뮬레이션을 통해 얻은 이동 가능 경로 집합과 그 시각화 결과입니다.

<div class="rescue-summary-grid">
  <div class="rescue-summary-box rescue-summary-box--mint">
    <h4>입력 관점</h4>
    <p>GIS 기반 공간정보를 단순 배경 지도가 아니라, 강화학습 상태를 구성하는 핵심 피처로 다뤘습니다.</p>
  </div>
  <div class="rescue-summary-box rescue-summary-box--blue">
    <h4>출력 관점</h4>
    <p>정답 경로 하나를 고정적으로 찍기보다, 여러 번의 시뮬레이션을 통해 수색 우선 구역을 좁히는 방식에 가깝습니다.</p>
  </div>
</div>

## 데이터 및 예측 절차

1. 산악 지역의 공간정보를 수집하고 경로 예측용 피처로 전처리합니다.
2. 요구조자 속성과 보상 함수를 정의해 강화학습 환경을 구성합니다.
3. DQN 기반 학습으로 상태-행동 값을 추정하고 이동 정책을 학습합니다.
4. 학습된 정책으로 여러 차례 시뮬레이션을 수행해 후보 경로를 생성합니다.
5. 예측 결과를 시각화해 수색 우선 구역과 구조 동선을 판단할 수 있도록 제공합니다.

## 프로젝트 의의

이 작업의 핵심은 산악 수색 문제를 정적인 지도 분석이 아니라, 불확실한 인간 이동을 포함한 순차적 의사결정 문제로 다시 모델링했다는 점입니다. 특히 공간정보 전처리, 강화학습 모델, 경로 시뮬레이션, 시각화까지 하나의 시스템으로 묶어 실제 수색 지원 시나리오로 연결했다는 점에서 의미가 있습니다.

또한 예측 결과를 통해 수색 지역을 더 좁히고, 구조 인력과 장비 배치를 우선순위화할 수 있기 때문에, 구조 초기 대응 시간을 줄이는 의사결정 도구로 확장 가능성이 높습니다.

## 느낀점

가장 크게 남은 부분은, 이전까지 강화학습을 직접 다뤄본 적이 없었는데도 이 프로젝트를 진행하면서 짧은 시간 안에 상태, 행동, 보상, 탐색-활용 균형 같은 핵심 개념을 빠르게 체득하고 바로 구현으로 연결해야 했다는 점입니다. 단순히 이론을 읽고 이해하는 수준이 아니라, 실제 산악 지형 문제에 맞게 강화학습 환경을 구성하고 보상 구조를 조정하면서 몸으로 익히듯 학습했다는 점이 개인적으로 의미가 컸습니다.

또 하나 인상적이었던 부분은 데이터 처리 방식이었습니다. 이 프로젝트는 일반적인 표 형태 데이터만으로 풀 수 없어서 `geopandas`를 사용해 등산로, 도로, 수계, 유역 경계 같은 공간 데이터를 직접 다뤄야 했고, 그 과정에서 좌표계와 레이어 기반 데이터 처리 방식에 자연스럽게 익숙해질 수 있었습니다. 덕분에 하나의 모델을 만드는 경험에 그치지 않고, 다양한 공간 데이터 형식과 분석 도구를 빠르게 받아들이고 연결하는 감각까지 함께 키울 수 있었습니다.

## 특허 정보

- 명칭: AI 기반 산악사고 요구조자 이동경로 예측 시스템
- 출원번호: `10-2024-0185328`
- 출원일: `2024-12-12`
- 등록번호: `10-2864114`
- 등록일: `2025-09-19`
- 공고일: `2025-09-24`
- 문서 기준: 첨부한 등록특허공보(B1) PDF
- 관련 링크: [10.8080/1020240185328](https://doi.org/10.8080/1020240185328)

</div>

<aside id="rescue-project-viewer">
  <div class="rescue-panel">
    <p class="rescue-eyebrow">Patent Viewer</p>
    <h2 class="rescue-title">Registered Patent B1</h2>
    <p class="rescue-subtitle">AI 기반 산악사고 요구조자 이동경로 예측 시스템 특허 원문을 같은 페이지에서 바로 확인할 수 있도록 붙였습니다.</p>
    <div class="rescue-pdf-frame">
      <iframe src="/assets/papers/rescue-route-prediction-patent-1020240185328.pdf#view=FitH" title="AI 기반 산악사고 요구조자 이동경로 예측 시스템 특허 PDF"></iframe>
    </div>
    <p class="rescue-note">브라우저 내장 PDF 뷰어에 따라 확대 비율과 인터페이스는 조금 다르게 보일 수 있습니다.</p>
  </div>
</aside>
</div>

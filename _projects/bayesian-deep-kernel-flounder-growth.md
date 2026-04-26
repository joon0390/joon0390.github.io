---
title: "베이지안 딥커널 머신을 이용한 양식 넙치 성장 예측"
excerpt: "수온, 용존산소, 사료량을 바탕으로 넙치 성장량을 예측하고 BDKMR로 기존 커널 모델보다 더 낮은 오차를 달성한 연구."
date: 2026-04-18
collection: projects
layout: single
order: 4
tags:
  - Bayesian Machine Learning
  - Deep Kernel Learning
  - Gaussian Process
  - Aquaculture
  - Growth Prediction
---

<style>
  #flounder-project-layout {
    display: block;
  }

  #flounder-project-main {
    min-width: 0;
    max-width: 48rem;
  }

  #flounder-project-viewer {
    margin-top: 2rem;
  }

  .flounder-panel {
    border: 1px solid rgba(18, 57, 91, 0.12);
    border-radius: 1.25rem;
    background:
      radial-gradient(circle at top, rgba(255, 255, 255, 0.72), transparent 40%),
      linear-gradient(180deg, #f8fbfc 0%, #edf5f7 100%);
    box-shadow: 0 18px 40px rgba(17, 57, 91, 0.08);
    padding: 1.1rem;
  }

  .flounder-eyebrow {
    margin: 0 0 0.35rem;
    color: #73899b;
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.14em;
    text-transform: uppercase;
  }

  .flounder-title {
    margin: 0 0 0.9rem;
    color: #103754;
    font-size: 1.05rem;
    font-weight: 700;
    line-height: 1.4;
  }

  .flounder-pdf-frame {
    overflow: hidden;
    border: 1px solid rgba(18, 57, 91, 0.1);
    border-radius: 1rem;
    background: #fff;
    height: 76vh;
    min-height: 52rem;
  }

  .flounder-pdf-frame iframe {
    display: block;
    width: 100%;
    height: 100%;
    border: 0;
  }

  .flounder-note {
    margin: 0.85rem 0 0;
    color: #627888;
    font-size: 0.9rem;
    line-height: 1.6;
  }

  .flounder-copy {
    word-break: keep-all;
    overflow-wrap: break-word;
  }

  .flounder-visual {
    margin: 1.75rem 0 2.2rem;
  }

  .flounder-visual + .flounder-visual {
    margin-top: 1.5rem;
  }

  .flounder-visual__head {
    margin-bottom: 0.9rem;
  }

  .flounder-visual__head h3 {
    margin: 0 0 0.25rem;
    color: #103754;
    font-size: 1.1rem;
  }

  .flounder-visual__head p {
    margin: 0;
    color: #627888;
    font-size: 0.95rem;
  }

  .flounder-pipeline-grid {
    display: grid;
    gap: 0.9rem;
  }

  .flounder-pipeline-card {
    border: 1px solid rgba(18, 57, 91, 0.1);
    border-radius: 1rem;
    background: #fff;
    padding: 1rem;
  }

  .flounder-pipeline-card h4 {
    margin: 0 0 0.55rem;
    color: #103754;
    font-size: 1rem;
  }

  .flounder-pipeline-card p,
  .flounder-pipeline-card li {
    margin: 0;
    color: #4f6678;
    font-size: 0.95rem;
    line-height: 1.65;
  }

  .flounder-pipeline-card ul {
    margin: 0;
    padding-left: 1.15rem;
  }

  .flounder-pipeline-arrow {
    display: none;
  }

  .flounder-summary-grid {
    display: grid;
    gap: 0.9rem;
    margin-top: 0.95rem;
  }

  .flounder-summary-box {
    border-radius: 1rem;
    padding: 1rem;
  }

  .flounder-summary-box h4 {
    margin: 0 0 0.35rem;
    color: #103754;
    font-size: 1rem;
  }

  .flounder-summary-box p {
    margin: 0;
    color: #4f6678;
    font-size: 0.95rem;
    line-height: 1.65;
  }

  .flounder-summary-box--mint {
    background: #e3f7f0;
  }

  .flounder-summary-box--blue {
    background: #e8f2fb;
  }

  .flounder-metric-grid {
    display: grid;
    gap: 1rem;
  }

  .flounder-metric-card {
    border: 1px solid rgba(18, 57, 91, 0.1);
    border-radius: 1rem;
    background: #fff;
    padding: 1rem;
  }

  .flounder-metric-card h4 {
    margin: 0 0 0.9rem;
    color: #103754;
    font-size: 1rem;
  }

  .flounder-metric-row + .flounder-metric-row {
    margin-top: 0.8rem;
  }

  .flounder-metric-row-head {
    display: flex;
    justify-content: space-between;
    gap: 0.8rem;
    margin-bottom: 0.35rem;
    color: #103754;
    font-size: 0.92rem;
    font-weight: 600;
  }

  .flounder-track {
    height: 0.7rem;
    border-radius: 999px;
    background: #e7eef3;
    overflow: hidden;
  }

  .flounder-bar {
    height: 100%;
    border-radius: 999px;
  }

  .flounder-bar--krr {
    background: #bccdda;
  }

  .flounder-bar--bkmr {
    background: #7da4c1;
  }

  .flounder-bar--equal {
    background: #7fd6c6;
  }

  .flounder-bar--bdkmr {
    background: #0f7c87;
  }

  .flounder-metric-foot {
    margin-top: 0.9rem;
    color: #627888;
    font-size: 0.88rem;
    line-height: 1.6;
  }

  @media (min-width: 980px) {
    #flounder-project-layout {
      display: grid;
      grid-template-columns: minmax(0, 1.45fr) clamp(27rem, 31vw, 32rem);
      gap: 2.5rem;
      align-items: start;
    }

    #flounder-project-viewer {
      margin-top: 0;
      position: sticky;
      top: 1rem;
      width: 100%;
      max-width: 32rem;
      justify-self: end;
    }

    .flounder-pipeline-grid {
      grid-template-columns: minmax(0, 1fr) 2rem minmax(0, 1fr) 2rem minmax(0, 1fr);
      align-items: stretch;
    }

    .flounder-pipeline-arrow {
      display: flex;
      align-items: center;
      justify-content: center;
      color: #18839a;
      font-size: 1.6rem;
      font-weight: 700;
    }

    .flounder-summary-grid,
    .flounder-metric-grid {
      grid-template-columns: repeat(2, minmax(0, 1fr));
    }
  }

  @media (max-width: 979px) {
    .flounder-pdf-frame {
      height: 72vh;
      min-height: 38rem;
    }
  }

  @media (max-width: 640px) {
    .flounder-panel {
      padding: 0.9rem;
    }

    .flounder-pdf-frame {
      height: 68vh;
      min-height: 32rem;
    }
  }
</style>

<div id="flounder-project-layout">
<div id="flounder-project-main" class="flounder-copy" markdown="1">

## 프로젝트 개요

국내 양식 넙치(olive flounder)의 성장 예측을 위해, 가우시안 프로세스 회귀와 신경망 기반 표현 학습을 결합한 Bayesian Deep Kernel Machine Regression (BDKMR) 모델을 제안한 연구입니다. 완도 2개 양식장과 제주 3개 양식장, 총 7개 수조에서 2023년 3월부터 2024년 7월까지 수집한 종단 데이터를 바탕으로 수온, 용존산소, 개체당 사료량이 성장에 미치는 비선형 관계를 모델링했습니다.

## 시각 자료

<section class="flounder-visual flounder-panel">
  <div class="flounder-visual__head">
    <h3>아키텍처 요약</h3>
    <p>데이터 정렬, 신경망 기반 특징 학습, 베이지안 커널 회귀를 한 흐름으로 정리했습니다.</p>
  </div>
  <div class="flounder-pipeline-grid">
    <div class="flounder-pipeline-card">
      <h4>1. Aquaculture Data</h4>
      <ul>
        <li>완도 2개, 제주 3개 양식장</li>
        <li>총 7개 수조, 2023.03-2024.07</li>
        <li>수온, 용존산소, 개체당 사료량, 초기 로그 체중</li>
      </ul>
    </div>
    <div class="flounder-pipeline-arrow" aria-hidden="true">→</div>
    <div class="flounder-pipeline-card">
      <h4>2. Feature Learning</h4>
      <p>측정 주기가 다른 센서 데이터와 월별 체중 데이터를 동일한 성장 구간으로 정렬한 뒤, ANN 기반 feature map으로 비선형 구조를 학습합니다.</p>
    </div>
    <div class="flounder-pipeline-arrow" aria-hidden="true">→</div>
    <div class="flounder-pipeline-card">
      <h4>3. Bayesian Kernel Layer</h4>
      <p>Gaussian process 기반 BDKMR로 예측과 불확실성을 함께 추정하고, 관측별 정밀도를 반영하기 위해 이분산 구조 <code>Var(y_i)=sigma^2/n_i</code> 를 사용합니다.</p>
    </div>
  </div>
  <div class="flounder-summary-grid">
    <div class="flounder-summary-box flounder-summary-box--mint">
      <h4>반응 변수</h4>
      <p>농가별 무작위 50마리 표본에서 계산한 월별 로그 평균 체중</p>
    </div>
    <div class="flounder-summary-box flounder-summary-box--blue">
      <h4>추론 방식</h4>
      <p>MAP estimation과 Laplace approximation을 사용해 계산 가능성과 베이지안 구조를 동시에 확보</p>
    </div>
  </div>
</section>

<section class="flounder-visual flounder-panel">
  <div class="flounder-visual__head">
    <h3>성능 비교</h3>
    <p>논문에 보고된 LOOCV 기준 MAE, MSE 수치를 그대로 시각화했습니다.</p>
  </div>
  <div class="flounder-metric-grid">
    <div class="flounder-metric-card">
      <h4>MAE</h4>
      <div class="flounder-metric-row">
        <div class="flounder-metric-row-head"><span>KRR</span><span>1.1141</span></div>
        <div class="flounder-track"><div class="flounder-bar flounder-bar--krr" style="width:100%"></div></div>
      </div>
      <div class="flounder-metric-row">
        <div class="flounder-metric-row-head"><span>BKMR</span><span>0.6977</span></div>
        <div class="flounder-track"><div class="flounder-bar flounder-bar--bkmr" style="width:62.6%"></div></div>
      </div>
      <div class="flounder-metric-row">
        <div class="flounder-metric-row-head"><span>BDKMR(Equal)</span><span>0.2006</span></div>
        <div class="flounder-track"><div class="flounder-bar flounder-bar--equal" style="width:18.0%"></div></div>
      </div>
      <div class="flounder-metric-row">
        <div class="flounder-metric-row-head"><span>BDKMR</span><span>0.1895</span></div>
        <div class="flounder-track"><div class="flounder-bar flounder-bar--bdkmr" style="width:17.0%"></div></div>
      </div>
    </div>
    <div class="flounder-metric-card">
      <h4>MSE</h4>
      <div class="flounder-metric-row">
        <div class="flounder-metric-row-head"><span>KRR</span><span>3.5665</span></div>
        <div class="flounder-track"><div class="flounder-bar flounder-bar--krr" style="width:100%"></div></div>
      </div>
      <div class="flounder-metric-row">
        <div class="flounder-metric-row-head"><span>BKMR</span><span>0.9447</span></div>
        <div class="flounder-track"><div class="flounder-bar flounder-bar--bkmr" style="width:26.5%"></div></div>
      </div>
      <div class="flounder-metric-row">
        <div class="flounder-metric-row-head"><span>BDKMR(Equal)</span><span>0.0721</span></div>
        <div class="flounder-track"><div class="flounder-bar flounder-bar--equal" style="width:2.0%"></div></div>
      </div>
      <div class="flounder-metric-row">
        <div class="flounder-metric-row-head"><span>BDKMR</span><span>0.0629</span></div>
        <div class="flounder-track"><div class="flounder-bar flounder-bar--bdkmr" style="width:1.8%"></div></div>
      </div>
      <p class="flounder-metric-foot">BDKMR은 BKMR 대비 MAE를 약 72.8%, MSE를 약 93.3% 줄였습니다.</p>
    </div>
  </div>
</section>

## 핵심 내용

- 수온과 용존산소는 1분 단위 센서 데이터, 사료량은 일 단위 기록, 체중은 월 단위 측정으로 수집되었으며, 이를 동일한 성장 관측 구간에 맞춰 정렬해 분석용 데이터셋을 구성했습니다.
- 월별 체중 측정에서는 농가별로 무작위 50마리를 표본 추출해 로그 평균 체중을 반응변수로 사용했고, 개체 수와 측정 변동성을 반영하기 위해 `Var(y_i) = \sigma^2 / n_i` 형태의 이분산 구조를 적용했습니다.
- BKMR의 불확실성 정량화 장점과 ANN의 표현 학습 능력을 결합한 BDKMR을 설계해, 환경 변수 간 복잡한 비선형 상호작용을 더 유연하게 학습하도록 구성했습니다.
- 추론은 MAP 추정과 Laplace approximation을 기반으로 수행해 베이지안 구조를 유지하면서도 실제 예측 문제에 적용 가능한 계산 효율을 확보했습니다.

## 데이터 및 실험 설계

- 대상 데이터: 완도 2개 양식장, 제주 3개 양식장, 총 7개 수조
- 수집 기간: 2023년 3월부터 2024년 7월까지
- 입력 변수: 수온, 용존산소, 개체당 사료량, 초기 로그 체중
- 반응 변수: 월별 로그 평균 체중
- 비교 모델: KRR, BKMR, BDKMR(Equal), BDKMR
- 평가 지표: Leave-One-Out Cross-Validation (LOOCV), MAE, MSE

## 주요 결과

- 제안한 BDKMR은 `MAE 0.1895`, `MSE 0.0629`를 기록해 비교 모델 중 가장 낮은 예측 오차를 보였습니다.
- 동분산 가정의 `BDKMR(Equal)`보다 이분산 구조를 반영한 BDKMR이 더 우수해, 관측별 정밀도 차이를 고려하는 것이 실제 양식 데이터 예측에 중요함을 확인했습니다.
- 기준 모델인 KRR은 `MAE 1.1141`, `MSE 3.5665`, BKMR은 `MAE 0.6977`, `MSE 0.9447`로 나타나, 딥커널 기반 표현 학습이 기존 커널 모델 대비 뚜렷한 성능 개선을 제공했습니다.

## 사용 기술

- Bayesian Deep Kernel Machine Regression
- Gaussian Process Regression
- Artificial Neural Networks
- Heteroscedastic Modeling
- Leave-One-Out Cross-Validation

## 프로젝트 의의

양식 데이터는 변수별 측정 주기가 다르고 환경 스트레스에 따라 변동성이 크게 달라지는 특성이 있습니다. 이 연구는 딥러닝의 표현 학습과 베이지안 커널 모델의 해석 가능성 및 불확실성 추정을 결합해, 급이 전략, 출하 시점, 환경 제어와 같은 실제 양식 운영 의사결정에 활용할 수 있는 성장 예측 프레임워크를 제시했다는 점에서 의미가 있습니다.

## 논문 정보

- Junhee Kim, Seung-Won Seo, Ho-Jin Jung, Hyun-Seok Jang, Han-Kyu Lim, Seongil Jo, "Predicting Flatfish Growth in Aquaculture Using Bayesian Deep Kernel Machines", *Applied Sciences*, 2025.
- DOI: [10.3390/app15179487](https://doi.org/10.3390/app15179487)

</div>

<aside id="flounder-project-viewer">
  <div class="flounder-panel">
    <p class="flounder-eyebrow">Paper Viewer</p>
    <p class="flounder-title">Applied Sciences 2025 Full Paper</p>
    <div class="flounder-pdf-frame">
      <iframe
        src="{{ '/assets/papers/flounder-bdkmr-applsci-2025.pdf' | relative_url }}#page=1&zoom=125&pagemode=none"
        title="Predicting Flatfish Growth in Aquaculture Using Bayesian Deep Kernel Machines PDF"
      ></iframe>
    </div>
    <p class="flounder-note">
      브라우저에서 PDF 임베드가 지원되지 않으면
      <a href="{{ '/assets/papers/flounder-bdkmr-applsci-2025.pdf' | relative_url }}" target="_blank" rel="noopener">새 탭에서 논문 열기</a>
    </p>
  </div>
</aside>
</div>

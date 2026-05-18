---
title: "About Me"
permalink: /about/
layout: single
author_profile: false
classes:
  - about-page
---

<div class="about-switch">
  <input class="about-switch__input" type="radio" name="about-language" id="about-lang-kor" checked>
  <input class="about-switch__input" type="radio" name="about-language" id="about-lang-eng">

  <div class="about-switch__controls" aria-label="Language switch">
    <label class="about-switch__button" for="about-lang-kor">KOR</label>
    <label class="about-switch__button" for="about-lang-eng">ENG</label>
  </div>

  <div class="about-switch__panels">
    <section class="about-panel about-panel--kor" lang="ko">
      <div class="about-hero">
        <div class="about-hero__media">
          <img src="/assets/img/taipei.png" alt="김준희 프로필 사진">
        </div>

        <div class="about-hero__body">
          <p class="about-hero__eyebrow">Bayesian ML · Deep Learning · Graphical Models</p>
          <h2 class="about-hero__title">안녕하세요, 저는 김준희입니다.</h2>
          <p class="about-hero__lead">베이지안 머신러닝, 딥러닝을 중심으로 공부하고 있는 주니어 데이터 사이언티스트입니다. 이 블로그에는 논문 리뷰, 이론 정리, 구현 기록, 실험 과정, 그리고 여러 메모를 정리하고 있습니다.</p>
          <p class="about-hero__text">확률적 모델, 신경망, 그래프 기반 방법론을 코드와 함께 정리하면서, 공부한 내용을 제 방식대로 구조화해 두는 곳으로 운영하고 있습니다.</p>

          <p class="about-hero__actions">
            <a href="/cv/" class="btn btn--primary">CV 보기</a>
            <a href="/cv/data/Portfolio.pdf" class="btn btn--inverse">포트폴리오 보기</a>
          </p>
        </div>
      </div>

      <div class="about-grid">
        <section class="about-card">
          <p class="about-card__eyebrow">What I Write</p>
          <h3 class="about-card__title">블로그에서 다루는 내용</h3>
          <ul class="about-card__list">
            <li>이론 개념 정리</li>
            <li>코드 구현 및 실험 기록</li>
            <li>논문 리뷰와 핵심 아이디어 요약</li>
            <li>오류 해결과 디버깅 메모</li>
          </ul>
        </section>

        <section class="about-card">
          <p class="about-card__eyebrow">Focus Areas</p>
          <h3 class="about-card__title">주요 관심 분야</h3>
          <ul class="about-card__list">
            <li>Bayesian Inference</li>
            <li>Probabilistic Deep Learning</li>
            <li>Graphical Models</li>
            <li>Time Series, Representation Learning</li>
          </ul>
        </section>

        <section class="about-card about-card--wide">
          <p class="about-card__eyebrow">Archive Note</p>
          <h3 class="about-card__title">이 페이지의 역할</h3>
          <p class="about-card__text">이 블로그는 제가 공부하고 구현한 내용을 공유하고, 복기할 수 있도록 정리해두는 개인 연구 블로그입니다. 시간이 지나도 흐름을 따라가기 쉽도록, 이론과 코드 사이를 연결하는 글들을 정리해가고 있습니다.</p>
        </section>
      </div>
    </section>

    <section class="about-panel about-panel--eng" lang="en">
      <div class="about-hero">
        <div class="about-hero__media">
          <img src="/assets/img/taipei.png" alt="Junhee Kim profile photo">
        </div>

        <div class="about-hero__body">
          <p class="about-hero__eyebrow">Bayesian ML · Deep Learning · Graphical Models</p>
          <h2 class="about-hero__title">Hello, I'm Junhee Kim.</h2>
          <p class="about-hero__lead">I am an aspiring data scientist focusing on Bayesian machine learning, deep learning, and graphical models. This blog is where I collect theory notes, implementation logs, experiment records, and research thoughts in one place.</p>
          <p class="about-hero__text">I use this space as a working archive for probabilistic models, neural networks, and graph-based methods, with a strong emphasis on turning ideas into code and documenting the process clearly.</p>

          <p class="about-hero__actions">
            <a href="/cv/" class="btn btn--primary">View CV</a>
            <a href="/cv/data/Portfolio.pdf" class="btn btn--inverse">View Portfolio</a>
          </p>
        </div>
      </div>

      <div class="about-grid">
        <section class="about-card">
          <p class="about-card__eyebrow">What I Write</p>
          <h3 class="about-card__title">Topics on This Blog</h3>
          <ul class="about-card__list">
            <li>Theory breakdowns</li>
            <li>Implementation notes and experiments</li>
            <li>Paper reviews and key takeaways</li>
            <li>Troubleshooting and debugging records</li>
          </ul>
        </section>

        <section class="about-card">
          <p class="about-card__eyebrow">Focus Areas</p>
          <h3 class="about-card__title">Main Interests</h3>
          <ul class="about-card__list">
            <li>Bayesian Inference</li>
            <li>Probabilistic Deep Learning</li>
            <li>Graphical Models</li>
            <li>Time Series and Representation Learning</li>
          </ul>
        </section>

        <section class="about-card about-card--wide">
          <p class="about-card__eyebrow">Archive Note</p>
          <h3 class="about-card__title">Why This Page Exists</h3>
          <p class="about-card__text">This blog is more than a place to post finished results. It is a personal research archive where I organize what I study and build so that the path from theory to implementation remains easy to revisit later.</p>
        </section>
      </div>
    </section>
  </div>
</div>

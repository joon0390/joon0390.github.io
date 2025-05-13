---
title: "산업용 회전 장비 이상 진동 탐지 시스템 (Edge Computing)"
excerpt: "Edge Impulse 기반 엣지 컴퓨팅으로 산업용 회전 장비 이상 진동을 실시간 탐지·분류"
collection: projects
layout: posts
date: 2024-5-13
tags:
  - Edge Computing
  - Vibration Analysis
  - Anomaly Detection
  - Edge Impulse
  - Bluetooth LE
  - IoT
---

## 🧠 프로젝트 개요

이 프로젝트는 산업 현장에서 사용되는 **회전 장비의 진동 패턴을 실시간 분석**하여,  
**불균형(Imbalance)**, **축 불일치(Misalignment)**, **정상(Normal)** 3가지 상태로 분류하는  
**엣지 컴퓨팅 기반 이상 진동 탐지 시스템**을 구현한 과제입니다.

- Arduino Nano 33 BLE Sense로 진동 데이터를 수집하고,  
- **Edge Impulse** 플랫폼에서 모델을 학습한 뒤  
- 모바일 기기와 **Bluetooth Low Energy(BLE)**로 연결하여 실시간 진동 상태를 스트리밍합니다.

🔗 GitHub Repository: [Gaebobman/Edge-Computing](https://github.com/Gaebobman/Edge-Computing)

---

## 🔑 주요 기능

- ✅ 산업용 회전 장비 진동 신호 기반 **이상 탐지**
- ✅ **Edge Impulse**에서 모델 학습 및 최적화된 Arduino 라이브러리 생성
- ✅ Arduino에서 실시간 추론 결과를 **Bluetooth LE**로 전송
- ✅ 모바일 기기와 **BLE 통신 파이프라인 구성**
- ✅ 반복 실험을 위한 자동화 스크립트 및 로깅 시스템 구축

---

## 📁 프로젝트 디렉토리 구조

```text
Edge-Computing/
├── Arduino/                     # Arduino Nano 33 BLE Sense 코드 (C/C++)
│   └── vibration_inferencing    # Edge Impulse 추론 라이브러리 포함
├── assets/                      # 시스템 다이어그램, 스크린샷 등
├── experiments/                 # 실험 스크립트 및 설정
│   └── run_all.sh               # 반복 실험 자동화 배치 스크립트
├── logs/                        # 실험 및 추론 결과 로그 (CSV, JSON)
├── results/                     # 시각화된 실험 결과 요약 파일
├── vibration_inferencing.zip    # Edge Impulse 내보낸 추론 라이브러리
└── README.md                    # 프로젝트 개요 및 실행 설명
```

---

## ⚙️ 시스템 구성 흐름

1. **데이터 수집**  
   - Machine Fault Simulator를 통해 다양한 고장 상태를 시뮬레이션  
   - Arduino Nano 33 BLE Sense 내장 IMU로 진동 데이터 측정

2. **모델 학습**  
   - Edge Impulse 플랫폼에서 데이터 업로드, 라벨링, 전처리  
   - FFT 기반 특성 추출 및 경량 분류 모델 학습  
   - Arduino에서 실행 가능한 C 라이브러리로 내보내기

3. **실시간 추론 & 전송**  
   - 학습된 모델을 탑재한 Arduino가 진동 상태를 실시간 추론  
   - 추론 결과를 BLE로 모바일 디바이스에 송신

4. **모바일 연동**  
   - 모바일 기기에서 BLE 데이터를 수신  
   - 추론 결과를 실시간 로그 또는 시각화 형태로 저장

---

## 📊 실험 및 성능 결과

- **클래스**: Imbalance / Misalignment / Normal  
- **데이터 수집 시간**: 각 상태당 약 5분  
- **Sampling Rate**: 400 Hz  
- **Accuracy**: 96.8%  
- **Precision / Recall**  
  - Imbalance: 0.95 / 0.97  
  - Misalignment: 0.97 / 0.96  
  - Normal: 0.98 / 0.98

---

## 👨‍💻 기여 및 역할: 김준희 (Inha University, Dept. of Statistics)

- 🔧 **BLE 통신 파이프라인 구성**  
  - 엣지 디바이스와 모바일 간 Bluetooth Low Energy 연결 설계  
  - 추론 결과 실시간 스트리밍 구조 구현 및 검증

- 🧪 **반복 실험 자동화 및 데이터 수집 관리**  
  - `experiments/run_all.sh` 작성으로 다수 조건 실험을 자동 실행  
  - 실험 결과를 `logs/`에 자동 저장하여 분석 효율성 향상

- 📦 **Edge Impulse 기반 모델 구축 및 최적화**  
  - FFT 기반 특성 추출 구성, 모델 아키텍처 설정 및 하이퍼파라미터 튜닝  
  - 모델을 Arduino용 라이브러리로 익스포트하여 엣지 추론 구현

- 📐 **시각자료 구성 및 전체 시스템 아키텍처 설계**  
  - 진동 수집 → 모델 학습 → BLE 연동까지의 흐름 설계  
  - 시스템 구성도를 포함한 `assets/` 디렉토리 작성

---

## 📚 참고 자료

- [Edge Impulse 공식 문서](https://docs.edgeimpulse.com/)  
- [Arduino Nano 33 BLE Sense](https://www.arduino.cc/pro/hardware/product/NANO-33-BLE)  
- [GitHub Repository](https://github.com/Gaebobman/Edge-Computing)

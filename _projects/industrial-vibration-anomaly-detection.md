---
title: "Edge AI 기반 산업용 회전 장비 이상 진동 탐지"
excerpt: "Machine Fault Simulator에서 수집한 진동 데이터를 Edge Impulse로 학습하고, Arduino Nano 33 BLE Sense에 배포해 실시간 이상 감지와 알림 구조를 구현한 프로젝트."
date: 2023-02-01
collection: projects
layout: single
order: 2
period: "2023.06-2023.12"
tags:
  - Edge AI
  - TinyML
  - Edge Impulse
  - Arduino
  - Vibration Analysis
  - Predictive Maintenance
---
## 프로젝트 요약

- 개요: 산업용 회전 장비의 진동 데이터를 이용해 이상 상태를 감지하고, 학습된 모델을 마이크로컨트롤러에 배포한 Edge AI 프로젝트
- 기간: 2023.06-2023.12
- 데이터: Machine Fault Simulator에서 수집한 정상/이상 진동 신호
- 기술 스택: Arduino Nano 33 BLE Sense, Edge Impulse, Arduino IDE, Bluetooth, TinyML, Time Series Classification
- 성과(성능): 진동 데이터 수집, 모델 학습, Arduino Library export, 디바이스 업로드, 실시간 모니터링까지 이어지는 온디바이스 이상 감지 흐름 구현

## 문제 정의

산업용 회전 장비는 베어링, 축, 모터, 부하 상태에 따라 진동 패턴이 달라집니다. 고장이 완전히 발생한 뒤 대응하면 장비 정지와 유지보수 비용이 커지기 때문에, 현장에서 발생하는 진동 변화를 빠르게 감지하고 이상 가능성을 조기에 알려주는 구조가 필요합니다.

이 프로젝트는 클라우드 서버에서만 모델을 돌리는 방식이 아니라, 센서가 붙어 있는 소형 보드에서 직접 추론하는 Edge AI 구조를 목표로 했습니다. Machine Fault Simulator로 진동 데이터를 만들고, Arduino Nano 33 BLE Sense에서 데이터를 수집한 뒤, Edge Impulse에서 모델을 학습하고 Arduino Library로 export해 실제 보드에 통합하는 흐름으로 진행했습니다.

## 시스템 구성

전체 구조는 `데이터 생성/수집 → 전처리 → 모델 학습/export → Arduino 앱 통합 → 실시간 알림/모니터링`으로 나눌 수 있습니다.

1. Machine Fault Simulator에서 회전 장비의 정상 및 이상 진동 상황을 구성합니다.
2. Arduino Nano 33 BLE Sense의 센서 데이터를 이용해 진동 시계열을 수집합니다.
3. Edge Impulse에서 데이터 전처리와 모델 학습을 수행합니다.
4. 학습된 모델을 Arduino Library 형태로 export합니다.
5. Arduino IDE에서 모델 라이브러리를 애플리케이션 코드에 통합하고 compile/build 후 보드에 업로드합니다.
6. 보드에서 실시간 추론을 수행하고, 결과를 Bluetooth 기반 모니터링/알림 구조로 전달합니다.

이 흐름의 핵심은 모델 학습에서 끝나지 않고, 학습된 모델을 실제 임베디드 환경에 올려 추론까지 확인했다는 점입니다. Edge Impulse는 데이터 수집, feature extraction, 모델 학습, Arduino Library export를 한 흐름으로 연결할 수 있어 TinyML 프로토타이핑에 적합했습니다.

## 데이터와 EDA

진동 데이터는 시간 축에 따라 연속적으로 들어오는 센서 신호이기 때문에, 단일 시점 값보다 일정 구간의 패턴을 보는 것이 중요합니다. 정상 상태에서는 진동의 크기와 주파수 구성이 비교적 안정적으로 유지되지만, 회전 불균형이나 기계적 이상이 생기면 특정 축의 진동 크기, 주기성, 에너지 분포가 달라질 수 있습니다.

EDA에서는 다음 질문을 중심으로 데이터를 확인했습니다.

- 정상 상태와 이상 상태의 진동 크기 차이가 충분히 나타나는가?
- 짧은 window 안에서도 분류 가능한 패턴이 유지되는가?
- 센서 노이즈나 순간적인 튐이 모델을 과도하게 흔들지는 않는가?
- 실시간 추론을 위해 window size와 sampling 설정을 어느 정도로 잡아야 하는가?
- 보드에서 처리 가능한 수준으로 feature와 모델 크기를 줄일 수 있는가?

이 프로젝트에서는 데스크톱에서 높은 성능을 내는 모델보다, 작은 보드에서 안정적으로 돌아가는 모델이 더 중요했습니다. 따라서 EDA도 모델 정확도만 보기보다, 수집 신호의 안정성, window 단위 패턴, 온디바이스 추론 가능성을 함께 보는 방식으로 접근했습니다.

## 접근 방법

먼저 Arduino Nano 33 BLE Sense를 데이터 수집 장치로 사용해 Machine Fault Simulator의 진동 신호를 수집했습니다. 이후 Edge Impulse로 데이터를 업로드하고, 클래스별 데이터 균형과 신호 품질을 확인한 뒤 모델 학습을 진행했습니다.

모델 학습 이후에는 Edge Impulse의 Deployment 기능을 사용해 Arduino Library로 export했습니다. 이렇게 생성된 라이브러리를 Arduino IDE에 추가하고, 추론 코드를 기존 애플리케이션 흐름에 통합했습니다. 보드에 업로드된 프로그램은 센서 값을 읽고, 일정 window 단위로 feature를 구성한 뒤, 로컬에서 모델 추론을 수행합니다.

마지막 단계에서는 추론 결과를 사용자에게 전달하기 위한 실시간 모니터링 구조를 붙였습니다. 단순히 serial monitor에서 값만 확인하는 것이 아니라, Bluetooth를 통해 사용자 장치로 결과를 전달하고 알림/모니터링이 가능한 구조를 지향했습니다.

## 성과(성능)

가장 중요한 성과는 모델을 학습하는 데서 멈추지 않고, 임베디드 보드에 올려 실제 데이터 수집과 추론 흐름까지 연결했다는 점입니다.

- Machine Fault Simulator 기반 진동 데이터 수집 흐름 구성
- Edge Impulse 기반 전처리, 모델 학습, 배포 파이프라인 구성
- 학습 모델을 Arduino Library로 export
- Arduino Nano 33 BLE Sense 애플리케이션에 모델 통합
- 보드 compile/build 및 upload 흐름 확인
- Bluetooth 기반 실시간 알림/모니터링 구조 설계

정량 성능은 현재 글에 임의로 넣지 않았습니다. 성능을 추가한다면 Edge Impulse의 validation/test 결과를 기준으로 accuracy, confusion matrix, latency, RAM/Flash 사용량, false alarm rate를 함께 정리하는 것이 적합합니다.

## 느낀점

이 프로젝트를 하면서 가장 크게 느낀 점은, Edge AI 프로젝트는 모델을 잘 학습시키는 것만으로 끝나지 않는다는 점이었습니다. 데스크톱 환경에서는 잘 돌아가는 모델도 실제 보드에 올리면 메모리, 추론 속도, 센서 입력 형식, 빌드 환경 같은 제약을 바로 만나게 됩니다.

특히 Edge Impulse에서 Arduino Library로 export한 뒤, 이를 Arduino IDE 프로젝트에 통합하고 실제 보드에 업로드하는 과정이 중요했습니다. 모델 파일 하나를 만든다고 바로 현장에서 쓸 수 있는 것이 아니라, 센서 데이터가 들어오는 방식과 모델이 기대하는 입력 형식을 정확히 맞춰야 했습니다.

또한 실시간 알림 구조를 고민하면서, 이상 탐지는 단순히 분류 결과를 출력하는 문제가 아니라 사용자가 바로 이해하고 대응할 수 있는 형태로 전달되어야 한다는 점을 느꼈습니다. 이 경험 덕분에 TinyML, 센서 데이터, 임베디드 배포, 실시간 모니터링이 어떻게 하나의 시스템으로 연결되는지 더 구체적으로 이해할 수 있었습니다.

## 참고자료

- GitHub: [joon0390/Edge-Computing](https://github.com/joon0390/Edge-Computing)
- Edge Impulse Docs: [Run Arduino library](https://docs.edgeimpulse.com/hardware/deployments/run-arduino-2-0)

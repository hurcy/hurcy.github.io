---
layout: post
title:  "간단리뷰: A clinically applicable approach to continuous prediction of future acute kidney injury"
date:   2019-08-18 23:49:29 +0900
categories: ko
---

- 코호트:
703,782 명
미국의 보훈부 산하 여러 기관에서 취합함 (그래서 여성비율이 6.38%로 낮음)
입원은 172 기관, 외래는 1,062 기관에서 취합함
free text 사용하지 않음
민감한 질병정보(HIV, ADIS, 약물 남용, 정신질환) 사용하지 않음


- 데이터:
60억건의 이벤트(6,352,945,637건)

- 전처리:
6시간간씩 데이터 그룹화, 변수별로 6시간의 요약통계&augmentation을 모두 합해서 피처로 사용
missing values는 보간하지 않음
완전히 값이 없는것과 실제로 값이 0인것을 구분함. (NaN != 0)

인코딩
수치형 -> normal, low, high 등으로 변환
범주형 -> one-hot encoding

데이터를 총 29개의 카테고리로 분류함 (의료행위, 진단, 투약 등)
특히 AKI 진단에 중요한 크레아티닌은 다양한 통계치를 변수화했음
3가지 기간으로 과거 데이터의 요약을 만들었음 (지난 48시간, 6개월, 5년)

타임스텝
1day = 4개의 6h 버켓 + 1개의 unknown-time entries 버켓
정확한 시간을 알수 있는 35% 데이터만 6h로 데이터를 잘라서 그룹화
나머지는 따로 모아서 별도의 unknown-time entries bucket을 만듬
진단은 data leak을 방지하기위해 퇴원시점에 발생한것으로 처리함

- 변수:
620,000 개의 피처

- 타겟:
매 time step 마다 48h 이내의 급성신부전(AKI)
라벨링 기준: KDIGO AKI(Kidney Disease: Improving Global Outcomes) Acute Kidney Injury를 따름.
소변검사 시 크레아티닌 수치가 48h 이내 0.3mg/dl 상승한 경우
지난 7일동안 측정된 크레아티닌 수치보다 1.5배 상승한 경우
6시간 이내 소변양이 0.5ml/kg/h 이하인 경우 (이 조건은 데이터가 없어서 사용하지 않음)
라벨 비율: 전체 입원 중 13.4%

- 모델구조:
RNN 사용함.
입력데이터:
48h, 6개월, 5년 동안의 과거 데이터
현재 6h치 데이터

타겟 데이터:
AKI 예측변수
maximum future observed values of seven biochemical tests of renal function (향후 신장 기능을 의사가 이해하는데 도움을 줌.)

- 성능:
AKI
입원환자의 48시간 이내 AKI 이벤트 중 55.8%가 조기 예측됨
AUROC 0.921, AUPRC 0.297
투석
AKI 발병 후 30일 이내 투석 = 84.3%
AKI 발병 후 90일 이내 투석 = 90.2%

- 검증:
train(80), validation(5), calibration(5), test(10)
베이스라인 모델: Gradient boosted tree

- 임상 환경에 적용:
AKI위험도가 특정 threshold 보다 클 경우, postivie로 인식하고 의료진에게 알람을 줌.
2 false prediction for every true positive 로 예측함. (이게 33% precision이란 뜻인가?)
결론적으로, 이 모델이 위험하다고 한 환자들을 케어한다면, 입원환자 중 0.8%를 의사들이 케어하게 됨.
시점에 상관없이, 기관에 상관없이 모델이 일반화 가능한지 검증함.
threshold를 조절하면 AKI 알람을 늘이거나 줄일 수 있음.


https://www.nature.com/articles/s41586-019-1390-1

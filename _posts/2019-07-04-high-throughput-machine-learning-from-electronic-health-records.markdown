---
title: 간단리뷰: High-Throughput Machine Learning from Electronic Health Records
layout: post
categories: ko
published: true
---


질병 종류와 예측 기간별 성능 차이를 볼만한 논문이다.
랜덤 포레스트로 질병 진단을 예측했는데, 분산 노드를 왕창 사용해서 학습했기 때문에 High-Throughput 이 제목이 들어간 것 같다.


- 인상적인 부분
  - 12p: 예측기간별 AUC 그래프를 보면, 기간이 짧을수록 AUC가 높음. --> 장기간 미래 예측은 정확도가 떨어질 수밖에 없음.
  - 13p: 같은 모델이라도 질병마다 AUC가 다름. --> 성능을 올리려면 질병별로 모델이 따로 학습되어야겠음.
  

https://arxiv.org/pdf/1907.01901.pdf

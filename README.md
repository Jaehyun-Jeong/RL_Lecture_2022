
---
# 개요
---

## 프로젝트 개요

**강화학습 Agent** 개발에 앞서 가장 큰 문제 중 하나는 **환경**(**Environment**)입니다.

예를 들어, **자율주행 자동차**를 생각해 봅시다. 자율주행 자동차를 학습하기 위해 진짜 자동차를 사용한다면 아주 많은 자동차 사고가 발생한 후에야 자동차 Agent가 학습될 것입니다. 이러한 상황이 발생하지 않도록 우리는 가상의 환경에서 자율주행 자동차를 학습시킵니다.
하지만 자율주행 자동차를 학습시키기 위한 가상환경 만들기는 자동차 게임 하나를 제작하는 것과 다를 바 없으므로 아주 어렵습니다. 따라서 이번 프로젝트에서는 **유니티**로 미리 만들어놓은 템플릿으로 에이전트 학습을 진행하겠습니다.

## 목차

#### 1. Introduction

    1. 이 강의를 위한 사전지식
    2. 강화학습이란?
    3. DQN이란?

#### 2. Python gym을 사용한 강화학습 프로젝트

    1. Introduction
    2. 강화학습의 기본 개념
        1) 수학적 관점의 MDP(Markov Decision Process)와 강화학습적 관점의 MDP        
        2) 강화학습의 목표와 Bellman Equation
        3) Value Iteration과 Function Approximator
        4) off-polocy와 model-free 알고리즘
        5) DQN에서의 Loss Function과 Stochastic Gradient Descent
        6) DQN 알고리즘 해석하기
    3. 강화학습 Python 코드 작성
        1) Conda 가상환경 생성과 라이브러리 다운로드
        2) 모듈 불러오기
        3) Experience 클래스 정의
        4) model 클래스 정의
        5) Trainer 클래스 정의
        6) 학습 진행하기
    4. Summary

#### 3. Unity를 사용한 강화학습 프로젝트

    1. Introduction

#### 4. Conclusion

#### 5. Frontier
    
    1. META LEARNING
    2. SELF SUPERVISED

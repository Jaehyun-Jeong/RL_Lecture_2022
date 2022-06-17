![](https://firebasestorage.googleapis.com/v0/b/aing-biology.appspot.com/o/Introduction_to_RL%2F2.1%2Fthumbnail.png?alt=media&token=7b97bf80-2368-4dbf-95d9-5e677baa4937)

---
# 무작정 따라하는 강화학습 프로젝트
---

## 1. Introduction

강화학습은 **환경**(**Environment**)과 **에이전트**(**Agent**)의 상호작용이다.

이번 챕터에서는 이러한 *환경*으로 파이썬의 gym 모듈들 사용하고 *Deep q-learning*(*DQN*)을 학습 알고리즘으로 사용하여, 간단한 모델을 생성하여 학습하고 결과를 보겠다.
하지만, 알고리즘을 이해하기 위해서는 강화학습과 *DQN* 알고리즘을 이해할 필요가 있다.
따라서 강화학습의 기본 개념을 시작으로 *DQN*알고리즘을 알아보고, 배운 개념을 Python 코드로 짜보려고 한다.
마지막으로는 학습의 결과를 확인하고, 배운 개념을 정리하면서 끝맺도록 하겠다.

## 2. 강화학습의 기본 개념

### 1) 수학적 관점의 **MDP**(**Markov Decision Process**)와 강화학습적 관점의 MDP

### 2) 강화학습의 목표와 **Bellman Equation**

#### **Reward Hypothesis**에 의하면 에이전트의 목표는 보상 축적값의 최대화이다.

보상의 최대화가 아닌 보상 축적값의 최대화가 목표인 이유가 무엇인지 생각해 볼 필요가 있다.

> 예를 들어, 기업의 이윤을 최대화하는 에이전트를 생각해보자. 이때 나는 에이전트를 기업의 경영자 A라 가정하겠다.A의 목표는 1년동안의 이윤을 최대화 하기이다. 하지만 A는 여름 계절상품을 기획하였고, 기업의 여름 이윤은 급 상승 하였지만 다른 계절에는 좋은 성과를 만들지 못하였다. 이로 인해, 기업의 1년간 이윤은 평균적으로 낮아졌고 A의 목표를 달성하지 못했다. 여기서 A의 실패 요인은 *근시안적*인 목표 달성에 중점을 두었다는 것이다. 따라서 A가 성공하기 위해서는 최소 1년동안 이윤을 보장하는 기획이 필요하다.

위의 예를 보면 알 수 있듯이, 바로 얻을 수 있는 *보상*(*여름 동안의 기업이윤*)만을 고려하게 되면 *보상의 축적값*(*1년 동안 축적된 기업이윤*)이라는 목표를 달성할 수 없다. 따라서, 에이전트는 *근시안적*인 단일 보상이 아닌, 보상의 축적값을 최대화 할 수 있는 방법을 찾아야한다.

#### 이러한 축적된 보상을 Return이라 하고, 다음과 같이 정의한다.

$ G_t = R_{t+1} + R_{t+2} + R_{t+3} + \cdots + R_T $

* $t$ : *time-step* $t$는 에이전트와 환경의 상호작용을 순서대로 구분할 때 사용하는 인덱스이다.
* $T$ : *Terminal time-step* $T$는 에이전트와 환경의 상호작용이 자연스럽게 종료되는 *time-step* 인덱스이다.
* $R_t$ : *time-step* $t$에서의 **보상**(**Reward**)
* $G_t$ : *time-step* $t$에서부터 $T$까지의 **축적된 보상**(**Return**)

에이전트의 목표인 *Return*의 최대화를 바탕으로, 강화학습의 목표를 다음곽 같이 설명할 수 있다.

#### 에이전트가 *Return*을 최대화 하는 행동을 선택할 때, 강화학습의 목표가 달성된다.

즉, 강화학습은 에이전트가 최적의 행동을 하도록 학습시킨다.

위같은 조건을 만족하기 위해서, 특정 상태의 가치와 그때 선택되는 행동의 가치를 판단할 필요가 있는데, 이를 결정하는 함수를 **Value Function**이라 하고 다음과 같이 정의한다.

**State-value Function** : $ v_\pi(s) = \mathbb{E}_\pi[G_t | S_t = s] = \mathbb{E} \left[ \displaystyle\sum_{k=0}^{\infty} \gamma^k R_{t+k+1} | S_t = s \right]\,\, , \forall x \isin \mathcal{S} $

**Action-value Function** : $ q_\pi(s, a) = \mathbb{E}_\pi[G_t | S_t = s, A_t = a] = \mathbb{E}_\pi \left[ \displaystyle\sum_{k=0}^{\infty} \gamma^k R_{t+k+1} | S_t = s, A_t = a \right]\,\, , \forall s \isin \mathcal{S}, a \isin \mathcal{A} $

* $\pi$ : 에이전트의 *Policy*
* $s$ : 환경에서 얻어진 *상태(State)* $s$
* $\gamma$ : *step-size* $\gamma$는 미래 보상의 반영도를 결정한다. e.g. $\gamma$가 0이면 $\displaystyle\sum_{k=0}^{\infty} \gamma^k R_{t+k+1} = R_{t+1}$이므로 즉각적인 보상만을 고려하게 되고, 1이면 $\displaystyle\sum_{k=0}^{\infty} \gamma^k R_{t+k+1} = \displaystyle\sum_{k=0}^{\infty} R_{t+k+1}$이므로 먼 미래의 보상과 즉각적인 보상을 같은 정도로 고려하게 된다.
* $v_\pi(s)$ : *Policy* $\pi$를 따라갈 때, *State* $s$에서의 가치
* $q_\pi(s, a)$ : *Policy* $\pi$를 따라갈 때, *State* $s$에서 *Action* $a$를 선택하면 얻을 수 있는 가치

## 3. 강화학습 Python 코드 작성

### 1) Conda 가상환경 생성과 라이브러리 다운로드

가상환경을 만드는 이유는 여러가지 이지만, 이 프로젝트에서는 파이썬 패키지의 버전을 관리하기 위해서이다.

> * 예를 들어, 프로젝트 A에서는 python2의 코드를 사용하는데, B에서는 python3을 사용하면 코드가 서로 호환이 안 되므로, 각각의 가상환경을 만들어서 다른 버전의 python을 다운로드하면 된다.

#### 우선 아나콘다를 다음 사이트에서 다운로드한다.
<https://www.anaconda.com/>

> * 만약 conda 명령어가 작동하지 않는다면 환경변수를 설정을 해야한다.

#### conda 가상환경 생성을 위해 다음과 같은 명령어를 실행한다.

```python
# env_name에 쓰고싶은 가상환경 이름을 적는다.
# conda create로 가상환경을 생성한다.
conda create -n env_name python="3.9.7"

# conda activate로 가상환경을 활성화한다.
conda activate env_name

# pip install로 파이썬 패키지를 다운로드한다.
pip install numpy matplotlib matplotlib-inline torch torchvision mlagents mlagents-envs
```

> 제가 사용한 버전은 다음과 같습니다.
>- python - 3.9.7
>- numpy - 1.21.2
>- matplotlib - 3.5.1
>- matplotlib-inline - 0.1.2
>- torch - 1.10.2 (파이토치)
>- torchvision - 0.11.3
>- mlagents - 0.28.0
>- mlagents-envs - 0.28.0

## 4. 강화학습 파이썬 코드 작성

```python

# PyTorch
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

# mlagents_envs는 강화학습 환경으로 Unity를 사용하기 위한 리이브러리이다.
from mlagents_envs.environment import UnityEnvironment, ActionTuple, BaseEnv

from collections import namedtuple, deque
import random
import math
import os
from itertools import count
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, NamedTuple, List

env_name = "RL_car" # 프로젝트의 경로
train_mode = True # training or inference mode 인지를 선택

# cuda와 cudnn을 다운받으면 gpu가 사용 가능하다.
# 하지만, 없어도 cpu로 가능하다.
device = torch.device = ('cuda' if torch.cuda.is_avilable() else 'cpu')
```

## Experience 클래스 정의하기

Deep Q_learning을 사용하는데

```python
class Experience(NamedTuple):
  """
  An experience contains the data of one Agent transition.
  - Observation
  - Action
  - Reward
  - Done flag
  - Next Observation
  """

  obs: np.ndarray
  action: np.ndarray
  reward: float
  done: bool
  next_obs: np.ndarray

# Trajectory는 Experience의 리스트 이다.
Trajectory = List[Experience]

# A Buffer is an unordered list of Experiences from multiple Trajectories
Buffer = List[Experience]
```

## 모델 클래스 생성하기

```python
class model(nn.Module):
    def __init__(self, input_size, output_size):
        super(model, self).__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(input_size, 24),
            nn.ReLU(),
            nn.Linear(24, 12),
            nn.ReLU(),
            nn.Linear(12, output_size)
        )

    def forward(self, x):
        x = self.linear_relu_stack(x)
        return x
```

## 학습자 클래스 정의하기

```python
class Trainer:
    @staticmethod
    def generate_trajectories(
        env: BaseEnv, q_net: model, buffer_size: int, epsilon: float
    ):
        # 빈 버퍼를 만든다.
        buffer: Buffer = []

        # Reset the environment
        env.reset()
        # Read and store the Behavior Name of the Environment
        behavior_name = list(env.behavior_specs)[0]
        # Read and store the Behavior Specs of the Environment
        spec = env.behavior_specs[behavior_name]

        # Create a Mapping from AgentId to Trajectories. This will help us create
        # trajectories for each Agents
        dict_trajectories_from_agent: Dict[int, self.Trajectory] = {}
        # Create a Mapping from AgentId to the last observation of the Agent
        dict_last_obs_from_agent: Dict[int, np.ndarray] = {}
        # Create a Mapping from AgentId to the last observation of the Agent
        dict_last_action_from_agent: Dict[int, np.ndarray] = {}
        # Create a Mapping from AgentId to cumulative reward (Only for reporting)
        dict_cumulative_reward_from_agent: Dict[int, float] = {}
        # Create a list to store the cumulative rewards obtained so far
        cumulative_rewards: List[float] = []

        is_experienced = False
            
        while len(buffer) < buffer_size:  # While not enough data in the buffer
            
            # Get the Decision Steps and Terminal Steps of the Agents
            decision_steps, terminal_steps = env.get_steps(behavior_name)
            
            if is_experienced:
                # For all Agents with a Terminal Step:
                for agent_id_terminated in terminal_steps:

                    # Create its last experience (is last because the Agent terminated)
                    last_experience = Experience(
                        obs=dict_last_obs_from_agent[agent_id_terminated].copy(),
                        reward=terminal_steps[agent_id_terminated].reward,
                        done=not terminal_steps[agent_id_terminated].interrupted,
                        action=dict_last_action_from_agent[agent_id_terminated].copy(),
                        next_obs=terminal_steps[agent_id_terminated].obs[0],
                    )
                    
                    # Report the cumulative reward
                    cumulative_reward = (
                        dict_cumulative_reward_from_agent.pop(agent_id_terminated)
                        + terminal_steps[agent_id_terminated].reward
                    )
                    cumulative_rewards.append(cumulative_reward)
                    # Add the Trajectory and the last experience to the buffer
                    buffer.extend(dict_trajectories_from_agent.pop(agent_id_terminated))
                    buffer.append(last_experience)

            # For all Agents with a Decision Step:
            for agent_id_decisions in decision_steps:
                
                is_experienced = True
                
                # If the Agent does not have a Trajectory, create an empty one
                if agent_id_decisions not in dict_trajectories_from_agent:
                    dict_trajectories_from_agent[agent_id_decisions] = []
                    dict_cumulative_reward_from_agent[agent_id_decisions] = 0

                # If the Agent requesting a decision has a "last observation"
                if agent_id_decisions in dict_last_obs_from_agent:
                    # Create an Experience from the last observation and the Decision Step
                    exp = Experience(
                        obs=dict_last_obs_from_agent[agent_id_decisions].copy(),
                        reward=decision_steps[agent_id_decisions].reward,
                        done=False,
                        action=dict_last_action_from_agent[agent_id_decisions].copy(),
                        next_obs=decision_steps[agent_id_decisions].obs[0],
                    )
                    # Update the Trajectory of the Agent and its cumulative reward
                    dict_trajectories_from_agent[agent_id_decisions].append(exp)
                    dict_cumulative_reward_from_agent[agent_id_decisions] += (
                        decision_steps[agent_id_decisions].reward
                    )
                # Store the observation as the new "last observation"

                dict_last_obs_from_agent[agent_id_decisions] = (
                    decision_steps[agent_id_decisions].obs[0]
                )
        
            # Generate an action for all the Agents that requested a decision
            # Compute the values for each action given the observation
            actions_values = (
                q_net(torch.from_numpy(decision_steps.obs[0])).detach().numpy()
            )
            
            if random.uniform(0, 1) <= epsilon:
                actions = np.random.randint(3, size=(decision_steps.agent_id.shape[0], 1))
            else:
                actions = np.argmax(actions_values, axis=1)
                actions.resize((len(decision_steps), 1))
            
            # Store the action that was picked, it will be put in the trajectory later
            for agent_index, agent_id in enumerate(decision_steps.agent_id):
                dict_last_action_from_agent[agent_id] = actions[agent_index]

            # Set the actions in the environment
            # Unity Environments expect ActionTuple instances.
            action_tuple = ActionTuple()
            action_tuple.add_discrete(actions)
            
            # Perform a step in the simulation
            env.set_actions(behavior_name, action_tuple)
            env.step()
        return buffer, np.mean(cumulative_rewards)

    @staticmethod
    def update_q_net(
        q_net: model,
        optimizer: torch.optim,
        buffer: Buffer,
        action_size: int
    ):
        """
        Performs an update of the Q-Network using the provided optimizer and buffer
        """
        BATCH_SIZE = 1000
        NUM_EPOCH = 3
        GAMMA = 0.9
        batch_size = min(len(buffer), BATCH_SIZE)
        random.shuffle(buffer)
        # Split the buffer into batches
        batches = [
            buffer[batch_size * start : batch_size * (start + 1)]
            for start in range(int(len(buffer) / batch_size))
        ]
        
        for _ in range(NUM_EPOCH):
            for batch in batches:
                # Create the Tensors that will be fed in the network
                obs = torch.from_numpy(np.stack([ex.obs for ex in batch]))
                reward = torch.from_numpy(
                    np.array([ex.reward for ex in batch], dtype=np.float32).reshape(-1, 1)
                )
                done = torch.from_numpy(
                    np.array([ex.done for ex in batch], dtype=np.float32).reshape(-1, 1)
                )
                action = torch.from_numpy(np.stack([ex.action for ex in batch]))
                next_obs = torch.from_numpy(np.stack([ex.next_obs for ex in batch]))
                
                # Use the Bellman equation to update the Q-Network
                target = (
                    reward
                    + (1.0 - done)
                    * GAMMA
                    * torch.max(q_net(next_obs).detach(), dim=1, keepdim=True).values
                )

                mask = torch.zeros((len(batch), action_size))
                mask.scatter_(1, action, 1)
                prediction = torch.sum(q_net(obs) * mask, dim=1, keepdim=True)
                criterion = torch.nn.SmoothL1Loss()
                loss = criterion(prediction, target)

                # Perform the backpropagation
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
```

## 학습하기

```python
```

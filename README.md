# project_01_first

파이썬 신경진화 프레임워크입니다. 완전연결 신경망의 가중치와 편향을 선택, 교잡, 변이로 진화시킵니다. Python 3.10 이상에서 표준 라이브러리만 사용합니다.

## 프로젝트 구조

| 경로 | 내용 |
| --- | --- |
| `neuroevolution/` | 공용 신경망 실행, 설정 검증, 세대 진행, 교잡, 변이, 진행 로그, JSON 저장 |
| `projects/xor/` | XOR 실행, 설정·평가 함수(`training.py`), 학습 모델 두 개(`models/`) |
| `projects/gomoku/` | 오목 규칙(`rules.py`), 착수 정책(`policy.py`), 설정·대국 평가(`training.py`), 브라우저 대국(`web.py`, `board.html`), 규칙별 1위 모델(`models/`) |

## Codespaces에서 오목 대국

저장소 루트에서 실행합니다.

```bash
python3 -m projects.gomoku play
```

Codespaces에서 **Ports → 8000 → Open in Browser**를 누릅니다. 목록에 8000이 없다면 **Forward a Port**로 8000을 추가합니다. 로컬에서는 `http://localhost:8000`을 엽니다. 규칙과 자신의 돌 색을 고르고 **새 대국**을 누르면 저장된 모델과 둘 수 있습니다. 방향키로 교차점을 이동하고 Enter로 둘 수도 있습니다.

- **기본 오목 (`freestyle`)**에서는 양쪽 모두 연속 5개 이상이면 승리합니다. 첫 수는 빈 교차점 어디든 둘 수 있습니다.
- **렌주 (`renju`)**에서는 흑이 중앙에서 시작하며 정확히 5개면 승리합니다. 흑의 장목, 4-4, 실제 열린 3-3은 백 승리로 처리합니다. 같은 착수에서 흑이 정확한 5도 만들면 흑이 승리합니다. 백은 연속 5개 이상이면 승리합니다. 금수와 열린 3 판정은 [RIF 규칙 3절과 9절](https://www.renju.net/rifrules/)에 따릅니다.
- 두 규칙 모두 15×15에서 번갈아 착수하며, 판이 가득 찰 때까지 승자가 없으면 무승부입니다. 렌주 대회용 오프닝 교환, 5수 제안, 패스, 시계는 적용하지 않습니다.

## 오목 학습

```bash
python3 -m projects.gomoku train --rules renju --generations 12 --population 12 --model renju-local.json
python3 -m projects.gomoku play --model renju-local.json
```

`--rules freestyle`로 기본 오목 개체를 학습할 수 있습니다. 학습할 때 `--model`을 생략하면 해당 규칙의 기본 모델 파일을 갱신합니다. 대국에서 `--model`을 생략하면 선택한 규칙에 맞는 기본 모델을 자동으로 사용합니다. 사용자 모델을 지정하면 규칙을 바꿔도 지정한 모델을 계속 사용합니다.

- 세대마다 모든 개체 쌍이 같은 초기 3수 배치를 두고 흑과 백을 바꿔 두 번 대국합니다. 승리는 1점, 무승부는 0.5점, 패배는 0점입니다. 경기당 평균 점수로 개체 순위를 정합니다.
- 12개체라면 세대마다 132경기를 진행합니다. 진행률은 완료 경기 수를 기준으로 표시합니다. 선택, 교잡과 변이는 공용 프레임워크에서 처리합니다.
- 상대가 세대마다 바뀌므로 최종 세대의 1위를 저장합니다. 함께 제공하는 두 모델은 각각 12세대와 12개체, 시드 42로 학습했습니다.
- 착수 정책은 주변 후보의 연속 돌 수, 열린 끝, 중앙 거리, 진행 단계에서 뽑은 18개 입력으로 신경망의 착수 점수를 계산합니다. 즉시 이기는 수와 상대의 즉시 승리 방어를 우선하고, 흑 금수는 후보에서 제외합니다. 오목판 전체를 탐색하는 수읽기는 포함하지 않습니다.

## XOR 실행

```bash
# 동봉된 저장 개체로 예측
python3 -m projects.xor --predict 0 1

# 별도 파일에 새로 학습하고 예측
python3 -m projects.xor --quiet --model xor-local.json
python3 -m projects.xor --model xor-local.json --predict 0 1
```

기존 `python3 -m neuroevolution` 명령도 XOR 예제를 실행하며 기본 모델 경로는 `best_individual.json`입니다.

## 내 문제에 적용하기

[`projects/xor/training.py`](projects/xor/training.py)의 설정과 점수 함수를 수정하거나, 아래처럼 다른 프로그램에서 가져다 쓸 수 있습니다.

```python
from neuroevolution import Individual, configure, evolve, run_saved


def score(individual: Individual) -> float:
    examples = [((0, 0), 0), ((0, 1), 1), ((1, 0), 1), ((1, 1), 0)]
    errors = [(individual(inputs)[0] - target) ** 2 for inputs, target in examples]
    return -sum(errors) / len(errors)


config = configure(
    input_size=2,
    hidden_layers=(4,),
    output_size=1,
    generations=160,
    population_size=100,
    parent_count=20,
    survivor_count=5,
    crossover="uniform",
    mutation_rate=0.2,
    mutation_range=(-0.8, 0.8),
    seed=42,
    debug=True,
)
result = evolve(config, score, save_path="best_individual.json")
print(result.score, result.generation)
print(run_saved("best_individual.json", (0, 1)))
```

점수 함수는 개체를 받아 필요한 입력으로 여러 번 실행하고, 출력들을 유한한 실수 점수 하나로 합칩니다. 점수가 높을수록 우선 선택됩니다. `load_individual(path)`로 개체를 한 번 읽어 여러 입력에 재사용할 수 있습니다. `save_path=None`이면 파일 저장을 생략합니다.

| 설정 | 동작 |
| --- | --- |
| `input_size`, `output_size` | 입력과 출력 뉴런 수 |
| `hidden_layers` | 은닉층별 뉴런 수. `(8, 4)`는 은닉층 두 개, `()`는 은닉층 없음 |
| `generations`, `population_size` | 평가할 세대 수와 세대당 개체 수 |
| `parent_count` | 교잡 부모 후보에 포함할 상위 개체 수. 자식마다 후보 중 서로 다른 두 개체를 무작위 선택 |
| `survivor_count` | 변이 없이 다음 세대로 넘길 상위 개체 수. 0이면 전원 교체 |
| `crossover` | `uniform`은 유전자마다 부모 선택, `single_point`는 한 지점에서 양쪽 부모 유전자 연결, `average`는 유전자별 평균 |
| `mutation_rate`, `mutation_range` | 자식의 각 유전자에 변이가 일어날 확률과 기존 값에 더할 균등 난수 범위 |
| `seed` | 프레임워크 난수 시드. 점수 함수도 결정적이면 같은 설정으로 결과 재현 가능 |
| `debug` | 표준 오류에 평가 진행률을 `\r`로 갱신하고 세대마다 최고·평균·최저·전체 최고 점수를 한 줄로 기록 |

한 번의 실행에서 모든 개체는 같은 층 구조를 사용합니다. 은닉층의 활성화 함수는 `tanh`이고 출력층은 선형입니다.


개체끼리 겨루는 프로젝트는 `evolve(config, population_score=evaluator)`를 사용합니다. 평가 함수는 `(population, report)`를 받아 원래 개체 순서대로 점수 하나씩을 반환합니다. `report(완료 경기 수, 전체 경기 수)`를 호출하면 진행률이 갱신됩니다.

회귀 테스트는 `python3 -m unittest discover -s tests`로 실행합니다.

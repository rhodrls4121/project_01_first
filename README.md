# project_01_first

파이썬 신경진화 프레임워크입니다. 완전연결 신경망의 가중치와 편향을 선택, 교잡, 변이로 진화시킵니다. Python 3.10 이상에서 표준 라이브러리만 사용합니다.

## 실행

저장소 루트에서 실행합니다.

```bash
python -m neuroevolution
python -m neuroevolution --predict 0 1
```

첫 명령은 XOR 예제를 실행하고 전체 세대에서 가장 점수가 높은 개체를 `best_individual.json`에 저장합니다. 두 번째 명령은 해당 개체를 불러와 입력 `(0, 1)`의 출력을 보여줍니다. `--model PATH`로 저장·불러오기 경로를 지정하고 `--quiet`으로 진행률을 끌 수 있습니다.

## 내 문제에 적용하기

[`neuroevolution/__main__.py`](neuroevolution/__main__.py)의 설정과 점수 함수를 수정하거나, 아래처럼 다른 프로그램에서 가져다 쓸 수 있습니다.

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

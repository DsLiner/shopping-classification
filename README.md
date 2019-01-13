# shopping-classification

팀 송골매의 `쇼핑몰 상품 카테고리 분류` 대회 참가용 코드입니다. 대회의 원본 코드(https://github.com/kakao-arena/shopping-classification)를 기반으로 작성하였습니다. (코드는 python2.7, keras, tensorflow, konlpy 기준으로 작성되었습니다.)

## 참가자

  | Username | Email                |
  | -------- | -------------------- |
  | k.dh     | fhggty2010@gmail.com |
  | diehoho2 | diehoho@naver.com    |
  | dsliner  | rgdkdlel4@naver.com  |

## 알고리즘

- 카테고리를 계층 구분없이 "대>중>소>세"로 표현하여 데이터를 구성했습니다. 그 뒤에 간단한 선형 모델로 네트워크를 구성했는데, 텍스트 데이터를 정규화한 후 단어 빈도가 높은 순서로 N개의 워드와 그에 대한 빈도를 입력으로 받습니다. 워드는 임베딩되고, 빈도는 가중치로 작용하게 됩니다.
- 입베딩한 노드와 데이터로부터 읽어온 `image_feature`을 `concat`한 후 한층의 히든 레이어를 이용해 결과를 예측합니다.
- 본 코드는 제공된 데이터를 `대중`, `소`, `세` 분류에 대해 별도의 학습을 진행합니다.
- 학습을 위한 네트워크 모델의 기본적인 구성은 동일합니다.
- `대중`, `소`, `세`는 동일한 기본모델에서 파라미터를 다르게 설정하여 훈련을 진행합니다.
- 훈련을 진행할 때 사용한 파라미터 설정은 `config.json`파일에 `대중(bm)`, `소(s)`, `세(d)`로 구분되어 있습니다.

## 실행 방법

1. 데이터의 위치
    - 내려받은 데이터의 위치는 `data` 폴더에 위치해야합니다.
2. `train` 데이터셋 생성
    - 아래의 명령어들은 `src` 폴더에서 수행하여야 합니다.
    1. `python data.py make_db train bm`
    2. `python data.py make_db train s`
    3. `python data.py make_db train d`
    - 위의 명령어들을 이용해 학습에 필도한 데이터 셋을 생성합니다.
    - 위 명렁어를 수행하면 `train` 데이터는 100%의 학습 데이터로 구성됩니다.
        - 학습 데이터의 비율을 조정하려면 `python data.py make_db train bm 0.8`과 같이 명령어를 수행하면 학습 80%, 평가 20%의 비율로 데이터가 나뉩니다.
    - 이 명령어를 실행하기 전에 `python data.py build_y_vocab`으로 데이터 생성이 필요한데, 코드 레파지토리에 생성한 파일이 포함되어 다시 만들지 않아도 됩니다. 
      - Python 2는 `y_vocab.cPickle` 파일을 사용하고, Python 3은 `y_vocab.py3.cPickle` 파일을 사용합니다.
    - `config.json` 파일에 동시에 처리할 프로세스 수를 `num_workers`로 조절할 수 있습니다.
3. `train` 데이터셋 학습
    - `./data/train`에 생성한 데이터셋으로 학습을 진행합니다.
    - 아래의 명령어들은 `src` 폴더에서 수행하여야 합니다.
    1. `python classifier.py train ../data/train/bm ../data/model/bm bm false`
    2. `python classifier.py train ../data/train/s ../data/model/s s false`
    3. `python classifier.py train ../data/train/d ../data/model/d d false`
    - 위의 명령어들을 수행하면 생성된 데이터셋에 대해 학습을 진행합니다.
    - 완성된 모델은 각각 아래의 폴더에 위치합니다.
    1. `./model/bm`
    2. `./model/s`
    3. `./model/d`
4. `dev` 데이터셋 생성
    1. `python data.py make_db dev --train_ratio=0.0 dm`
    2. `python data.py make_db dev --train_ratio=0.0 s`
    3. `python data.py make_db dev --train_ratio=0.0 d`
5. `dev` 데이터셋에 대한 예측
    1. `python classifier.py predict ../data/train/bm ../data/model/bm ../data/dev/bm dev bmcate.tsv bm`
    2. `python classifier.py predict ../data/train/s ../data/model/s ../data/dev/s dev scate.tsv s`
    3. `python classifier.py predict ../data/train/d ../data/model/d ../data/dev/d dev dcate.tsv d`
6. `test` 데이터셋 생성
    1. `python data.py make_db test --train_ratio=0.0 bm`
    2. `python data.py make_db test --train_ratio=0.0 s`
    3. `python data.py make_db test --train_ratio=0.0 d`
7. `test` 데이터셋에 대한 예측
    1. `python classifier.py predict ../data/train/bm ../data/model/bm ../data/test/bm test bmcate.tsv bm`
    2. `python classifier.py predict ../data/train/s ../data/model/s ../data/test/s test scate.tsv s`
    3. `python classifier.py predict ../data/train/d ../data/model/d ../data/test/d test dcate.tsv d`
8. 완성된 예측파일 합치기
    - `python sum_tsv.py`
    - 위의 명령어를 이용해 생성된 `bmcate.tsv`, `scate.tsv`, `dcate.tsv` 파일을 합쳐서 최종 예측파일인 `output.tsv` 파일을 생성할 수 있습니다.
    - 이 명령어는 `dev`와 `test`에 대한 구분이 없으므로 `dev`와 `test` 데이터셋에 대한 예측이 끝날경우 각각 따로 실행해 주어야 합니다.

## model 파일
- 생성된 모델의 사이즈는 `949MB`입니다.
- 모델은 아래의 url에서 다운로드 받으실 수 있습니다.
    - https://1drv.ms/f/s!Aie6oAOsjTGjipkGyowLhIBJvP_aLQ

## 테스트 가이드라인
학습데이터의 크기가 100GB 이상이므로 사용하는 장비에 따라서 설정 변경이 필요합니다. `config.json`에서 수정 가능한 설정 중에서 아래 항목들이 장비의 사양에 민감하게 영향을 받습니다.

    - train_data_list
    - chunk_size
    - num_workers
    - num_predict_workers


`train_data_list`는 학습에 사용할 데이터 목록입니다. 전체 9개의 파일이며, 만약 9개의 파일을 모두 사용하여 학습하기 어려운 경우는 이 파일 수를 줄일 경우 시간을 상당히 단축시킬 수 있습니다. 

`chunk_size`는 전처리 단계에서 저장하는 중간 파일의 사이즈에 영향을 줍니다. Out of Memory와 같은 에러가 날 경우 이 옵션을 줄일 경우 해소될 수 있습니다.

`num_workers`는 전처리 수행 시간과 관련이 있습니다. 장비의 코어수에 적합하게 수정하면 수행시간을 줄이는데 도움이 됩니다.

`num_predict_workers`는 예측 수행 시간과 관련이 있습니다. `num_workers`와 마찬가지로 장비의 코어수에 맞춰 적절히 수정하면 수행시간을 단축하는데 도움이 됩니다.

## 라이선스

This software is licensed under the Apache 2 license, quoted below.

Copyright 2018 Kakao Corp. http://www.kakaocorp.com

Licensed under the Apache License, Version 2.0 (the “License”); you may not use this project except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0.

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an “AS IS” BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.

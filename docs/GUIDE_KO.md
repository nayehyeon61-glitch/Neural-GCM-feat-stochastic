# NeuralGCM에 공간 통계 memory를 연결한 첫 구현

기존 NeuralGCM의 physics dynamics와 learned physics를 유지하고, 그 예측
상태에 **공간 통계로 초기화한 memory와 확률적 온도 보정항**을 연결했습니다.
NeuralGCM 자체에도 확률적 모델이 있으므로, 이번 연구의 가설은 noise 추가
자체보다 **통계적 과거 정보가 이후의 물리 예측에 유용한가**입니다.

현재 코드는 구현·수치 검증 단계입니다. 실제 학습된 새 모델이나 장기 성능
향상 결과를 제공하는 단계는 아닙니다.

## 구조

```mermaid
flowchart TD
  O["예측 시작 전 관측장"] --> E["기존 spatial encoder: JAX 변환"]
  E --> M["여러 시간척도의 memory 초기화"]
  X["현재 NeuralGCM 상태"] --> B["기존 dynamics + learned physics"]
  X --> M
  M --> C["memory 보정 + OU 확률 보정"]
  B --> Y["다음 대기 상태: 온도 보정 적용"]
  C --> Y
  Y --> X
```

- 원래 적분기를 전부 Euler–Maruyama로 바꾸지 않습니다. 기존 semi-implicit
  dynamics solver 뒤에 보정항을 결합하는 operator splitting입니다.
- memory는 단순한 현재 latent가 아니라, 상태로 유지되는 지수 기억항입니다.
- 확률 부분은 보조 OU 상태의 SDE이며, 그 시간적분을 온도에 가합니다.
  시간 간격마다 독립 noise를 무작정 더하는 구조가 아닙니다.
- 첫 보정 대상은 `temperature_variation`입니다. 압력·수분·바람에 직접적인
  새 보정항을 넣은 것은 아닙니다. 이후 물리 적분에서는 온도 변화에 반응합니다.

## 기존 manifold를 실제로 쓰는 방식

최신 확인 branch의 spatial encoder를 JAX로 옮겼고, 기존 PyTorch 구현과
일치하는지 검사했습니다. 기존 `msl,t2m,u10,v10`과 선택적 pressure/terrain
정보, 정규화, 변수 순서, pooling, fusion을 checkpoint 계약으로 보존합니다.

NeuralGCM sigma 층의 온도·바람을 `t2m,u10,v10`이라고 취급하지 않습니다.
대신 기존 encoder로 **시작 시점 이전 관측장**을 encode하여 memory를
초기화합니다. 예측 중에는 현재 NeuralGCM 온도 상태가 memory를 갱신합니다.

따라서 현재 버전은 **초기 조건에 과거 공간 통계를 반영하는 모델**입니다.
예측 내내 기존 surface encoder를 재호출하는 모델은 아닙니다. 원래 공간
통계의 직접적인 영향은 설정한 시간척도에 따라 감소합니다. 기본 최장
시간척도는 120시간이고, 더 긴 기억은 별도 비교 실험으로 검증해야 합니다.

기존 저장소에서 학습된 `.pt` 가중치는 찾지 못했습니다. 아래 연결 단계에는
사용자가 실제로 학습한 spatial checkpoint와 `.manifest.json`이 필요합니다.
코드만으로 학습된 통계가 자동 생성되지는 않습니다.

## 1. 설치와 코드 검사

Terra에서는 프로젝트와 결과를 `/lustre/home/yehyeon` 아래에 둡니다.
원본 ERA5 디렉터리는 읽기만 하며 데이터를 이동하지 않습니다.

```bash
cd /lustre/home/yehyeon
git clone --recurse-submodules https://github.com/nayehyeon61-glitch/Neural-GCM-feat-stochastic.git
cd Neural-GCM-feat-stochastic
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[test,export,netcdf]'
python -m pytest -q
python scripts/smoke_neuralgcm.py
```

마지막 명령은 공식 패키지의 작은 **toy checkpoint**를 사용합니다. 실제
NeuralGCM API 연결과 gradient 검사용이며 실제 기상 성능 검증은 아닙니다.
GPU 학습에는 서버 환경에 맞는 JAX GPU 설치가 별도로 필요합니다.

## 2. 실제 NeuralGCM 입력 준비

공식 pretrained checkpoint를 `checkpoints/base.pkl`에 준비합니다.
[공식 checkpoint 목록](https://neuralgcm.readthedocs.io/en/stable/checkpoints.html)을
참고합니다. 이 코드가 checkpoint를 자동 다운로드하지는 않습니다.

아래 경로는 예시이며 실제 파일명으로 바꿉니다.

```bash
python scripts/prepare_neuralgcm_data.py \
  --base-checkpoint checkpoints/base.pkl --trust-base-checkpoint \
  --input '/lustre/home/mahmed/ERA5_0p25_DAILY/실제_선택한_파일.nc' \
  --forcings-file data/sst_seaice_same_times.nc \
  --time-start 2010-01-01 --time-end 2010-01-31 \
  --output data/prepared_2010_01.nc
```

`--input`은 반복할 수 있고 따옴표로 감싼 glob도 받습니다. SST/해빙이 이미
입력 파일에 있으면 `--forcings-file`은 생략합니다. SST/해빙이 없으면 실제
자료를 추가해야 하며, 임의 상수로 만들어 채우지 않습니다. checkpoint에
따라 cloud ice/liquid 변수도 필요합니다.

pressure level은 checkpoint가 요구하는 값이 모두 있어야 합니다. 수평은
공식 conservative regrid를 사용하지만 누락된 수직 층을 합성하지 않습니다.
기존 SST/해빙의 육지 마스크만 수평 nearest fill하려면
`--fill-forcing-nans`를 명시합니다. 온도 등 대기장 NaN은 거부합니다.

선택한 기간은 메모리에 적재하므로 고해상도 전체 연도를 한 번에 넣지 말고
작은 기간부터 확인합니다. 현재 학습 스크립트도 준비된 저해상도 NetCDF를
메모리에 적재하는 초기 구현입니다. 다년간 전역 자료의 분산 학습기는 아닙니다.

## 3. 기존 spatial checkpoint와 과거 통계 준비

```bash
python scripts/export_spatial.py \
  --checkpoint checkpoints/source_spatial.pt \
  --output checkpoints/spatial_encoder.npz

python scripts/prepare_spatial_history.py \
  --encoder checkpoints/spatial_encoder.npz \
  --data data/source_surface_history.nc \
  --latitude-name lat --longitude-name lon \
  --max-history-gap-hours 24 \
  --output data/spatial_history.npz
```

첫 명령에는 source에서 생성한 `source_spatial.manifest.json`도 필요합니다.
본인이 만든 기존 NumPy metadata 포함 checkpoint에서 안전 로딩이 거부되는
경우에만 `--trusted-pickle`을 사용합니다.

두 번째 입력은 **source encoder가 학습한 격자·변수명·단위**와 일치해야
합니다. 예를 들어 source의 z500은 m 단위 geopotential height일 수 있고,
NeuralGCM 입력 z는 m²/s² 단위 geopotential입니다. 자동으로 같은 변수라고
간주하지 않습니다. source encoder와 정규화는 반드시 training 기간 자료로
학습한 것을 사용합니다.

history 파일에 미래 자료가 포함되어 있어도 각 예측은 자신의 시작 시점보다
엄격히 이전인 자료만 선택합니다. `--through YYYY-MM-DD`로 파일 준비
자체의 종료 시점도 제한할 수 있습니다.

위 `24`는 일별 자료 예시입니다. 실제 관측 주기에 맞춰 **허용할 최대 공백**을
지정합니다. 각 선택된 과거 표본 사이뿐 아니라 마지막 표본부터 예측 시작까지의
간격도 이 값을 넘으면 해당 window를 제외하고 이유를 출력합니다. 학습·검증에
유효한 window가 남지 않으면 오류로 종료합니다. 불규칙 자료도 명시한 범위
안에서는 사용할 수 있습니다. 옵션이 없으면 시간 간격의 중앙값을 사용하지만,
자료 자체가 성기게 누락된 경우 이를 알아낼 수 없으므로 실제 실험에는 명시값을
권장합니다. 단일 표본은 명시값이 필요하며, 평가에도 학습과 같은 최대 공백을
사용해야 합니다. 오래된 마지막 표본을 임의로 예측 시점까지 늘리지 않습니다.

## 4. 학습

우선 native temperature memory만 검사하려면 `--spatial-history`를 생략합니다.
아래는 기존 spatial 통계를 연결하는 명령입니다.

```bash
python scripts/train.py \
  --base-checkpoint checkpoints/base.pkl --trust-base-checkpoint \
  --data data/train_validation_prepared.nc \
  --spatial-history data/spatial_history.npz --spatial-history-steps 8 \
  --output runs/full/closure.npz \
  --warmup-steps 2 --horizon-steps 4 --ensemble-size 4 \
  --epochs 5 --train-fraction 0.8 --seed 0 \
  --tau-hours 6,24,120 --stochastic-tau-hours 24 \
  --variables temperature --scales 10
```

`steps`는 관측 자료 간격 기준입니다. daily 자료의 `--horizon-steps 4`는
4일이고, 내부에서는 NeuralGCM timestep에 맞춰 더 촘촘하게 적분합니다.
처음에는 `--max-train-windows 1 --max-validation-windows 1 --epochs 1`로
연결과 메모리 사용량부터 점검할 수 있습니다.

학습 대상은 추가 closure 파라미터입니다. NeuralGCM 파라미터와 원본 spatial
encoder는 고정됩니다. 정답 미래장은 loss에만 사용하며 모델 입력으로
재주입하지 않습니다. 최적 checkpoint는 validation loss로 선택합니다.

| 비교 | 별도 학습에서 사용할 옵션 |
|---|---|
| 원래 NeuralGCM | 평가 때 `--baseline` |
| memory 없이 보정·확률항 | `--no-memory` |
| 확률항 없이 결정론적 memory | `--no-stochastic` |
| 현재 상태의 직접 보정 없이 memory·확률항 | `--no-spatial` |
| 기존 spatial history 없이 native memory | `--spatial-history` 생략 |
| 전체 구조 | 위 명령 |

`--no-spatial`은 **현재 상태의 순간 readout**을 끕니다. source spatial
history를 끄는 옵션이라는 뜻은 아닙니다. 각 구조는 별도로 학습해야 하며,
source 유무 비교에서는 파라미터 수와 초기화 차이도 함께 보고해야 합니다.

## 5. 독립 평가와 원래 모델 비교

학습 및 checkpoint 선택에 사용한 validation 기간보다 뒤의 별도 test 자료를
준비합니다. 코드는 validation까지의 날짜를 제외합니다.

```bash
python scripts/evaluate.py \
  --base-checkpoint checkpoints/base.pkl --trust-base-checkpoint \
  --closure runs/full/closure.npz --data data/test_prepared.nc \
  --spatial-history data/spatial_history_test.npz \
  --horizon-steps 30 --ensemble-size 8 --seed 1000 \
  --output runs/full/test_metrics.json

python scripts/evaluate.py \
  --base-checkpoint checkpoints/base.pkl --trust-base-checkpoint \
  --closure runs/full/closure.npz --data data/test_prepared.nc \
  --spatial-history data/spatial_history_test.npz \
  --horizon-steps 30 --ensemble-size 8 --seed 1000 --baseline \
  --output runs/full/base_metrics.json
```

test history도 같은 source encoder로 만들어야 합니다. native-only 학습을
했다면 두 평가 명령에서 `--spatial-history`를 생략합니다. 초기 시점·seed·
forcing 정책을 맞춰 비교합니다. 기본 forcing은 시작 시점 SST/해빙의
persistence입니다. `--forcing-mode observed`는 미래의 실제 경계조건을
사용하는 별도의 hindcast 설정입니다.

출력에는 lead별 CRPS, ensemble-mean RMSE, spread, bias, 전역 평균 변화와
관측 대비 drift error가 들어갑니다. 이는 장기 기후 검증의 첫 부분이며,
계절 통계·스펙트럼·에너지 및 수분 수지까지 검증한 것은 아닙니다.

## 해석에서 지켜야 할 범위

Hasselmann 논문은 빠른 날씨의 확률적 영향과 느린 응답, 안정화 feedback의
중요성을 뒷받침합니다. 이번 memory 구조가 그 논문에서 자동으로 도출되거나
장기 성능을 보장하는 것은 아닙니다. 또한 NeuralGCM은 prescribed SST에서
다년 기후 통계를 재현한 결과가 있으므로, 문제를 “장기 예측 전체가 안 된다”로
두기보다 **어떤 기간·변수·forcing에서 어떤 drift/분산 오류를 줄일 것인지**로
좁혀 실험하는 것이 맞습니다.

현재 온도 보정은 moist-energy 보존을 강제하지 않습니다. base가 저장한
진단량도 보정 후 다시 계산하지 않으므로, 그것을 새 모델의 강수·에너지 수지
검증 결과로 사용하지 않습니다. 이 제약을 해결하는 깊은 parameterization
통합은 첫 ablation 결과를 확인한 뒤 진행할 다음 구현 단계입니다.

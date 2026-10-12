# Manifold와 memory 구현 점검 — 2026-10-12

> 이 문서는 `d2c2ab8`의 **초기화 전용 버전**을 점검한 기록이다. 이후의
> recurrent latent 수정은 [RECURRENT_LATENT_KO.md](RECURRENT_LATENT_KO.md)를
> 참고한다. 새 버전은 아래의 입력 0 전환·memory 대체 한계를 수정했으며,
> 이 기록은 `--spatial-mode initialization` 비교 모델의 설명으로 남긴다.

현재 구현은 **기존 spatial encoder로 과거 관측의 공간 정보를 추출해 memory를
초기화하고, NeuralGCM 예측 중 온도 보정항으로 사용하는 연구용 baseline**이다.
실제 장기 예측 향상을 확인한 학습 모델은 아직 아니다.

## 1. 기존 manifold에서 보존한 부분과 달라진 역할

| 구성 | 현재 처리 |
|---|---|
| Spatial encoder `E_s` | 원본 구조를 JAX로 이식. 지리적 padding, pooling, fusion, 선택적 정보 encoder, 정규화를 보존하고 추론 가중치는 고정 |
| 공간 latent | 전역 단일 벡터로 소거하지 않고 `[channel, longitude, latitude]` 공간장으로 사용. NeuralGCM 격자로 보간 |
| Dynamics encoder와 원본 예측 decoder | 새 예측 경로에는 사용하지 않음. 실제 NeuralGCM의 dynamics와 learned physics가 상태를 전진시킴 |
| 원본 W2/KL/signed 등의 학습 목적 | 원본 encoder를 학습한 효과만 간접 전달. 새 closure의 loss로 재사용하지 않음 |
| 새로 학습하는 부분 | 입력 변환, 순간 온도 보정, memory readout, 확률항 진폭. NeuralGCM과 source encoder는 고정 |

`msl,t2m,u10,v10` 같은 surface 진단량을 NeuralGCM의 최하층 값과 같다고
가정하지 않는다. 이 때문에 첫 연결은 예측 전 실제 관측을 사용하는 방식이다.
`--spatial-history`가 없으면 native-temperature memory만 쓰며, 사용자의
spatial manifold를 사용하는 실험이 아니다. 실제 학습된 source checkpoint는
이 저장소에 포함되어 있지 않다.

## 2. Memory가 만들어지는 정확한 경로

예측 시작 시각을 0, 그 이전 관측 시각을 \(t_j<0\)라 하자.

\[
z_j=E_s(s_j),\qquad
u_j=\tanh(W[0,z_j]+b).
\]

각 과거 latent를 다음 관측 시각까지 유지해 아래 memory 식을 적분하고,
마지막 구간은 예측 시작 시각에서 끝낸다. 예측 시작 시각의 관측 latent도
사용하지 않는 엄격한 과거 선택이다. 이렇게 얻은 \(m_r(0)\)가 초기 memory다.

예측이 시작되면 입력은 다음과 같이 바뀐다.

\[
u_t=\tanh(W[T'_t/10\mathrm{K},0]+b),\qquad
\frac{dm_r}{dt}=\frac{u_t-m_r}{\tau_r}.
\]

현재 기본 시간척도는 6, 24, 120시간이다. 고정된 양의 설정값이며 학습되는
시간척도는 아니다. 한 스텝 동안 입력을 고정하면 다음 갱신이 정확하다.

\[
m_r(t+h)=a_r m_r(t)+(1-a_r)u_t,\qquad a_r=e^{-h/\tau_r}.
\]

즉 memory는 격자별로 여러 속도의 지수 이동평균을 유지한다. 과거장을 전부
저장하거나 기상계를 따라 memory를 이류시키는 구조는 아니다.

현재 source 초기화는 이전의 thermal-only warmup memory를 **대체**한다.
두 종류 memory가 독립적으로 공존하는 구조는 아니다. base 상태와 기존 OU
상태는 유지된다. 예측 도중 source encoder는 재호출되지 않고 source 입력
채널은 0이다. 이 입력 전환은 의도한 첫 설계지만 개선할 중요한 부분이다.

## 3. Memory가 실제 예측을 보정하는 방식

Memory를 단순히 출력 후처리로 붙이는 것이 아니라 매 내부 적분 스텝의
대기 온도 상태에 반영한다. 시간은 시간(hour), 보정량은 K 단위다.

\[
\Delta T=\int_t^{t+h}\left(A_su+\sum_r A_rm_r+Z/\tau_Z\right)ds,
\qquad dZ=-Zdt/\tau_Z+\sigma(u)dW_t.
\]

1. 현재 상태에서 feature를 구한다.
2. 기존 NeuralGCM의 `advance`로 물리·학습 parameterization을 전진시킨다.
3. memory와 OU 보조 상태를 적분해 \(\Delta T\)를 구한다.
4. 단위 변환과 spectral 변환을 거쳐 `temperature_variation`에 더한다.
5. 다음 NeuralGCM 스텝은 보정된 온도에서 출발한다.

같은 현재 온도라도 직전에 지속적으로 따뜻했던 경우와 차가웠던 경우에는
memory가 달라 보정량이 다를 수 있다. 학습은 그 차이가 이후 관측 오차를
줄이도록 readout을 조정한다. 양의 memory가 반드시 가열을 뜻하지는 않는다.
보정의 부호와 크기는 학습되는 readout에 달려 있다. 보정된 온도는 다음
물리 적분에서 순환·수분 등에 간접 영향을 준다.

OU 항은 시간적으로 상관된 확률 보정이다. endpoint와 시간적분의 공분산까지
맞춰 샘플링한다. **기존 ODE 적분기를 전부 SDE solver로 교체한 것이 아니라**
원래 solver와 보조 SDE를 분할 결합했다. 보조 선형계의 정확 적분이 전체
비선형 결합계의 정확 적분이나 안정성 증명을 뜻하지 않는다.

## 4. 장기 보정에 대한 현재의 한계

\[
m_r(t)=e^{-t/\tau_r}m_r(0)
+\frac{1}{\tau_r}\int_0^t e^{-(t-s)/\tau_r}u_s\,ds.
\]

첫 항은 source 관측으로 초기화한 정보의 직접 기여다. 최장 기본 시간척도
120시간에 대해 다음처럼 줄어든다.

| 예측 lead | 초기 memory 직접 계수 |
|---|---:|
| 5일 | 36.8% |
| 10일 | 13.5% |
| 30일 | 0.248% |
| 90일 | 0.00000152% |

이는 **이후 입력을 동일하게 유지했을 때의 직접 memory 계수**다. 초기에
가한 온도 보정은 물리 상태를 통해 이후에도 전파될 수 있으므로 manifold의
모든 영향이 이 비율로 소멸한다고 해석하면 안 된다. 다만 현재 구조에는
예측 내내 공급되는 source 공간 통계 기준점이 없다는 의미는 분명하다.

학습 기본 horizon도 4개 관측 간격이다. 일별 자료에서는 4일이고, 30일이나
계절 예측의 평균·분산·저주파 스펙트럼을 직접 학습한 것이 아니다. 기억을
길게 설정하는 것만으로 장기 성능을 보장하지 못한다. 예컨대 30일 시간척도에
8일짜리 cold-start 이력만 넣으면 일정 입력에 대한 초기화도 약 23.4%만
채워진다. 시간척도, 이력 길이, 학습 horizon을 함께 설계해야 한다.

## 5. 제공된 long-term memory 논문과의 관계

Yuan et al. (2022), *The impact of long-term memory on the climate response to
greenhouse gas emissions*, [doi:10.1038/s41612-022-00298-8](https://doi.org/10.1038/s41612-022-00298-8)의
FISM은 과거 excitation에 대략 \((t-s)^{q-1}\) 형태의 멱법칙 kernel을 적용한다.
해당 전 지구 월평균 온도 분석에서는 \(q=0.4\)를 사용한다.

현재 코드의 유한한 지수 kernel 합은 그 fractional 모델과 같지 않다.
충분한 범위의 시간척도와 적절한 가중치를 가진 지수 합으로 유한 구간의
멱법칙을 근사하는 확장은 가능하지만 현재 구현·검증된 기능은 아니다.
논문의 전 지구 평균에 추정한 값을 각 격자·고도에 그대로 적용해서도 안 된다.
그 논문 역시 memory가 과도하면 응답을 과대평가할 수 있다는 문제를 다룬다.
목표는 memory를 무조건 키우는 것이 아니라 관측된 시간 상관과 응답을 맞추는 것이다.

## 6. 이번 점검에서 확인한 오류와 미흡한 부분

**수정한 오류: 오래된 spatial history를 최근 정보처럼 사용하던 오류.** 기존에는
origin 이전 표본 수만 확인해 마지막 latent를 수개월 또는 수년 동안 유지할
수 있었다. 재현에서는 2001-01-01/02 자료로 2002-01-01을 초기화하면서 마지막
latent를 8,736시간 유지했다. 내부 공백과 마지막 관측부터 origin까지의 공백을
동일한 최대 간격 계약으로 검사하도록 수정했다. 적용된 계약과 회귀 결과는
`VALIDATION.md`에 기록한다.

| 중요도 | 남은 미흡한 부분 | 영향 / 다음 작업 |
|---|---|---|
| 높음 | source memory가 초기화에만 쓰이고 thermal memory를 대체 | 두 memory state 분리. 예측 중 사용 가능한 정적·계절 통계와 동적 정보의 입력 경로를 구분 |
| 높음 | 온도만 직접 수정, 완전한 수지 제약 없음 | 수분·질량가중 에너지 수지와 안정화 feedback을 검증. 작은 tendency 상한만으로 기후 drift 억제를 보장할 수 없음 |
| 높음 | base 진단량을 보정 후 재계산하지 않음 | 저장된 강수·에너지 진단을 보정계와 일관된 진단으로 취급하지 않기 |
| 높음 | source encoder/정규화의 학습 날짜 검증 미완 | hash 동일성은 확인하지만 training split 인덱스만으로 달력 cutoff는 입증되지 않음. 실제 날짜 provenance를 확보해야 전체 pipeline의 held-out 평가를 주장 가능 |
| 높음 | 장기 통계 학습·검증 부족 | 여러 lead, seed, 계절의 CRPS, bias, variance, ACF/PSD 및 물리 수지 비교. 짧은 예측 성공과 계절 기후 안정성을 구분 |
| 중간 | noise의 공간 공분산과 memory 수송 미모델링 | 해상도별 보정, 공간 상관 noise 및 이류/비국소 memory의 ablation 필요 |
| 중간 | 전체 자료 RAM 적재, 긴 역전파의 메모리 비용 | 다년 훈련 전 chunked loader와 gradient checkpointing 등 확장 필요 |

원래 NeuralGCM, native memory, source-initialized memory, noise 없는 memory를
같은 기간·경계조건·초기 시각·ensemble 크기로 비교하는 것이 다음 실험이다.
실제 source 가중치와 ERA5 실험 없이 장기 개선 수치를 제시하지 않는다.

## 코드 위치

- `src/ngcm_memory/spatial.py`: source encoder 이식과 checkpoint 계약
- `src/ngcm_memory/features.py`: 과거 선택, 격자 보간, memory 초기화
- `src/ngcm_memory/closure.py`: memory/OU 갱신과 온도 적분 보정
- `src/ngcm_memory/neuralgcm_adapter.py`: 실제 NeuralGCM 상태에 적용
- `src/ngcm_memory/training.py`: 보정된 recurrence를 통한 학습
- `scripts/train.py`, `scripts/evaluate.py`: 시간 분할과 평가 조건

# 연속 latent 전이와 누적 memory — v0.2

이전 구현도 여러 과거 표본을 memory에 누적했다. 그러나 예측 중 source latent
입력은 0이었으므로 **새로운 manifold latent를 예측하고 다음 입력으로 전달하는
경로**는 없었다. 이번 버전은 그 경로를 구현한다. 단순히 고정된 과거 `z_0`를
반복 공급하는 모델이 아니다.

![실제 학습 구조](figures/recurrent_manifold_training.png)

[SVG 원본](figures/recurrent_manifold_training.svg)은
`python scripts/render_architecture.py`로 PNG와 함께 다시 생성할 수 있다.

## 1. 예측 상태를 명시적으로 확장

\[
S_t=(X_t,z_t,M^s_t,m^c_t,Z_t).
\]

| 상태 | 의미 |
|---|---|
| \(X_t\) | NeuralGCM의 spectral/sigma 대기 상태 |
| \(z_t\) | source spatial encoder의 좌표계에서 전진하는 spatial latent |
| \(M^s_t\) | latent 이력을 누적하는 별도 공간 memory |
| \(m^c_t\) | 온도 특징과 latent를 함께 요약하는 closure memory |
| \(Z_t\) | 온도 보정에 쓰이는 OU 보조 상태 |

각 ensemble member는 별도의 상태와 RNG를 가진다. NeuralGCM, latent 전이,
온도 OU의 난수 경로도 분리한다.

시작 시각 이전의 관측 latent들로 \(M^s_0\)를 채우고, **시작 시각과 정확히
일치하는 관측 latent**로 \(z_0\)를 초기화한다. 최신 과거 표본을 현재 표본으로
이름만 바꾸지 않는다. 이것은 origin 관측을 허용하는 새 초기화 계약이다.
이전 초기화 전용 모델은 strictly pre-origin 계약을 그대로 유지한다.

NeuralGCM의 과거 관측 warmup은 source 채널을 0으로 두고 latent 상태를
동결한 채 closure memory/OU를 채운다. 이후 spatial 초기화는 그 closure
상태를 보존한다. 초기화 때 서로 다른 memory가 덮어써지지 않는다.

## 2. 조건부 확률 전이

현재 온도 특징을 \(T_t=T'_t/10\mathrm K\)라 쓰면 구현된 전이는 다음과 같다.
10 K는 기본 feature scale이며 설정할 수 있다.

\[
dz_t=\frac{g_\theta(z_t,T_t,M^s_t)-z_t}{\tau_z}dt
+\sigma_\theta(z_t,T_t,M^s_t)dW_t.
\]

한 내부 스텝 \(h\) 동안 두 network 출력을 고정하면

\[
z_{t+h}=a_z z_t+(1-a_z)g_\theta
+\sigma_\theta\sqrt{\frac{\tau_z}{2}(1-a_z^2)}\,\epsilon_t,
\quad a_z=e^{-h/\tau_z},\quad\epsilon_t\sim\mathcal N(0,I).
\]

따라서 코드에 실제 \(p_\theta(z_{t+h}\mid z_t,T_t,M^s_t)\)가 있으며, 평균과
대각 분산은 `transition_moments()`에서 직접 얻는다. \(z_{t+h}\)를 다음
상태에 저장해 계속 사용한다. 새 innovation만 추출하고 기존 latent를 버리는
구조가 아니다. 시간은 hour, latent는 무차원, diffusion은 latent 단위/√hour다.

이는 **지도 확률 상태공간 모델**이다. 강화학습의 reward, action, policy,
value function은 구현하지 않았다. 전이 확률을 사용한다고 강화학습인 것은 아니다.
또한 위 Gaussian은 frozen-coefficient 한 스텝의 전이이며, 전체 비선형 결합
SDE를 정확히 적분했다는 의미는 아니다.

## 3. 여러 시점이 어떻게 계속 누적되는가

각 spatial memory는 현재 \(z_t\)로 갱신된다.

\[
M^s_{r,t+h}=a_rM^s_{r,t}+(1-a_r)z_t,\qquad a_r=e^{-h/\tau_r}.
\]

고정된 시간 간격에서 이를 펼치면

\[
M^s_{r,n}=a_r^nM^s_{r,0}
+(1-a_r)\sum_{j=0}^{n-1}a_r^{n-1-j}z_j.
\]

즉 **\(z_0,z_1,\ldots,z_{n-1}\)가 모두 가중 누적**된다. 새 \(z_j\)는 이전
latent와 새 noise의 영향을 포함하므로 그 영향도 다음 전이와 memory로 이어진다.
다만 무제한 단순합이 아니라 감쇠와 갱신이 있는 누적이다. 일반적인 가변 시간
간격에서는 각 단계의 감쇠계수 곱으로 같은 원리가 적용된다.

매 내부 스텝의 실행 순서는 다음과 같다.

1. 현재 \(X_t,z_t\)에서 온도·latent 특징을 만든다.
2. 원래 NeuralGCM이 대기 상태를 한 단계 전진시킨다.
3. 현재 특징과 \(m^c_t,Z_t\)로 적분 온도 보정 \(\Delta T\)를 계산해 더한다.
4. 현재 \(z_t,T_t,M^s_t\)에서 다음 latent를 생성하고 spatial memory를 갱신한다.
5. 보정된 대기 상태와 갱신된 모든 보조 상태를 다음 단계에 전달한다.

구현상 독립인 계산의 실행 순서는 바뀔 수 있지만, 이 한 스텝의 조건은 모두
**현재 상태**다. 새 \(z_{t+h}\)가 closure 입력으로 쓰이는 것은 다음 스텝부터다.
공간 memory는 전이 모델을 거쳐 미래 latent와 이후 온도 보정에 영향을 준다.
예측 중 실제 미래 관측으로 \(z\)를 다시 설정하는 teacher forcing은 없다.

## 4. 전이 모델도 실제로 학습한다

\[
\mathcal L=\mathcal L_{\mathrm{fields}}
+\lambda_z\mathcal L_{\mathrm{latent}},\qquad
z^*_{t_k}=E_s(s^*_{t_k}).
\]

두 손실은 자유 예측 ensemble의 fair CRPS와 분산 보정 predictive-mean MSE를
사용한다. latent loss는 기본 가중치 0.1, 좌표별 scale 1이며 원본 encoder의
정규화가 유지된 좌표계에서 계산한다. 물리장과 latent는 각각의 격자에 맞는
면적 가중치를 사용한다. source latent의 정규화와 loss 가중치는 실험에서
검증해야 하며, validation 자료로 정규화를 다시 맞추지 않는다.

일별 자료라면 내부 시간별 전이를 반복해 **하루 뒤의 예측 latent**를 그날의
정답과 비교한다. 일별 정답을 매 시간 주입하거나, 하루 전이를 시간별 NLL의
정답인 것처럼 사용하지 않는다. 따라서 현재 학습은 sequence ensemble score이고
각 관측 간격의 정확한 transition likelihood를 계산하는 구현은 아니다.

미래 source 관측은 frozen encoder로 정답을 만드는 용도뿐이다. rollout은
context에서 `latent_targets`를 제거하고 초기화하며, loss만 이 key를 읽는다.
NeuralGCM과 source encoder/정규화는 고정하고 **closure + latent 전이**를
하나의 optimizer로 갱신한다. validation 역시 같은 총점으로 checkpoint를 고른다.
원래 source 학습의 PINN/W2/KL/signed/flow/reconstruction loss는 이 optimizer에
추가하지 않았다.

## 5. 사용 및 호환성

```bash
python scripts/train.py \
  --base-checkpoint checkpoints/base.pkl --trust-base-checkpoint \
  --data data/train_validation_prepared.nc \
  --spatial-history data/spatial_history.npz --spatial-history-steps 8 \
  --spatial-mode recurrent --latent-loss-weight 0.1 \
  --latent-tau-hours 24 --latent-memory-tau-hours 24,120,720 \
  --output runs/recurrent/closure.npz \
  --warmup-steps 2 --horizon-steps 4 --ensemble-size 4
```

설치·데이터 준비·평가 명령은 [GUIDE_KO.md](GUIDE_KO.md)를 따른다.

- Source history 제공 시 새 학습 기본값은 recurrent다.
- `--spatial-mode initialization`은 이전 초기화 전용 비교 모델이다.
- history를 생략하면 기존 native-temperature 모델이다.
- `--latent-loss-weight 0`은 latent label 감독을 끄는 명시적 ablation이다.
- `--no-memory`는 두 지수 memory를 끄지만 recurrent latent 자체는 남긴다.
- `--no-stochastic`은 추가한 latent noise와 온도 OU를 끈다. 원본 stochastic
  NeuralGCM의 확률성까지 끄지는 않는다.
- 이전 checkpoint는 `latent_dim=0`으로 복원된다. 새 파라미터를 임의로 채워
  이미 학습한 recurrent 모델인 것처럼 변환하지 않는다.
- 학습에는 origin과 예측 정답 시각의 source latent가 정확히 있어야 한다.
  평가에는 source 자료가 각 origin까지만 있어도 된다.

Native warmup/target 창은 시간 분할을 지킨다. 추가 source 과거는 origin보다
이르면 partition 경계를 넘어 이전 기간을 사용할 수 있는 정책이다. 이것은
과거 conditioning이며 미래 target 누출은 아니지만, 모든 입력 history가
partition 안에만 있다는 뜻은 아니다. Source encoder/정규화 자체의 학습 날짜는
별도로 검증해야 전체 pipeline의 독립 평가를 주장할 수 있다.

## 6. 남은 연구 범위

기본 공간 memory 24/120/720시간은 시험할 설정이다. **8개의 일별 표본은 8일**이므로
30일 시간척도의 memory를 충분히 초기화하지 못한다. 더 긴 기억을 쓸 때에는
과거 길이와 학습 horizon도 늘려야 한다. 현재의 유한 지수 합은 fractional
power-law 장기 기억이나 해양의 느린 상태 자체를 구현한 것이 아니다.

전이 NN은 각 격자에서 온도·latent·memory를 섞는다. 바람·수분·압력을 직접
조건화하지 않고, memory 이류나 격자 사이 learned covariance도 없다. Gaussian
latent가 원래 기하학적 manifold 위에 머무른다는 hard constraint는 없다.
온도만 보정하는 구조의 에너지·수분 수지, 진단량 일관성, 장기 drift 및 기후
분포 개선은 별도 검증이 필요하다. 실행·gradient 검증은 그 성능의 증거가 아니다.

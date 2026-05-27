# MNIST 손글씨 인식 — NumPy 신경망

---

## MNIST 손글씨 인식 과제

### NumPy만으로 구현한 신경망 — Baseline → 회고 → 개선 모델 (v2)

| 반 | 팀 | 팀원 |
| -- | -- | --- |
| SW-AI | 1조 | 박지용 · 김석제 · 서원규 · 김진호 |

Baseline 모델로 **Test 98.41%**를 먼저 달성한 뒤, 회고에서 발견한 3가지 문제를 1:1로 처방한 **개선 모델 (v2)**로 **Test 98.62%** 도달

---

## 1. 과제 & 목표

**무엇을 하는가**
- MNIST 10-class 분류를 PyTorch / TensorFlow 없이 **NumPy만으로**
- Forward → Loss → Backward → Optimizer Update **전 사이클 직접 구현**

**목표**
- Test accuracy **≥ 95%** (권장 **≥ 97%**)
- 참고: 『밑바닥부터 시작하는 딥러닝』 1–6장

**검증 기준**
- 학습 정확도 + **단위 테스트 21개** (각 layer / optimizer forward·backward) 전부 통과

---

## 2. Baseline — 먼저 만든 모델과 그 한계

**Baseline 구성**
- 은닉층 폭 512 / 256 · Dropout 0.3 · learning rate 1e-3 **고정** · 15 epoch · Adam · He 초기화
- 총 파라미터 537,354개, 학습 시간 약 4분

**Baseline 결과 — 목표는 통과, 하지만…**
- **Test 98.41% / Train 99.85%** — 권장 목표 97%를 통과
- 그러나 **세 가지 신호**가 회고에서 눈에 띔

| 신호 | 무엇을 보았나 | 어떤 문제로 해석했나 |
| -- | -- | -- |
| **① 후반 loss saturation** | epoch 13→14→15 loss: 0.0306 → 0.0270 → 0.0252. 더 이상 의미 있게 안 떨어짐 | **lr이 후반에는 너무 커서 minimum 주변을 진동** — fine-tuning이 안 됨 |
| **② Train/Test 1.44%p 격차** | Train 99.85% vs Test 98.41% | **경미한 과적합** — train이 거의 100%까지 올라온 만큼 일반화 여력을 더 짤 수 있음 |
| **③ Capacity 한계 의문** | 537K 파라미터 모델이 train 99.85%에서 멈춤 | 이게 진짜 **capacity 상한**인지, 학습 부족인지 불명확 — 더 큰 모델로 확인 필요 |

이 세 신호가 다음 슬라이드의 **세 가지 처방의 근거**가 됩니다.

---

## 3. 개선 방향 — 문제별 처방

세 신호에 **1:1로 대응**하는 처방을 선택했습니다.

| Baseline 문제 | 처방 | 이론적 근거 |
| -- | -- | -- |
| ① loss saturation | **lr step decay**: 1e-3 → 1e-4 (epoch 10/11 경계) | lr이 클 때는 진동만, 작을 때는 fine-tuning. **후반에 lr을 줄여 fine-tuning 단계**를 만드는 표준 전략 |
| ② 1.44%p train/test 격차 | **Dropout 0.3 → 0.4** | 정규화 강도 ↑ → 일반화 ↑ (격차 축소 기대) |
| ③ Capacity 의문 | **은닉층 폭 2배** (512/256 → 1024/512) | (a) capacity 한계 가설 검증 + (b) **Dropout이 강해지면 effective capacity가 줄어드므로 폭 증가로 보상**. "정규화 강화 + capacity 증가"는 표준 페어링 |

**Xavier 초기화 비교는 왜 제외했나**
- **He 초기화**는 ReLU의 "음수 절반 차단" 손실을 분산 2배(√(2/fan_in))로 보상하는 표준 — **우리 활성화 함수에 정확히 맞음**
- **Xavier 초기화**는 sigmoid/tanh처럼 대칭 활성화에 맞춰진 분산 √(1/fan_in) — ReLU에선 신호가 깊어질수록 작아지는 **vanishing 가능성 ↑**
- 이론적으로 He 우위가 명백 → "한 번 더 음성 결과를 확인하는 실험"보다는 **위 3개 처방의 효과 확인이 더 의미 있다고 판단**

세 처방을 한 번에 적용한 결과물을 **"개선 모델 (v2)"**이라고 부릅니다.

> **왜 "Best"가 아니라 "v2"인가?**
> 단 1회 비교에서의 우승자일 뿐이고, 진정한 최적은 더 많은 실험(특히 ablation)을 거쳐야 알 수 있어 "Best"는 과한 표현으로 판단. **"baseline 회고로부터 도출한 두 번째 버전"**이라는 의미로 v2로 통일합니다.

---

## 4. 모델 구조 — Baseline vs v2

| 구분           | Baseline                                   | 개선 모델 (v2)                                       |
| ------------ | ------------------------------------------ | --------------------------------------------------- |
| 은닉층 1        | Affine **512** + BN + ReLU + Dropout **0.3** | Affine **1024** + BN + ReLU + Dropout **0.4**       |
| 은닉층 2        | Affine **256** + BN + ReLU + Dropout **0.3** | Affine **512**  + BN + ReLU + Dropout **0.4**       |
| 출력층          | Affine 10 + Softmax                        | Affine 10 + Softmax                                 |
| 초기화          | He                                         | He (동일)                                             |
| **총 파라미터 수** | **537,354**                                | **1,336,842** (≈ 2.5×)                              |

```
입력 784  (28×28 픽셀, 0–1 정규화)
   │
   ▼
[Affine] → BatchNorm → ReLU → Dropout    ← 은닉 블록 ×2 (폭과 dropout 비율이 두 모델 차이)
   │
   ▼
[Affine 10] → Softmax → 클래스 확률
```

은닉층 폭을 두 배로 키운 이유는 §3에서 설명한 처방 ③ — capacity 가설 검증 + Dropout 강화 보상입니다.

---

## 5. 학습 설정 — Baseline vs v2

| 항목                     | Baseline                         | 개선 모델 (v2)                                       |
| ---------------------- | -------------------------------- | --------------------------------------------------- |
| 옵티마이저                  | Adam (β₁=0.9, β₂=0.999, ε=1e-8)  | Adam (동일)                                          |
| **learning rate (lr)** | **1e-3 고정**                      | **Step decay**: epoch 1–10 = 1e-3 → 11–15 = 1e-4    |
| Dropout 비율             | 0.3                              | 0.4                                                 |
| epochs                 | 15                               | 15                                                  |
| batch size             | 128                              | 128                                                 |
| 셔플                     | 매 epoch `np.random.permutation`  | 동일                                                  |
| seed                   | 42                               | 42                                                  |

**미니배치마다 반복하는 4단계** (둘 다 동일)

1. **Forward** — `model.forward(x_batch, train=True)`
2. **Loss** — `cross_entropy_loss(y_pred, y_batch)`
3. **Backward** — Softmax+CE 결합 gradient `(y_pred − y_one_hot) / N`
4. **Update** — `optimizer.update(params, grads)` (in-place)

---

## 6. 학습 손실 — 처방 ①의 효과가 가장 또렷

![Training loss curve — Baseline vs v2](figures/loss_curve.png)

- 두 모델 모두 **단조 감소**, 진동·재상승 없음
- **epoch 10 → 11**에서 v2의 lr이 1e-3 → 1e-4로 전환되며 loss **0.0375 → 0.0257 (-31%)** 한 번 더 큰 폭 감소 → **처방 ①(lr step decay)가 saturation을 풀어냈다는 직접 증거**
- 마지막 epoch loss: Baseline **0.0252** vs v2 **0.0154**

---

## 7. 최종 정확도 — 세 처방의 종합 효과

![Baseline vs v2 comparison](figures/baseline_vs_best.png)

| 지표              | Baseline   | 개선 모델 (v2) | 변화           |
| --------------- | ---------- | ------------ | ------------ |
| **Test accuracy** | 98.41%   | **98.62%**   | **+0.21%p**  |
| Train accuracy  | 99.85%     | 99.95%       | +0.10%p      |
| Train/Test 격차   | 1.44%p     | 1.33%p       | -0.11%p      |
| 총 파라미터 수        | 537,354    | 1,336,842    | ≈ 2.5×       |
| 학습 시간           | 234초 (≈ 4분) | 1371초 (≈ 23분) | ≈ 6×        |
| 단위 테스트          | 21개 전부 통과 | 21개 전부 통과   | —            |

두 모델 모두 권장 목표 97% 초과 달성, v2는 **+1.62%p 초과 달성**.

환경: Python 3.11.15 · NumPy 2.4.6 · Matplotlib 3.10.9 · pytest 9.0.3 · Windows 11 (로컬 CPU)

---

## 8. 처방별 효과 회고 — 세 처방은 의도대로 작동했나

세 처방을 동시에 적용한 단일 실험이라 효과 분리(ablation)는 한계지만, 관찰 신호로부터 다음을 추정합니다.

**① lr step decay — 의도대로, 효과 가장 또렷**
- 의도: 후반 saturation 해소
- 관찰: epoch 10→11에서 loss -31% 추가 감소. saturate되던 후반에서 명확히 작동
- 결론: **세 처방 중 가장 큰 단일 기여로 추정**

**② Dropout 0.3 → 0.4 — 의도대로, 효과는 작지만 방향 일치**
- 의도: 1.44%p 격차 축소
- 관찰: 격차 1.44%p → 1.33%p (-0.11%p)
- 결론: **방향은 맞지만 폭은 작음.** 더 큰 폭(0.5)이나 다른 정규화(데이터 증강)와 조합 필요할 수 있음

**③ 은닉층 폭 2배 — capacity 한계가 아니었음을 확인**
- 의도: (a) capacity 한계 가설 검증 + (b) Dropout 강화 보상
- 관찰: train 99.85% → 99.95% (+0.10%p). **더 큰 모델이 더 높은 train accuracy를 달성** → baseline의 99.85%는 진정한 capacity 한계가 아니었음
- 결론: **capacity 가설은 기각**, 단독 효과는 작지만 Dropout 0.4와 함께 묶여야 효과 — 처방 ②의 동반자 역할

---

## 9. 구현하며 배운 핵심

**in-place 갱신 (`-=`)**
`=` 재할당하면 dict만 바뀌고 layer는 옛 배열을 계속 봐서 학습이 안 됨

**train / test 모드 분리**
BatchNorm은 배치 통계 ↔ running 통계, Dropout은 mask ↔ scale — 잘못 쓰면 정확도 폭락

**Softmax + Cross Entropy 결합 미분**
둘을 합쳐 미분하면 `(y_pred − y_true) / N`으로 깔끔하게 정리

**Shape 매칭이 backward 디버깅 1차 관문**
`dW.shape == W.shape`, `db.shape == b.shape`, `dγ.shape == γ.shape`

---

## 10. 한계 & 향후 과제

**현재 한계**
- **세 처방을 동시에 적용해 효과 분리가 어렵다** — 진짜 ablation은 처방을 하나씩 빼며 비교해야 함
- 학습 시간 trade-off: +0.21%p 정확도를 위해 학습 시간 **약 6배** (4분 → 23분)
- CNN 없이 MLP로는 약 98.6%대가 사실상 상한

**다음에 해 볼 것**
- **Ablation 실험**: 처방을 1개씩만 적용한 3개 모델을 만들어 단독 효과 정량화
- Xavier vs He 정량 비교 (이론적 결론과 실측 일치 확인)
- Dropout 0.5 + 더 깊은 네트워크 (3–4 은닉층)
- Data augmentation (작은 회전 / 시프트로 train/test gap 축소)
- Cosine annealing 등 부드러운 lr 스케줄링과 step decay 비교

---

## 부록 A · 코드 구조 & 재현 방법

```
src/
├── activations.py     # ReLU, Softmax
├── layers.py          # Affine, BatchNorm, Dropout
├── losses.py          # cross_entropy_loss
├── optimizers.py      # SGD, Adam
├── network.py         # NeuralNetwork (조립, hidden_sizes/init 인자)
├── training.py        # train, evaluate, plot_loss_history
└── data.py            # load_mnist (제공됨)

scripts/
├── run_mnist.py       # Baseline 구성 학습 (≈ 4분, Test 98.41%)
├── run_experiment.py  # 개선 모델 (v2) 학습 (≈ 23분, Test 98.62%)
└── make_plots.py      # 발표용 PNG 생성

figures/
├── loss_curve.png
└── baseline_vs_best.png
```

```bash
conda create -n mnist-lab python=3.11 -y
conda activate mnist-lab
pip install -r requirements.txt
python download_mnist.py            # 데이터 다운로드 (최초 1회)
pytest tests/ -v                    # 단위 테스트 21개
python scripts/run_mnist.py         # Baseline 학습 + 평가
python scripts/run_experiment.py    # v2 학습 + 평가
python scripts/make_plots.py        # 발표 그림 재생성
```

---

## 부록 B · 손실 커브 원자료

| Epoch | Baseline | v2 (lr)       | Epoch | Baseline | v2 (lr)       | Epoch | Baseline | v2 (lr)       |
| ----- | -------- | ------------- | ----- | -------- | ------------- | ----- | -------- | ------------- |
| 1     | 0.2884   | 0.2577 (1e-3) | 6     | 0.0554   | 0.0540 (1e-3) | 11    | 0.0351   | 0.0257 (1e-4) |
| 2     | 0.1281   | 0.1219 (1e-3) | 7     | 0.0491   | 0.0487 (1e-3) | 12    | 0.0307   | 0.0213 (1e-4) |
| 3     | 0.0951   | 0.0907 (1e-3) | 8     | 0.0430   | 0.0442 (1e-3) | 13    | 0.0306   | 0.0195 (1e-4) |
| 4     | 0.0763   | 0.0737 (1e-3) | 9     | 0.0405   | 0.0398 (1e-3) | 14    | 0.0270   | 0.0175 (1e-4) |
| 5     | 0.0670   | 0.0630 (1e-3) | 10    | 0.0376   | 0.0375 (1e-3) | 15    | 0.0252   | 0.0154 (1e-4) |

---

## 부록 C · 단위 테스트 결과

```
tests/test_relu.py ...                          [ 14%]
tests/test_softmax.py ...                       [ 28%]
tests/test_affine.py ..                         [ 38%]
tests/test_cross_entropy_loss.py ..             [ 47%]
tests/test_sgd.py .                             [ 52%]
tests/test_adam.py .                            [ 57%]
tests/test_neural_network.py ...                [ 71%]
tests/test_batchnorm.py ..                      [ 80%]
tests/test_dropout.py ..                        [ 90%]
tests/test_evaluate.py .                        [ 95%]
tests/test_training.py .                        [100%]

============================ 21 passed in 0.60s ===============================
```

---

## 부록 D · 파라미터 분포 (개선 모델 v2)

| Layer       | 파라미터                  | 개수            |
| ----------- | --------------------- | ------------- |
| Affine 1    | W(784×1024) + b(1024) | 803,840       |
| BatchNorm 1 | γ(1024) + β(1024)     | 2,048         |
| Affine 2    | W(1024×512) + b(512)  | 524,800       |
| BatchNorm 2 | γ(512) + β(512)       | 1,024         |
| Affine 3    | W(512×10) + b(10)     | 5,130         |
| **합계**      |                       | **1,336,842** |

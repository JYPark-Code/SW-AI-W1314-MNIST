# MNIST 손글씨 인식 — NumPy 신경망

---

## MNIST 손글씨 인식 과제

### NumPy만으로 구현한 신경망 — BatchNorm + Dropout + Adam + learning rate decay

| 반 | 팀 | 팀원 |
| -- | -- | --- |
| SW-AI | 1조 | 박지용 · 김석제 · 서원규 · 김진호 |

프레임워크 없이 직접 짠 2-층 MLP로 **Test 98.62%** 달성

---

## 1. 과제 & 목표

**무엇을 하는가**
- MNIST 10-class 분류를 PyTorch / TensorFlow 없이 **NumPy만으로**
- Forward → Loss → Backward → Optimizer Update **전 사이클 직접 구현**

**목표**
- Test accuracy **≥ 95%** (권장 **≥ 97%**)
- 참고: 『밑바닥부터 시작하는 딥러닝』 1~6장

**검증 기준**
- 학습 정확도 + **단위 테스트 21개** (각 layer / optimizer forward·backward) 전부 통과

---

## 2. 접근 방식 — 두 모델

**두 단계로 진행**

```
① Baseline 모델   →   ② 회고에서 개선 후보 도출   →   ③ Best 모델
   (목표 달성 확인)      (4가지 개선안)                 (3가지 적용한 개선판)
```

**왜 두 모델인가**
- **Baseline**: 기본 구성(은닉층 512/256, Dropout 0.3, learning rate 1e-3 고정)으로 먼저 학습. 권장 목표 97%를 달성할 수 있는지 확인.
- **개선 후보 4가지**: ① Dropout 강화, ② **learning rate**(학습률, 이하 **lr**) 스케줄링, ③ 은닉층 폭 변경, ④ He vs Xavier 초기화 비교
- **Best**: 후보 중 효과 클 것 같은 **3가지를 한 번에 적용**. Xavier는 ReLU에서 이론적으로 He보다 불리해 단독 비교는 제외.

이후 슬라이드의 **모든 비교는 Baseline vs Best 두 모델 기준**.

---

## 3. 모델 구조 — Baseline vs Best

| 구분           | Baseline                                   | Best                                                |
| ------------ | ------------------------------------------ | --------------------------------------------------- |
| 은닉층 1        | Affine **512** + BN + ReLU + Dropout **0.3** | Affine **1024** + BN + ReLU + Dropout **0.4**       |
| 은닉층 2        | Affine **256** + BN + ReLU + Dropout **0.3** | Affine **512**  + BN + ReLU + Dropout **0.4**       |
| 출력층          | Affine 10 + Softmax                        | Affine 10 + Softmax                                 |
| 초기화          | He (W ~ N(0, √(2/fan_in)))                 | He (동일)                                             |
| **총 파라미터 수** | **537,354**                                | **1,336,842** (≈ 2.5×)                              |

```
입력 784  (28×28 픽셀, 0~1 정규화)
   │
   ▼
[Affine] → BatchNorm → ReLU → Dropout    ← 은닉 블록 ×2 (폭과 dropout 비율이 두 모델 차이)
   │
   ▼
[Affine 10] → Softmax → 클래스 확률
```

---

## 4. 학습 설정 — Baseline vs Best

| 항목                  | Baseline       | Best                                              |
| ------------------- | -------------- | ------------------------------------------------- |
| 옵티마이저               | Adam (β₁=0.9, β₂=0.999, ε=1e-8) | Adam (동일)                          |
| **learning rate (lr)** | **1e-3 고정**  | **Step decay**: epoch 1~10 = 1e-3 → 11~15 = 1e-4 |
| Dropout 비율          | 0.3            | 0.4                                               |
| epochs              | 15             | 15                                                |
| batch size          | 128            | 128                                               |
| 셔플                  | 매 epoch `np.random.permutation` | 동일                              |
| seed                | 42             | 42                                                |

**미니배치마다 반복하는 4단계** (둘 다 동일)

1. **Forward** — `model.forward(x_batch, train=True)`
2. **Loss** — `cross_entropy_loss(y_pred, y_batch)`
3. **Backward** — Softmax+CE 결합 gradient `(y_pred − y_one_hot) / N`
4. **Update** — `optimizer.update(params, grads)` (in-place)

> 두 모델의 차이는 **은닉층 폭 / Dropout / lr 스케줄링 세 가지뿐**, 나머지는 모두 동일.

---

## 5. 학습 손실 — Baseline vs Best

![Training loss curve — Baseline vs Best](figures/loss_curve.png)

- 두 모델 모두 진동·재상승 없는 **단조 감소**
- **epoch 10 → 11**에서 Best의 lr이 1e-3 → 1e-4로 전환되며 loss **0.0375 → 0.0257 (-31%)** 한 번 더 큰 폭 감소
- 마지막 epoch loss: Baseline **0.0252** vs Best **0.0154**

---

## 6. 최종 정확도 — Baseline vs Best

![Baseline vs Best comparison](figures/baseline_vs_best.png)

| 지표              | Baseline   | Best         | 변화           |
| --------------- | ---------- | ------------ | ------------ |
| **Test accuracy** | 98.41%   | **98.62%**   | **+0.21%p**  |
| Train accuracy  | 99.85%     | 99.95%       | +0.10%p      |
| Train/Test 격차   | 1.44%p     | 1.33%p       | -0.11%p      |
| 총 파라미터 수        | 537,354    | 1,336,842    | ≈ 2.5×       |
| 학습 시간           | 234초 (≈ 4분) | 1371초 (≈ 23분) | ≈ 6×        |
| 단위 테스트          | 21개 전부 통과 | 21개 전부 통과   | —            |

두 모델 모두 권장 목표 97% 초과 달성, Best는 **+1.62%p 초과 달성**.

환경: Python 3.11.15 · NumPy 2.4.6 · Matplotlib 3.10.9 · pytest 9.0.3 · Windows 11 (로컬 CPU)

---

## 7. 어떤 변경이 가장 효과적이었나

**1. lr step decay — 가장 또렷한 효과**
saturate되던 후반부에서 loss를 추가로 -31% 감소시킴 (epoch 10→11)

**2. 은닉층 폭 ↑ — capacity 확보**
train 99.85 → 99.95%. 단독으로는 큰 차이 없지만 다른 변수와 함께 시너지

**3. Dropout 0.3 → 0.4 — 일반화 향상**
train/test 격차 1.44%p → 1.33%p

---

## 8. 구현하며 배운 핵심

**in-place 갱신 (`-=`)**
`=` 재할당하면 dict만 바뀌고 layer는 옛 배열을 계속 봐서 학습이 안 됨

**train / test 모드 분리**
BatchNorm은 배치 통계 ↔ running 통계, Dropout은 mask ↔ scale — 잘못 쓰면 정확도 폭락

**Softmax + Cross Entropy 결합 미분**
둘을 합쳐 미분하면 `(y_pred − y_true) / N`으로 깔끔하게 정리

**Shape 매칭이 backward 디버깅 1차 관문**
`dW.shape == W.shape`, `db.shape == b.shape`, `dγ.shape == γ.shape`

---

## 9. 한계 & 향후 과제

**현재 한계**
- 학습 시간 trade-off: +0.21%p 정확도를 위해 학습 시간 **약 6배** (4분 → 23분)
- CNN 없이 MLP로는 ~98.6%대가 사실상 상한

**다음에 해 볼 것**
- Xavier vs He 정량 비교 (ReLU에서 실제 성능 저하 폭)
- Dropout 0.5 + 더 깊은 네트워크 (3~4 은닉층)
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
├── run_experiment.py  # Best 구성 학습 (≈ 23분, Test 98.62%)
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
python scripts/run_experiment.py    # Best 학습 + 평가
python scripts/make_plots.py        # 발표 그림 재생성
```

---

## 부록 B · 손실 커브 원자료

| Epoch | Baseline | Best (lr) | Epoch | Baseline | Best (lr) | Epoch | Baseline | Best (lr)    |
| ----- | -------- | --------- | ----- | -------- | --------- | ----- | -------- | ------------ |
| 1     | 0.2884   | 0.2577 (1e-3) | 6 | 0.0554 | 0.0540 (1e-3) | 11 | 0.0351 | 0.0257 (1e-4) |
| 2     | 0.1281   | 0.1219 (1e-3) | 7 | 0.0491 | 0.0487 (1e-3) | 12 | 0.0307 | 0.0213 (1e-4) |
| 3     | 0.0951   | 0.0907 (1e-3) | 8 | 0.0430 | 0.0442 (1e-3) | 13 | 0.0306 | 0.0195 (1e-4) |
| 4     | 0.0763   | 0.0737 (1e-3) | 9 | 0.0405 | 0.0398 (1e-3) | 14 | 0.0270 | 0.0175 (1e-4) |
| 5     | 0.0670   | 0.0630 (1e-3) | 10 | 0.0376 | 0.0375 (1e-3) | 15 | 0.0252 | 0.0154 (1e-4) |

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

## 부록 D · 파라미터 분포 (Best 모델)

| Layer       | 파라미터                  | 개수            |
| ----------- | --------------------- | ------------- |
| Affine 1    | W(784×1024) + b(1024) | 803,840       |
| BatchNorm 1 | γ(1024) + β(1024)     | 2,048         |
| Affine 2    | W(1024×512) + b(512)  | 524,800       |
| BatchNorm 2 | γ(512) + β(512)       | 1,024         |
| Affine 3    | W(512×10) + b(10)     | 5,130         |
| **합계**      |                       | **1,336,842** |

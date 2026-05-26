# MNIST 손글씨 인식 과제 보고서

## 0. 반·팀원

| 항목     | 내용                              |
| ------ | ------------------------------- |
| **반**  | SW-AI                           |
| **팀**  | 1조                              |
| **팀원** | 박지용, 김석제, 서원규, 김진호              |

---

## 1. 실험 목적

MNIST 10-class 분류를 **NumPy만으로 구현한 신경망**으로 수행하고, 테스트 정확도와 학습 과정을 보고합니다. PyTorch/TensorFlow 같은 딥러닝 프레임워크 없이 Forward → Loss → Backward → Optimizer Update 전 사이클을 직접 구현해 신경망의 동작 원리를 검증하는 것이 목표입니다.

- **요구 정확도**: 테스트 95% 이상 (권장 97% 이상)
- **참고 도서**: 『밑바닥부터 시작하는 딥러닝』 1~6장

---

## 2. 모델 구조

| 구분      | 내용                                                                  |
| ------- | ------------------------------------------------------------------- |
| **입력**  | 784 (28×28 픽셀, 0~1 정규화)                                             |
| **은닉층** | 2개. 각 블록: Affine → BatchNorm → ReLU → Dropout                       |
| **출력**  | Affine(→10) + Softmax                                               |
| **초기화** | He 초기화 (W ~ N(0, √(2/fan_in)), b = 0)                               |

**상세 구조**

```
입력 784
  → Affine(512) → BatchNorm → ReLU → Dropout(0.3)
  → Affine(256) → BatchNorm → ReLU → Dropout(0.3)
  → Affine(10) → Softmax
```

**총 파라미터 수**: 537,354

| Layer        | 파라미터                                       | 개수      |
| ------------ | ----------------------------------------- | ------- |
| Affine 1     | W(784×512) + b(512)                       | 401,920 |
| BatchNorm 1  | γ(512) + β(512)                           | 1,024   |
| Affine 2     | W(512×256) + b(256)                       | 131,328 |
| BatchNorm 2  | γ(256) + β(256)                           | 512     |
| Affine 3     | W(256×10) + b(10)                         | 2,570   |
| **합계**       |                                           | **537,354** |

---

## 3. 학습 설정

| 항목                 | 값      |
| ------------------ | ------ |
| 옵티마이저              | Adam   |
| 학습률 (lr)           | 0.001  |
| Adam β₁ / β₂ / ε   | 0.9 / 0.999 / 1e-8 |
| epochs             | 15     |
| batch_size         | 128    |
| Dropout 비율         | 0.3    |
| BatchNorm momentum | 0.9    |
| BatchNorm eps      | 1e-7   |
| 가중치 초기화            | He (bias 0, γ=1, β=0) |
| 데이터 셔플             | 매 epoch마다 `np.random.permutation` |
| Random seed        | 42     |

**학습 루프 (미니배치마다 동일 순서)**
1. **Forward**: `model.forward(x_batch, train=True)` — BatchNorm/Dropout 학습 모드
2. **Loss**: `cross_entropy_loss(y_pred, y_batch)` — 기록용
3. **Backward**: Softmax+CE 결합 gradient `(y_pred - y_one_hot) / batch_size` 만들어 `model.backward(dout)`
4. **Update**: `optimizer.update(model.params, model.grads)` — in-place

---

## 4. 실험 환경

| 항목         | 내용                                                |
| ---------- | ------------------------------------------------- |
| Python     | 3.11.15 (conda env: mnist-lab)                    |
| NumPy      | 2.4.6                                             |
| Matplotlib | 3.10.9                                            |
| pytest     | 9.0.3                                             |
| OS         | Windows 11 (로컬 CPU 학습)                            |
| 학습 시간      | 234.6초 (약 3분 55초) — epoch당 평균 15.6초                |
| 단위 테스트     | 21개 항목 전부 통과 (`pytest tests/ -v`)                  |

---

## 5. 결과

### 5.1 정확도

| 항목              | 값          |
| --------------- | ---------- |
| **Test accuracy** | **98.41%** |
| Train accuracy  | 99.85%     |
| Train/Test 격차    | 1.44%p     |
| 총 파라미터 수        | 537,354    |

목표(권장 97%)를 1.41%p 초과 달성.

### 5.2 손실 커브

| Epoch | Loss   | Epoch | Loss   | Epoch | Loss   |
| ----- | ------ | ----- | ------ | ----- | ------ |
| 1     | 0.2884 | 6     | 0.0554 | 11    | 0.0351 |
| 2     | 0.1281 | 7     | 0.0491 | 12    | 0.0307 |
| 3     | 0.0951 | 8     | 0.0430 | 13    | 0.0306 |
| 4     | 0.0763 | 9     | 0.0405 | 14    | 0.0270 |
| 5     | 0.0670 | 10    | 0.0376 | 15    | 0.0252 |

- 단조 감소, 진동·재상승 없음
- Epoch 1→2에서 가장 큰 감소 (0.288 → 0.128), 이후 완만한 안정 수렴
- Loss 0.025 부근에서 saturation 양상

### 5.3 단위 테스트 결과

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

모든 layer/optimizer/loss의 forward·backward shape와 동작이 검증된 상태에서 학습을 진행.

---

## 6. 회고

### 6.1 수렴 양상

Adam + He 초기화 + BatchNorm의 조합으로 첫 epoch부터 loss가 빠르게 떨어졌고(0.288), 이후 epoch마다 안정적으로 감소했습니다. 진동이나 발산이 없었던 점에서 학습률 0.001과 BatchNorm 정규화가 잘 조합되었다고 판단합니다.

### 6.2 과적합 분석

Train 99.85% vs Test 98.41%로 1.44%p 차이가 발생. 일반적으로 1~2%p 격차는 정상 범위로 보지만, Dropout 비율을 0.3 → 0.4~0.5로 올리면 격차를 더 줄일 여지가 있어 보입니다. Loss curve도 epoch 13~15에서 더 이상 크게 떨어지지 않아 추가 epoch보다는 정규화 강도 조정이 효과적일 것으로 추정합니다.

### 6.3 구현 과정에서 배운 점

- **in-place 갱신의 중요성**: `params[key] -= lr * grads[key]`로 in-place 갱신해야 layer가 들고 있는 가중치 참조까지 함께 업데이트됨. `=` 재할당으로 쓰면 dict만 갱신되고 layer는 옛 배열을 그대로 써서 학습이 안 되는 미묘한 버그가 됨.
- **train/test 모드 분리**: BatchNorm은 학습 시 배치 통계, 추론 시 running 통계. Dropout은 학습 시 mask 적용, 추론 시 scale 보정. 모드를 잘못 쓰면 정확도가 폭락.
- **Softmax + Cross Entropy 결합 gradient**: 둘을 합쳐 미분하면 `(y_pred - y_true)/N`으로 단순해짐. Softmax.backward를 통과(identity)로 두고, train 루프에서 결합 gradient를 직접 만들어 model.backward에 넘기는 패턴이 깔끔.
- **Shape 매칭 규칙**: `dW.shape == W.shape`, `db.shape == b.shape`, `dγ.shape == γ.shape`. 어긋나면 optimizer.update에서 즉시 터지므로 backward 디버깅의 첫 단계.

### 6.4 추가 개선 시도 (향후 과제)

- Dropout 비율을 0.4~0.5로 올려 train/test 격차 축소
- learning rate scheduling (epoch 후반 lr 감소)
- 은닉층 폭 변경 실험 (512→1024 vs 384→192 등)
- He vs Xavier 초기화 비교

---

## 7. 코드 구조

```
src/
├── activations.py     # ReLU, Softmax
├── layers.py          # Affine, BatchNorm, Dropout
├── losses.py          # cross_entropy_loss
├── optimizers.py      # SGD, Adam
├── network.py         # NeuralNetwork (조립)
├── training.py        # train, evaluate, plot_loss_history
└── data.py            # load_mnist (제공됨)

scripts/
└── run_mnist.py       # 학습 실행 진입점 (python scripts/run_mnist.py)
```

**재현 방법** (로컬 Conda 환경)
```bash
conda create -n mnist-lab python=3.11 -y
conda activate mnist-lab
pip install -r requirements.txt
python download_mnist.py            # 데이터 다운로드 (최초 1회)
pytest tests/ -v                    # 단위 테스트
python scripts/run_mnist.py         # 학습 + 평가
```

# -*- coding: utf-8 -*-
"""
신경망 layer 모음.

학생 구현 대상:
- Affine.forward, Affine.backward
- BatchNorm.forward, BatchNorm.backward
- Dropout.forward, Dropout.backward
"""

import numpy as np


class Affine:
    """
    완전연결층(Fully Connected Layer).

    수식은 y = xW + b 입니다.
    MNIST에서는 784개 픽셀 입력을 은닉층/출력층 차원으로 선형 변환하는 역할을 합니다.
    """

    def __init__(self, W, b):
        """가중치 W와 편향 b를 외부 params dict와 같은 배열 객체로 공유합니다."""
        self.W = W
        self.b = b

    def forward(self, x):
        """
        Args:
            x: (batch_size, input_dim)

        Returns:
            (batch_size, output_dim)
        """
        # backward에서 dW = x.T @ dout 계산에 필요하므로 입력 x를 저장.
        self.x = x
        # 행렬곱 + 브로드캐스팅: (N, D_in) @ (D_in, D_out) → (N, D_out)에 b(D_out,)를 모든 행에 더함.
        return x @ self.W + self.b

    def backward(self, dout):
        """
        Args:
            dout: (batch_size, output_dim)

        Returns:
            dx: (batch_size, input_dim)

        Side effects:
            self.dW, self.db에 optimizer가 사용할 gradient를 저장합니다.
        """
        # dL/dW = x^T · dout : (D_in, N) @ (N, D_out) = (D_in, D_out) → W와 같은 shape ✓
        # 의미: "입력 feature i가 출력 j에 기여한 정도"를 모든 샘플에서 합산.
        self.dW = self.x.T @ dout
        # dL/db = sum_i dout_i : forward에서 b가 모든 샘플에 브로드캐스팅됐으니, 역방향은 합산.
        self.db = dout.sum(axis=0)
        # dL/dx = dout · W^T : (N, D_out) @ (D_out, D_in) = (N, D_in) → x와 같은 shape ✓
        # 의미: 출력 gradient를 가중치로 거슬러 입력 공간으로 되돌림.
        dx = dout @ self.W.T
        return dx


class BatchNorm:
    """
    Batch Normalization.

    미니배치 단위로 각 feature의 평균과 분산을 맞춰 학습을 안정화합니다.
    train=True일 때는 현재 배치 통계를 쓰고, 추론 때는 누적 running_mean/running_var를 사용합니다.
    """

    def __init__(self, gamma, beta, momentum=0.9):
        """
        Args:
            gamma: 정규화된 값을 다시 scale하는 학습 파라미터
            beta: 정규화된 값에 더하는 shift 학습 파라미터
            momentum: running_mean/running_var 이동평균 비율
        """
        self.gamma = gamma
        self.beta = beta
        self.momentum = momentum
        self.running_mean = np.zeros_like(beta)
        self.running_var = np.zeros_like(beta)
        self.eps = 1e-7

    def forward(self, x, train=True):
        """
        Args:
            x: (batch_size, feature_dim)
            train: True면 배치 통계, False면 running 통계 사용

        Returns:
            정규화 후 gamma, beta가 적용된 배열
        """
        if train:
            # 학습 모드: 현재 미니배치의 통계로 정규화.
            # mu: 각 feature(열)별 평균. shape (D,)
            mu = x.mean(axis=0)
            # xc: 평균을 뺀 centered 값. shape (N, D)
            xc = x - mu
            # var: 각 feature별 분산. (xc^2의 평균)
            var = (xc ** 2).mean(axis=0)
            # std: sqrt(var + eps). eps는 분산이 0일 때 0 나눗셈 방지.
            std = np.sqrt(var + self.eps)
            # xn: 정규화된 값 (평균 0, 표준편차 1).
            xn = xc / std
            # backward에서 재사용할 중간값들을 저장. (xc, xn, std, batch_size)
            self.batch_size = x.shape[0]
            self.xc = xc
            self.xn = xn
            self.std = std
            # running_mean/var: 추론에서 쓸 누적 통계. momentum=0.9면 옛 값 90% + 새 값 10%로 천천히 갱신.
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * mu
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * var
        else:
            # 추론 모드: 학습 중 누적해 둔 running 통계로 정규화.
            # 배치 크기 1로 들어와도 안정적이고, 같은 입력은 항상 같은 결과를 낸다.
            xc = x - self.running_mean
            xn = xc / np.sqrt(self.running_var + self.eps)
        # 정규화된 xn을 학습 가능한 gamma로 scale, beta로 shift → 모델이 필요하면 정규화를 "해제"할 수도 있음.
        return self.gamma * xn + self.beta

    def backward(self, dout):
        """
        BatchNorm 입력 x, scale gamma, shift beta에 대한 gradient를 계산합니다.

        Args:
            dout: 다음 층에서 넘어온 gradient

        Returns:
            dx: BatchNorm 입력 x에 대한 gradient
        """
        # BatchNorm backward는 forward의 계산 그래프를 역순으로 따라간다.
        # forward 순서: x → mu → xc → var → std → xn → out = gamma*xn + beta
        # backward는 out 쪽부터 거꾸로.

        N = self.batch_size

        # out = gamma * xn + beta
        # → dbeta = sum_i dout_i (beta는 모든 샘플에 더해졌으니 역방향은 합산)
        # → dgamma = sum_i (xn_i * dout_i)
        self.dbeta = dout.sum(axis=0)
        self.dgamma = (self.xn * dout).sum(axis=0)

        # gamma*xn 부분에서 xn 쪽으로 흘러간 gradient
        dxn = self.gamma * dout
        # xn = xc / std → dxc 첫 기여분 = dxn / std
        dxc = dxn / self.std

        # xn = xc * (1/std). 1/std 쪽으로 가는 미분과 std 쪽으로 가는 미분 분리.
        # d(1/std)/dstd = -1/std^2 이므로 dstd = -sum(dxn * xc / std^2)
        dstd = -((dxn * self.xc) / (self.std * self.std)).sum(axis=0)
        # std = sqrt(var + eps) → dvar = (1/2) * dstd / sqrt(var + eps) = 0.5 * dstd / std
        dvar = 0.5 * dstd / self.std

        # var = mean(xc^2) → 각 xc_i의 기여분: (2/N) * xc_i * dvar 를 dxc에 더해줌
        dxc += (2.0 / N) * self.xc * dvar

        # xc = x - mu → dmu = -sum(dxc), dx의 직접 기여분은 dxc 그대로
        # mu = mean(x) → dx에 더해질 mu 경유분: -dmu/N * 1, 즉 +dmu/N(부호 주의)
        # 본 식에서 dmu = sum(dxc, axis=0)로 두면 dx = dxc - dmu / N 형태로 깔끔히 정리됨.
        dmu = dxc.sum(axis=0)
        dx = dxc - dmu / N
        return dx


class Dropout:
    """
    Dropout.

    학습 중 일부 뉴런 출력을 무작위로 0으로 만들어 과적합을 줄입니다.
    이 구현은 추론 시 출력에 (1 - drop_ratio)를 곱하는 기본 dropout 방식을 사용합니다.
    """

    def __init__(self, drop_ratio=0.5):
        """Args: drop_ratio: 학습 중 0으로 만들 뉴런 비율."""
        self.drop_ratio = drop_ratio

    def forward(self, x, train=True):
        """
        Args:
            x: 입력 배열
            train: True면 무작위 mask 적용, False면 평균적인 출력 크기로 scale
        """
        if train:
            # 학습 시: 매 forward마다 새 random mask 생성.
            # np.random.rand(*x.shape): x와 같은 shape, [0, 1) 균등 난수.
            # > drop_ratio: 그 중 drop_ratio 이상인 위치만 True(살아남음).
            # 예: drop_ratio=0.3이면 평균적으로 70%만 True → 30%는 0으로 막힘.
            self.mask = np.random.rand(*x.shape) > self.drop_ratio
            return x * self.mask
        # 추론 시: mask 없이 그대로 통과 + scale 보정.
        # 학습 때는 평균적으로 (1-drop_ratio)만 활성화 → 추론도 같은 평균 크기에 맞추려고 곱해줌.
        return x * (1 - self.drop_ratio)

    def backward(self, dout):
        """forward에서 꺼졌던 뉴런 위치에는 gradient도 흘리지 않습니다."""
        # forward에서 막혀 0으로 나간 위치는 출력에 영향 없음 → gradient도 0.
        # 살아남은 위치에는 1이 곱해져 그대로 통과.
        return dout * self.mask

# -*- coding: utf-8 -*-
"""
활성화 함수 모음.

학생 구현 대상:
- ReLU.forward, ReLU.backward
- Softmax.forward, Softmax.backward
"""

import numpy as np


class ReLU:
    """
    ReLU(Rectified Linear Unit) 활성화 함수.

    은닉층에서 음수 값은 0으로 막고, 양수 값은 그대로 통과시킵니다.
    forward에서 만든 mask는 backward 때 "어느 위치로 gradient를 흘릴지" 결정하는 데 사용됩니다.
    """

    def forward(self, x):
        """
        Args:
            x: 임의 shape의 입력 배열

        Returns:
            x와 같은 shape. x > 0인 위치만 원래 값을 유지합니다.
        """
        # 수식: ReLU(x) = max(0, x). 음수/0 자리에는 0, 양수는 그대로.
        # self.mask: True = "막힐 위치"(x <= 0). backward에서 같은 mask로 gradient 차단.
        self.mask = (x <= 0)
        # 호출자가 넘긴 x를 망가뜨리면 안 되니 copy()로 새 배열을 만든다.
        out = x.copy()
        # mask가 True인 위치만 0으로. NumPy의 boolean indexing 활용.
        out[self.mask] = 0
        return out

    def backward(self, dout):
        """
        Args:
            dout: 다음 층에서 넘어온 gradient

        Returns:
            ReLU 입력 x에 대한 gradient. forward 때 x <= 0이었던 위치는 0입니다.
        """
        # ReLU의 미분: x > 0이면 1, x <= 0이면 0.
        # forward에서 저장해 둔 mask를 그대로 써서 "막힌 위치"의 gradient를 0으로.
        dout[self.mask] = 0
        # dx == dout (mask를 in-place로 적용한 dout이 곧 dx).
        dx = dout
        return dx


class Softmax:
    """
    Softmax 출력층.

    각 샘플의 로짓(logit)을 클래스별 확률로 바꿉니다.
    exp 계산 전에 행별 최댓값을 빼면 큰 숫자에서 overflow가 나는 것을 줄일 수 있습니다.
    """

    def forward(self, x):
        """
        Args:
            x: (batch_size, num_classes) 로짓

        Returns:
            (batch_size, num_classes) 확률. 각 행의 합은 1입니다.
        """
        # 수치 안정성(max-shift): 큰 값에서 exp(x)는 overflow 발생.
        # 모든 행에서 자기 행의 max를 빼면 결과는 같은데(분자/분모에 같은 상수가 곱해져 약분),
        # exp 최대값이 0 이하로 떨어져 overflow가 사라진다.
        # axis=1 → 행 기준(샘플별). keepdims=True → (N,1)로 모양 유지해 브로드캐스팅 가능.
        x_shifted = x - np.max(x, axis=1, keepdims=True)
        # 각 로짓을 exp로 변환 (모두 양수).
        exp_x = np.exp(x_shifted)
        # 각 행의 합으로 나눠 확률화. self.out에 저장(현 프로젝트는 backward에서 안 쓰지만 관습적으로 보존).
        self.out = exp_x / np.sum(exp_x, axis=1, keepdims=True)
        return self.out

    def backward(self, dout):
        """
        Softmax와 Cross Entropy를 함께 미분한 gradient를 train()에서 직접 만들기 때문에
        여기서는 받은 gradient를 그대로 통과시킵니다.
        """
        # Softmax+CE를 합치면 dL/dz = (y_pred - y_true) / N 으로 식이 단순해진다.
        # 이 결합 gradient는 train()에서 직접 만들어 model.backward()에 넣으므로,
        # Softmax 자체는 통과(identity)시키는 것으로 충분.
        return dout

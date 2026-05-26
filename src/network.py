# -*- coding: utf-8 -*-
"""
MNIST 분류용 신경망 조립 모듈.

개별 layer를 OrderedDict에 쌓아 forward/backward 순서를 명확히 유지합니다.
"""

from collections import OrderedDict

import numpy as np

from activations import ReLU, Softmax
from layers import Affine, BatchNorm, Dropout
from losses import cross_entropy_loss


class NeuralNetwork:
    """
    MNIST 분류용 신경망.
    입력 784 -> 은닉층(들) -> 출력 10 (Softmax).
    은닉층 구성: Affine -> BatchNorm -> ReLU -> Dropout (모두 필수)
    가중치 초기화: He 또는 Xavier 중 선택.
    """

    def __init__(self, use_batchnorm=True, use_dropout=True, dropout_ratio=0.5):
        """
        구조: 입력 784 -> [Affine -> (BatchNorm) -> ReLU -> (Dropout)] * 2 -> Affine(10) -> Softmax.
        가중치는 He 초기화.
        """
        sizes = [784, 512, 256, 10]
        self.params = {}
        self.layers = OrderedDict()

        n_layers = len(sizes) - 1
        for i in range(n_layers):
            in_size, out_size = sizes[i], sizes[i + 1]
            W = np.random.randn(in_size, out_size) * np.sqrt(2.0 / in_size)
            b = np.zeros(out_size)
            self.params[f"W{i + 1}"] = W
            self.params[f"b{i + 1}"] = b
            self.layers[f"Affine{i + 1}"] = Affine(
                self.params[f"W{i + 1}"], self.params[f"b{i + 1}"]
            )

            if i < n_layers - 1:
                if use_batchnorm:
                    self.params[f"gamma{i + 1}"] = np.ones(out_size)
                    self.params[f"beta{i + 1}"] = np.zeros(out_size)
                    self.layers[f"BatchNorm{i + 1}"] = BatchNorm(
                        self.params[f"gamma{i + 1}"], self.params[f"beta{i + 1}"]
                    )
                self.layers[f"ReLU{i + 1}"] = ReLU()
                if use_dropout:
                    self.layers[f"Dropout{i + 1}"] = Dropout(dropout_ratio)

        self.last_layer = Softmax()
        self.grads = {}

    def forward(self, x, train=True):
        """
        Args:
            x: (batch_size, 784) 정규화된 MNIST 이미지
            train: BatchNorm/Dropout의 학습 모드 여부

        Returns:
            (batch_size, 10) 각 숫자 클래스의 확률
        """
        for layer in self.layers.values():
            if isinstance(layer, (BatchNorm, Dropout)):
                x = layer.forward(x, train=train)
            else:
                x = layer.forward(x)
        return self.last_layer.forward(x)

    def backward(self, dout):
        """
        네트워크 전체 역전파를 수행하고 self.grads를 채웁니다.

        Args:
            dout: Softmax+CrossEntropy를 합친 출력층 gradient
        """
        dout = self.last_layer.backward(dout)
        for layer in reversed(self.layers.values()):
            dout = layer.backward(dout)

        self.grads = {}
        for name, layer in self.layers.items():
            if isinstance(layer, Affine):
                idx = name.replace("Affine", "")
                self.grads[f"W{idx}"] = layer.dW
                self.grads[f"b{idx}"] = layer.db
            elif isinstance(layer, BatchNorm):
                idx = name.replace("BatchNorm", "")
                self.grads[f"gamma{idx}"] = layer.dgamma
                self.grads[f"beta{idx}"] = layer.dbeta

    def loss(self, x, y):
        """현재 모델의 예측 확률을 만든 뒤 cross entropy loss를 반환합니다."""
        y_pred = self.forward(x, train=True)
        return cross_entropy_loss(y_pred, y)

    def predict(self, x):
        """추론 모드로 확률을 예측합니다. BatchNorm/Dropout은 train=False로 동작합니다."""
        return self.forward(x, train=False)

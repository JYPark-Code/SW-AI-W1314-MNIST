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
        # 입력층 + 은닉층(2개) + 출력층 크기. sizes[i] → sizes[i+1] 변환마다 Affine 한 개.
        sizes = [784, 512, 256, 10]
        # params: optimizer가 갱신할 모든 학습 파라미터를 모아둔 dict.
        self.params = {}
        # layers: forward 순서대로 보존되는 layer dict (역순 backward도 reversed로 가능).
        self.layers = OrderedDict()

        # 총 Affine 개수 = 입력→은닉1, 은닉1→은닉2, 은닉2→출력 = 3개.
        n_layers = len(sizes) - 1
        for i in range(n_layers):
            in_size, out_size = sizes[i], sizes[i + 1]

            # He 초기화: ReLU에서 음수의 절반이 0으로 막히는 손실을 보상하려고 분산을 2배(=sqrt(2/fan_in))로.
            # Sigmoid/tanh라면 Xavier(sqrt(1/fan_in))가 표준.
            W = np.random.randn(in_size, out_size) * np.sqrt(2.0 / in_size)
            b = np.zeros(out_size)
            self.params[f"W{i + 1}"] = W
            self.params[f"b{i + 1}"] = b
            # 중요: Affine에 self.params[...]를 그대로 넘김 → 같은 ndarray를 공유.
            # optimizer가 params[key]를 in-place로 갱신하면 Affine의 self.W도 자동 반영.
            self.layers[f"Affine{i + 1}"] = Affine(
                self.params[f"W{i + 1}"], self.params[f"b{i + 1}"]
            )

            # 마지막 Affine 뒤에는 BatchNorm/ReLU/Dropout 모두 안 붙임 (출력은 Softmax로 바로 변환).
            if i < n_layers - 1:
                if use_batchnorm:
                    # gamma는 1, beta는 0으로 초기화 → 처음엔 정규화 결과를 그대로 통과.
                    # 학습이 진행되며 scale/shift가 데이터에 맞게 조정됨.
                    self.params[f"gamma{i + 1}"] = np.ones(out_size)
                    self.params[f"beta{i + 1}"] = np.zeros(out_size)
                    self.layers[f"BatchNorm{i + 1}"] = BatchNorm(
                        self.params[f"gamma{i + 1}"], self.params[f"beta{i + 1}"]
                    )
                # 은닉 블록의 비선형성. ReLU는 학습 파라미터 없음 → params dict에 추가할 게 없다.
                self.layers[f"ReLU{i + 1}"] = ReLU()
                if use_dropout:
                    # Dropout도 학습 파라미터 없음 (mask는 매 forward마다 새로 만듦).
                    self.layers[f"Dropout{i + 1}"] = Dropout(dropout_ratio)

        # 출력층: Softmax. 학습 파라미터 없음.
        self.last_layer = Softmax()
        # backward 후에 채워지는 gradient dict (params와 같은 key 집합).
        self.grads = {}

    def forward(self, x, train=True):
        """
        Args:
            x: (batch_size, 784) 정규화된 MNIST 이미지
            train: BatchNorm/Dropout의 학습 모드 여부

        Returns:
            (batch_size, 10) 각 숫자 클래스의 확률
        """
        # OrderedDict라 순서대로 통과. x를 매 layer의 출력으로 갱신.
        for layer in self.layers.values():
            # BatchNorm/Dropout만 train 인자가 필요(학습/추론 동작이 다름).
            # 나머지 layer(Affine, ReLU)는 train 모드 개념이 없음.
            if isinstance(layer, (BatchNorm, Dropout)):
                x = layer.forward(x, train=train)
            else:
                x = layer.forward(x)
        # 마지막에 Softmax로 확률 변환. last_layer는 layers에 안 넣음
        # → backward에서 따로 처리 + Softmax+CE 합산 gradient를 받기 위함.
        return self.last_layer.forward(x)

    def backward(self, dout):
        """
        네트워크 전체 역전파를 수행하고 self.grads를 채웁니다.

        Args:
            dout: Softmax+CrossEntropy를 합친 출력층 gradient
        """
        # Softmax.backward는 통과(identity)지만 인터페이스 일관성을 위해 호출.
        dout = self.last_layer.backward(dout)
        # 은닉 layer들을 역순으로 통과. 각 layer가 받은 dout으로 자신의 gradient를 만들며 dx를 다음으로 전달.
        for layer in reversed(self.layers.values()):
            dout = layer.backward(dout)

        # backward로 각 layer 안에 모인 gradient(self.dW, self.dbeta 등)를
        # optimizer가 쓰기 좋은 dict 형태로 모아준다.
        self.grads = {}
        for name, layer in self.layers.items():
            if isinstance(layer, Affine):
                # "Affine1" → "1"로 인덱스 추출.
                idx = name.replace("Affine", "")
                self.grads[f"W{idx}"] = layer.dW
                self.grads[f"b{idx}"] = layer.db
            elif isinstance(layer, BatchNorm):
                idx = name.replace("BatchNorm", "")
                self.grads[f"gamma{idx}"] = layer.dgamma
                self.grads[f"beta{idx}"] = layer.dbeta
            # ReLU/Dropout은 학습 파라미터가 없으니 grads에 추가하지 않는다.

    def loss(self, x, y):
        """현재 모델의 예측 확률을 만든 뒤 cross entropy loss를 반환합니다."""
        y_pred = self.forward(x, train=True)
        return cross_entropy_loss(y_pred, y)

    def predict(self, x):
        """추론 모드로 확률을 예측합니다. BatchNorm/Dropout은 train=False로 동작합니다."""
        return self.forward(x, train=False)

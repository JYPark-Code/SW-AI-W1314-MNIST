# -*- coding: utf-8 -*-
"""
MNIST 분류용 신경망 조립 모듈.

개별 layer를 OrderedDict에 쌓아 forward/backward 순서를 명확히 유지합니다.
"""

from collections import OrderedDict
from collections import defaultdict

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
        Args:
            use_batchnorm: 은닉층마다 BatchNorm을 넣을지 여부
            use_dropout: 은닉층마다 Dropout을 넣을지 여부
            dropout_ratio: Dropout에서 끌 뉴런 비율
        """
        # TODO: params dict를 만들고 Affine/BatchNorm/ReLU/Dropout layer를 순서대로 구성하세요.
        # 권장 구조: 784 -> 512 -> 256 -> 10
        # self.layers는 OrderedDict로 만들고, self.grads는 params와 같은 key를 갖게 합니다.
        
        self.softmax = Softmax()

        # params 설정
        self.params = {}
        self.params['W1'] = np.random.randn(784, 512) * np.sqrt(2/784) 
        self.params['b1'] = np.zeros(512)
        self.params['W2'] = np.random.randn(512, 256) * np.sqrt(2/512)
        self.params['b2'] = np.zeros(256)
        self.params['W3'] = np.random.randn(256, 10) * np.sqrt(2/256)
        self.params['b3'] = np.zeros(10)

        # grads 설정
        self.grads = defaultdict(dict)

        # 레이어 모음
        self.layers = OrderedDict()

        # 첫번째 계층 설정
        self.layers['L1'] = OrderedDict()
        self.layers['L1']['affine'] = Affine(self.params['W1'], self.params['b1'])
        if use_batchnorm:
            self.layers['L1']['batchnorm'] = BatchNorm(1, 0)
        self.layers['L1']['activation'] = ReLU()
        if use_dropout:
            self.layers['L1']['dropout'] = Dropout(dropout_ratio)
        
        # 두번째 계층 설정
        self.layers['L2'] = OrderedDict()
        self.layers['L2']['affine'] = Affine(self.params['W2'], self.params['b2'])
        if use_batchnorm:
            self.layers['L2']['batchnorm'] = BatchNorm(1, 0)
        self.layers['L2']['activation'] = ReLU()
        if use_dropout:
            self.layers['L2']['dropout'] = Dropout(dropout_ratio)

        # 세번째 계층 설정
        self.layers['L3'] = OrderedDict()
        self.layers['L3']['affine'] = Affine(self.params['W3'], self.params['b3'])

    def forward(self, x, train=True):
        """
        Args:
            x: (batch_size, 784) 정규화된 MNIST 이미지
            train: BatchNorm/Dropout의 학습 모드 여부

        Returns:
            (batch_size, 10) 각 숫자 클래스의 확률
        """
        # TODO: self.layers를 순서대로 통과시키고 마지막에 Softmax를 적용하세요.
        for layers in self.layers.values():
            for layer in layers.values():
                if isinstance(layer, (BatchNorm, Dropout)):
                    x = layer.forward(x, train)
                else:
                    x = layer.forward(x)
        
        if train:
            return self.softmax.forward(x)
        else:
            return x

    def backward(self, dout):
        """
        네트워크 전체 역전파를 수행하고 self.grads를 채웁니다.

        Args:
            dout: Softmax+CrossEntropy를 합친 출력층 gradient
        """
        if not dout.shape:
            dout = dout.copy().reshape(1,)

        # TODO: layer를 역순으로 통과시키고 Affine/BatchNorm의 gradient를 self.grads에 모으세요.
        for layers in reversed(self.layers.values()):
            for layer in reversed(layers.values()):
                dout = layer.backward(dout)
    
        self.grads['W1'] = self.layers['L1']['affine'].dW
        self.grads['b1'] = self.layers['L1']['affine'].db

        self.grads['W2'] = self.layers['L2']['affine'].dW
        self.grads['b2'] = self.layers['L2']['affine'].db

        self.grads['W3'] = self.layers['L3']['affine'].dW
        self.grads['b3'] = self.layers['L3']['affine'].db

    def loss(self, x, y):
        """현재 모델의 예측 확률을 만든 뒤 cross entropy loss를 반환합니다."""
        y_pred = self.forward(x, train=True)
        return cross_entropy_loss(y_pred, y)

    def predict(self, x):
        """추론 모드로 확률을 예측합니다. BatchNorm/Dropout은 train=False로 동작합니다."""
        return self.forward(x, train=False)

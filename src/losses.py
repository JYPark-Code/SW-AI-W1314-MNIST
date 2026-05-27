# -*- coding: utf-8 -*-
"""손실 함수 모음."""

import numpy as np


def cross_entropy_loss(y_pred, y_true):
    """
    Cross Entropy Error (배치 평균).
    y_pred: (batch_size, 10) 확률
    y_true: (batch_size,) 정수 레이블 0~9
    """
    # Cross Entropy: L = -1/N * sum_i log(y_pred[i, t_i]) — 정답 클래스 확률만 log.
    # (One-hot 레이블이라면 정답 위치만 1이고 나머지는 0이라, 결국 정답 항만 살아남는다.)
    batch_size = y_pred.shape[0]
    # Fancy indexing: 각 행 i에서 y_true[i] 위치 확률만 뽑아 (N,) 벡터 생성.
    # 예) y_pred.shape=(N,10), y_true.shape=(N,) → 결과 shape=(N,)
    correct_probs = y_pred[np.arange(batch_size), y_true]
    # log(0) = -inf 방지. epsilon 1e-12로 하한 클립 (너무 크게 잡으면 손실이 부정확해짐).
    correct_probs = np.clip(correct_probs, 1e-12, 1.0)
    # 각 샘플의 -log(정답 확률)을 모두 평균 → 배치 평균 loss(스칼라).
    return -np.mean(np.log(correct_probs))

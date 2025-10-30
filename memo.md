#### idea

- SAR, EO이미지에 대해서 Wavlet 변환을 해서 저주파, 고주파 성분을 분석해서 일치하면, 이건 temporal mismatch에 의한거고, 다 같다면 그냥 모델이 어려워 하는거임

--> 모델을 통과한 결과인 OPT이미지와 GT OPT이미지를 주파수 분석해서 만약 confidence가 낮고, temporal mismatch가 있다면 이건 무시하고, 고주파 정보는 가중치 높게


warmup: uncertainty map을 예측하는 능력 기르기 -> C-DiffSet 방법 (안정적인 학습)
w_uncertainty가 높은 곳은 loss가중치 낮춤

--> 그 후 hard한 부분 집중 학습 5K iteration.

Case A: temporal mismatch
TMM > threshold 로 TMM이 있는 곳을 구함 -> 이 부분에 대해서는 loss scale줄여야함

WVT(SYN OPT), WVT(GT OPT)를 비교해서 동일한 위치에 고주파 정보가 없다면, w_uncertainty값을 확 줄여버림.

Case B: high frequency, low frequency detail

L2 * w_uncertainty



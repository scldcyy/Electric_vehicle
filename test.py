import numpy as np
from pymoo.indicators.hv import HV

# def calculate_hv(front, ref_point):
#     # 计算参考点到前沿集合的距离
#     distances = []
#     for point in front:
#         distance = np.sqrt(np.sum((ref_point - point) ** 2))
#         distances.append(distance)
#
#     # 计算前沿集合的体积
#     volume = 1
#     for i in range(len(ref_point)):
#         values = [point[i] for point in front]
#         max_value = max(values)
#         min_value = min(values)
#         volume *= (max_value - min_value)
#
#     # 计算HV指标
#     hv = volume * sum(distances) / len(distances)
#
#     return hv


front = np.array([[1, 2], [2, 1]])
ref_point = np.array([4, 4])
ind = HV(ref_point=ref_point)
print("HV", ind(front))

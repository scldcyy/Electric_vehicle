from bisect import bisect_left

import matplotlib.pyplot as plt
import copy
import random
import json

# BSS信息
import numpy as np
import pandas as pd

Inital_battery_Soc = [[0.92, 1, 1, 1, 0.8], [0.52, 0.8, 0.6, 0.92, 0.52, 0.92, 0.62, 1, 0.6, 0.8, 1, 0.87, 0.93],
                      [1, 0.93, 1, 1, 0.73], [0.6, 0.73, 0.93, 1, 0.87, 0.67, 1, 0.8, 0.73, 0.87, 0.67, 0.6, 0.53],
                      [0.73, 0.67, 1, 0.6, 0.53, 0.67, 0.87, 0.8, 0.73, 0.87, 0.6, 0.93, 0.53]]
Inital_battery_Soc_copy = copy.deepcopy(Inital_battery_Soc)
# EV信息
N = 20
# EV的SoC相关信息 SoC_zeros/SoC_hats/SoC_thrs
EV_SoC = [[0.25, 0.16, 0.8],
          [0.3, 0.19, 0.85],
          [0.26, 0.15, 0.8],
          [0.22, 0.15, 0.8],
          [0.275, 0.18, 0.9],
          [0.38, 0.25, 0.75],
          [0.4, 0.28, 0.7],
          [0.265, 0.16, 0.75],
          [0.395, 0.27, 0.75],
          [0.37, 0.26, 0.85],
          [0.35, 0.23, 0.8],
          [0.28, 0.18, 0.9],
          [0.35, 0.22, 0.75],
          [0.37, 0.25, 0.9],
          [0.4, 0.27, 0.85],
          [0.22, 0.15, 0.75],
          [0.38, 0.26, 0.8],
          [0.39, 0.26, 0.75],
          [0.31, 0.2, 0.8],
          [0.27, 0.15, 0.75]]

# BSS与EV的联合信息  km单位
l_n_k = [[17, 20, 15, 25, 5],
         [24, 17, 20, 15, 7],
         [20, 15, 17, 13, 6],
         [16, 20, 13, 7, 10],
         [15, 5, 9, 20, 13],
         [20, 13, 12, 15, 7],
         [15, 20, 11, 7, 13],
         [16, 23, 16, 12, 10],
         [8, 20, 18, 25, 13],
         [20, 13, 15, 5, 10],
         [11, 13, 23, 12, 15],
         [13, 7, 15, 20, 24],
         [16, 22, 12, 11, 10],
         [18, 15, 17, 10, 5],
         [12, 21, 15, 8, 14],
         [26, 10, 15, 10, 12],
         [15, 11, 23, 14, 10],
         [15, 23, 12, 10, 8],
         [13, 14, 22, 7, 5],
         [7, 6, 8, 15, 14]]
K = 5
JK = np.array([5, 13, 5, 13, 13])
sum_JK = np.sum(JK)
cum_Jk = np.append(0, np.cumsum(JK))
# 其他信息
# 速度单位km/h
S_a = 50
# 额定电量kWh
E_n = 75
# 充电率 kW/min
rho = 1.25
# 电网价格 $/kWh
p_grid = 0.85
# 系数 $/km
tao = 0.6
# 时间间隔 min
deta_t = 10
# alpha
alpha = 0.5
# belta
belta = 0.5
delta_yita_a = 7.344 * 1e-5 * S_a ** 2 - 7.656 * 1e-3 * S_a + 0.3536
Lkavi_pre = np.array([np.sum(Inital_battery_Soc[k] * E_n) for k in range(K)])
Rk = 12.5 * JK
Lk_max = JK * E_n


def f(x):  # 目标函数
    Lkavi_cur = Lkavi_pre + Rk
    enk = np.zeros(N)
    for n in range(N):
        # 到车站约束 Soc_nk>=SoC_hats
        k, j = x[n]
        Soc_nk = EV_SoC[n][0] - l_n_k[n][k] * delta_yita_a / E_n
        # 到目的地约束 SoC_init[x[i]]>=SoC_thrs
        en = (Inital_battery_Soc[k][j] - Soc_nk) * E_n
        Lkavi_cur[k] -= en
        enk[n] = en
    pk = p_grid + p_grid * (1 - Lkavi_cur / Lk_max)
    ret = []
    for n in range(N):
        k, j = x[n]
        tmp = alpha * enk[n] * pk[k] + belta * tao * l_n_k[n][k]
        ret.append(tmp)
    return np.array(ret)


def f2(x):  # 目标函数
    Lkavi_cur = Lkavi_pre + Rk
    enk = np.zeros(N)
    ret = []
    for n in range(N):
        # 到车站约束 Soc_nk>=SoC_hats
        k, j = x[n]
        Soc_nk = EV_SoC[n][0] - l_n_k[n][k] * delta_yita_a / E_n
        # 到目的地约束 SoC_init[x[i]]>=SoC_thrs
        en = (Inital_battery_Soc[k][j] - Soc_nk) * E_n
        Lkavi_cur[k] -= en
        enk[n] = en
        pk = p_grid + p_grid * (1 - Lkavi_cur[k] / Lk_max[k])
        tmp = alpha * enk[n] * pk + belta * tao * l_n_k[n][k]
        ret.append(tmp)
    return np.array(ret)


def init_strategy():
    # SoC_zeros / SoC_hats / SoC_thrs
    bts = []
    for i in range(len(JK)):
        for j in range(JK[i]):
            bts.append((i, j))
    random.shuffle(bts)
    # bts = json.load(open('bts.json', 'r'))
    Q = [[] for _ in range(N)]
    for ev in range(N):
        for bt in bts:
            k, j = bt
            Soc_nk = EV_SoC[ev][0] - l_n_k[ev][k] * delta_yita_a / E_n
            enk = (Inital_battery_Soc[k][j] - Soc_nk) * E_n
            if Soc_nk >= EV_SoC[ev][1] and Inital_battery_Soc[k][j] >= EV_SoC[ev][2]:
                Q[ev] = [ev, k, j, enk]
                bts.remove(bt)
                break
    return Q


def update_BSS(Q, k):
    # 算法2
    Lkavi_cur = Lkavi_pre[k] + Rk[k]
    for n, kq, j, enk in Q:
        if kq == k:
            Lkavi_cur -= enk
    omega_k = p_grid * (1 - Lkavi_cur / Lk_max[k])
    pkt = p_grid + omega_k
    return pkt


def update_EV(Q, n, pKt):
    # 算法1
    _, kq, jq, enkq = Q[n]
    fnk_para = []  # fnk,k,j,enk
    # for k in random.sample(range(K), K):
    for k in range(K):
        Soc_nk = EV_SoC[n][0] - l_n_k[n][k] * delta_yita_a / E_n
        if k == kq:
            fnk = alpha * enkq * pKt[k] + belta * tao * l_n_k[n][k]
            fnk_para.append([fnk, k, jq, enkq])
            continue
        elif Soc_nk >= EV_SoC[n][1]:
            pi_j_n = [q[2] for q in Q if q[1] == k]
            # for j in random.sample(range(JK[k]), JK[k]):
            for j in range(JK[k]):
                if j not in pi_j_n and Inital_battery_Soc[k][j] >= EV_SoC[n][2]:
                    enk = (Inital_battery_Soc[k][j] - Soc_nk) * E_n
                    fnk = alpha * enk * pKt[k] + belta * tao * l_n_k[n][k]
                    fnk_para.append([fnk, k, j, enk])
                    break
    select = min(fnk_para, key=lambda x: x[0])
    if kq != select[1]:
        # pKt[kq] = update_BSS(Q, kq)
        Q[n] = [n] + select[1:]
        pKt[kq] = update_BSS(Q, kq)
    return Q, select, pKt


def cal_f_n_k(n, k, p_k, e_n_k):
    return alpha * e_n_k * p_k[k] + belta * tao * l_n_k[n][k]


def NE_Seeking(I=100, save_path=""):
    # 算法3
    Q = json.load(open(f'init_Qs/{save_path}.json', 'r'))
    print("初始策略：", sum(f([[q[1], q[2]] for q in Q])))
    costs = [[] for _ in range(N)]
    pKt = np.zeros(K)
    for k in range(K):
        pKt[k] = update_BSS(Q, k)
    shuffle_n = random.sample(range(N), N)
    for i in range(I):
        # for n in random.sample(range(N), N):
        seq_n = range(N) if i < 0 else shuffle_n
        # seq_n = range(N) if i < I * 2 else shuffle_n
        for n in seq_n:
            Q, select, pKt = update_EV(Q, n, pKt)
            costs[n].append(select[0])
            k = Q[n][1]
            pKt[k] = update_BSS(Q, k)
    for n in range(N):
        costs[n].append(cal_f_n_k(n, Q[n][1], pKt, Q[n][3]))
    return Q, costs, shuffle_n


def draw(costs, save_path=""):
    for c in costs:
        plt.plot(c)
    plt.title(f'{save_path}-sum_cost={sum([c[-1] for c in costs])}')
    plt.xlabel("iteration")
    plt.ylabel("cost")
    plt.ylim([10, 50])
    plt.savefig(f'NE_imgs/{save_path}.png')
    plt.close()


def save_table(Q, costs, save_path):
    Q_data = pd.DataFrame(Q)
    Q_data.to_excel(f'table/Q_{save_path}.xlsx')
    costs_data = pd.DataFrame(costs)
    costs_data.to_excel(f'table/costs_{save_path}.xlsx')


exp_name = "evs_NE_shuffle"
for i in range(10):
    Q, costs, shuffle_n = NE_Seeking(save_path=f"init_Q{i}")
    draw(costs, save_path=f"init_Q{i}_{exp_name}_{shuffle_n}")
    save_table(Q, costs, save_path=f"init_Q{i}_{exp_name}_{shuffle_n}")
    print(Q)
    print("迭代结束：", sum([c[-1] for c in costs]))
cum_Jk = np.append(0, np.cumsum(JK))
ans = [cum_Jk[q[1]] + q[2] for q in Q]
print(ans)
# BSS_num = [0 for i in range(K)]
# for q in Q:
#     BSS_num[q[1]] += 1
# print(BSS_num)

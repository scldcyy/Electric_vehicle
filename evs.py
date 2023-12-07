import matplotlib.pyplot as plt
import copy
import random
import json

# BSS信息
import numpy as np

import pandas as pd

K = 5
JK = [5, 13, 5, 13, 13]
Inital_battery_Soc = [[0.92, 1, 1, 1, 0.8], [0.52, 0.8, 0.6, 0.92, 0.52, 0.92, 0.62, 1, 0.6, 0.8, 1, 0.87, 0.93],
                      [1, 0.93, 1, 1, 0.73], [0.6, 0.73, 0.93, 1, 0.87, 0.67, 1, 0.8, 0.73, 0.87, 0.67, 0.6, 0.53],
                      [0.73, 0.67, 1, 0.6, 0.53, 0.67, 0.87, 0.8, 0.73, 0.87, 0.6, 0.93, 0.53]]

# EV信息
N = 20
# EV的SoC相关信息
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


def SoC_n_k(n, k):
    # tmp=(7.344 * 1e-5 * S_a ** 2 - 7.656 * 1e-3 * S_a + 0.3536)
    eta = l_n_k[n][k] * (7.344 * 1e-5 * S_a ** 2 - 7.656 * 1e-3 * S_a + 0.3536)
    E_n_k = E_n * EV_SoC[n][0] - eta
    return E_n_k / E_n


def cal_e_n_k(n, k, j):
    # n车行驶到k的能耗
    soc_n_k = SoC_n_k(n, k)

    e_n_k = (Inital_battery_Soc[k][j] - soc_n_k) * E_n
    # print(e_n_k)
    return e_n_k


def subtract_lists(list1, list2):
    result = []
    for element in list1:
        if element not in list2:
            result.append(element)
    return result


def NE_Seeking(K, N, S_a, E_n, rho, p_grid, tao, deta_t, save_path):
    # 初始化每台车n的决策C,Q
    # 读取初始策略配置文件
    # with open("init_strategy.json", "r", encoding="utf-8") as f:
    #     content = json.load(f)
    #
    # Q = [[0, 0, 0, 0] for _ in range(N)]
    # select = []
    # for n in range(N):
    #     Q[n][0] = n
    #     result = subtract_lists(content[f"{n}"], select)
    #     strategy = random.sample(result, 1)
    #     Q[n][1] = strategy[0][0]
    #     Q[n][2] = strategy[0][1]
    #     Q[n][3] = cal_e_n_k(n, Q[n][1], Q[n][2])
    Q = json.load(open(f'init_Qs/{save_path}.json', 'r'))

    #  k BSS的当前总电价
    p_k = [0 for i in range(K)]

    # 一开始更新电池电量
    def init_BSS(k):
        """
        :param N:
        :param k: 更新编号为k的 BSS
        :return:  返回编号为k的BSS的单位千瓦时的美金价格
        """

        for n in range(N):
            select_k = Q[n][1]
            j = Q[n][2]
            e_n_k = Q[n][3]
            # print(e_n_k)
            # 如果n车选择的就是k BSS
            if select_k == k:
                Inital_battery_Soc[k][j] -= (e_n_k / E_n)

        # print(k,Inital_battery_Soc[k])

        L_avi = sum(Inital_battery_Soc[k]) * E_n + rho * deta_t * JK[k]
        w = p_grid * (1 - L_avi / (JK[k] * E_n))

        p = p_grid + w
        p_k[k] = p

    for k in range(K):
        init_BSS(k)

    # 计算f_n_k
    def cal_f_n_k(n, k, j, e_n_k):
        # print("asdfg:", cal_e_n_k(n,k,j),p_k[k])
        return alpha * e_n_k * p_k[k] + belta * tao * l_n_k[n][k]

    # 定义恢复电量算法
    def recovery(k, j, e_n_k):
        Inital_battery_Soc[k][j] += e_n_k / E_n

        L_avi = sum(Inital_battery_Soc[k]) * E_n + rho * deta_t * JK[k]
        w = p_grid * (1 - L_avi / (JK[k] * E_n))

        p = p_grid + w
        p_k[k] = p

    # 用于记录每一次迭代的cost结果
    cost = [[] for i in range(N)]

    jk = [0 for i in range(len(JK))]
    for i in range(len(JK)):
        jk[i] = [j for j in range(JK[i])]
        # random.shuffle(jk[i])
    print(jk)

    # 定义算法1
    def EV(n):
        """
        :param n: 车子编号
        :return:
        """
        # n车辆上一轮的选择 type:[n,k,v,e_n_k]
        Q_n = Q[n]

        # 车子n选择所有k的代价
        f_n = [float('inf')] * K
        new_Q_n = [0] * K

        # 获取所有电池的SoC情况
        last_choose = Q_n[1]

        # global K
        # 遍历所有K个BSS
        for k in range(K):
            if k == last_choose:
                j = Q_n[2]
                e_n_k = Q_n[3]
                f_n[k] = cal_f_n_k(n, k, j, e_n_k)
                new_Q_n[k] = [n, k, j, e_n_k]
                continue
            elif SoC_n_k(n, k) > EV_SoC[n][1]:
                flag = False
                # 查找没人使用的j以及j满足车n的心理预期
                pi_j_n = [each[2] for each in Q if each[1] == k]
                # print(f"前人所选BSS{k}的所有电池j编号：", pi_j_n)

                # 打乱j
                ls_jk = [j for j in range(JK[k])]
                random.shuffle(ls_jk)

                # # 电量从小到大排序
                # def sort_indices(ls_jk):
                #     # Enumerate the list to get pairs of (index, value), sort by value, and then extract the indices
                #     return [index for index, value in sorted(enumerate(ls_jk), key=lambda pair: pair[1])]
                #
                # ls_jk = copy.deepcopy(Inital_battery_Soc[k])
                # ls_jk = sort_indices(ls_jk)

                for j in range(JK[k]):
                    # print("答复",k,j,Inital_battery_Soc[k][j],EV_SoC[n][2])
                    if Inital_battery_Soc[k][j] >= EV_SoC[n][2] and j not in pi_j_n:
                        e_n_k = cal_e_n_k(n, k, j)
                        f_n[k] = cal_f_n_k(n, k, j, e_n_k)
                        new_Q_n[k] = [n, k, j, e_n_k]
                        flag = True
                        break
                if not flag:
                    f_n[k] = float('inf')
                    new_Q_n[k] = None
        k = f_n.index(min(f_n))
        # print(last_choose,f_n)
        cost[n].append(min(f_n))

        # 回复电量与电价
        recovery(Q_n[1], Q_n[2], Q_n[3])
        if k != last_choose:
            # 更新Q
            Q[n] = copy.deepcopy(new_Q_n[k])
        # print(f"车辆{n}的选择结果：",Q[n][1],Q[n][2],Q[n][3])

    # 算法2 根据新变动的车子的策略去更新电池信息以及电价
    def BSS(n, k):
        """
        :param N:
        :param k: 更新编号为k的 BSS
        :return:  返回编号为k的BSS的单位千瓦时的美金价格
        """
        j = Q[n][2]
        e_n_k = Q[n][3]
        # n车选择的就是k BSS
        Inital_battery_Soc[k][j] -= (e_n_k / E_n)

        print(k,Inital_battery_Soc[k])
        L_avi = sum(Inital_battery_Soc[k]) * E_n + rho * deta_t * JK[k]
        w = p_grid * (1 - L_avi / (JK[k] * E_n))

        p = p_grid + w
        p_k[k] = p
        # recovery(k, j, e_n_k)

    iteration = 50

    for i in range(iteration):
        # # 打乱n
        # ls_N = [j for j in range(N)]
        # random.shuffle(ls_N)
        for n in range(N):
            # print(Q)
            # 调用算法1更新Q
            EV(n)

            # 调用算法2，更新价格和电池信息
            k = Q[n][1]
            BSS(n, k)
        print(i)
    for n in range(N):
        cost[n].append(cal_f_n_k(n, Q[n][1], p_k, Q[n][3]))
    # print("BSS电价：",p_k)

    # 可视化所有cost
    for c in cost:
        plt.plot(c)
    plt.title(f'{save_path}_evs-sum_cost={sum([c[-1] for c in cost])}')
    plt.xlabel("iteration")
    plt.ylabel("cost")
    plt.ylim([10, 50])
    plt.savefig(f'NE_imgs/{save_path}_evs.png')
    plt.close()
    return Q, [c[-1] for c in cost]


costs = []
for i in range(10):
    Inital_battery_Soc_copy = copy.deepcopy(Inital_battery_Soc)
    Q, cost = NE_Seeking(K, N, S_a, E_n, rho, p_grid, tao, deta_t, save_path=f"init_Q{i}")
    costs.append(cost)
    print(Q)
    Inital_battery_Soc = Inital_battery_Soc_copy
data = pd.DataFrame(costs)
data.to_excel('sss.xlsx')
cum_Jk = np.append(0, np.cumsum(JK))
ans = [cum_Jk[q[1]] + q[2] for q in Q]
print(ans)
BSS_num = [0 for i in range(K)]
for q in Q:
    BSS_num[q[1]] += 1
print(BSS_num)

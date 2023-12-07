import collections
import json
import random
from bisect import bisect_left
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

# 参数初始化
# np.random.seed(0)
N = 20
K = 5
alpha = 0.5
beta = 0.5
tao = 0.6
SoC_zeros = np.array([25, 30, 26, 22, 27.5, 38, 40, 26.5, 39.5, 37, 35, 28, 35, 37, 40, 22, 38, 39, 31, 27]) / 100
SoC_hats = np.array([16, 19, 15, 15, 18, 25, 28, 16, 27, 26, 23, 18, 22, 25, 27, 15, 26, 26, 20, 15]) / 100
SoC_thrs = np.array([80, 85, 80, 80, 90, 75, 70, 75, 75, 85, 80, 90, 75, 90, 85, 75, 80, 75, 80, 75]) / 100
Jk = np.array([5, 13, 5, 13, 13])
cum_Jk = np.append(0, np.cumsum(Jk))
sum_Jk = np.sum(Jk)
sa = 50
delta_yita_a = 7.344 * 1e-5 * sa ** 2 - 7.656 * 1e-3 * sa + 0.3536
# SoC_init = np.random.uniform(50, 100, sum_Jk ) / 100  # 随机初始电量
SoC_init = np.array([0.92, 1, 1, 1, 0.8, 0.52, 0.8, 0.6, 0.92, 0.52, 0.92, 0.62, 1, 0.6, 0.8, 1, 0.87, 0.93,
                     1, 0.93, 1, 1, 0.73, 0.6, 0.73, 0.93, 1, 0.87, 0.67, 1, 0.8, 0.73, 0.87, 0.67, 0.6, 0.53,
                     0.73, 0.67, 1, 0.6, 0.53, 0.67, 0.87, 0.8, 0.73, 0.87, 0.6, 0.93, 0.53])
# ln = np.random.uniform(3, 30, (N, K))  # 随机距离
ln = np.array([[17, 20, 15, 25, 5],
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
               [7, 6, 8, 15, 14]])
En = 75
Lk_max = Jk * En
Rk = 12.5  # rho * delta_t=1.25*10
p_grid = 0.85
Lkavi_pre = np.array([np.sum(SoC_init[cum_Jk[k - 1]:cum_Jk[k]] * En) for k in range(1, K + 1)])


def is_dominate(arr1, arr2):
    """
    判断arr1是否支配arr2。这里arr表示个体的多个目标函数适应值组成的向量

    :param arr1: 向量1
    :param arr2: 向量2
    :return: 是否支配
    """
    return np.all(arr1 <= arr2) and np.any(arr1 != arr2)


def split_kj(kj):
    """
    将Jk分解为电站k与电池j
    :param kj: 电池总序号
    :return: k,j
    """
    kj += 1
    k = bisect_left(cum_Jk, kj) - 1
    j = kj - cum_Jk[k] - 1
    return k, j


def f(x):  # 目标函数
    """
    返回每个EV的cost，以及相应的惩罚
    :param x:
    :return:
    """
    Lkavi_cur = Lkavi_pre + Rk
    enk = np.zeros(N)
    punnishs = np.zeros(N)
    for n in range(N):
        # 到车站约束 Soc_nk>=SoC_hats
        k, _ = split_kj(x[n])
        Soc_nk = SoC_zeros[n] - ln[n, k] * delta_yita_a / En
        # 到目的地约束 SoC_init[x[i]]>=SoC_thrs
        en = (SoC_init[x[n]] - Soc_nk) * En
        Lkavi_cur[k] -= en
        enk[n] = en
        punnishs[n] = 0 if Soc_nk >= SoC_hats[n] and SoC_init[x[n]] >= SoC_thrs[n] else 1e3
    pk = p_grid + p_grid * (1 - Lkavi_cur / Lk_max)
    ret = []
    for n in range(N):
        k, _ = split_kj(x[n])
        # tmp = alpha * enk[n] * pk[k] + beta * tao * ln[n, k] + punnishs[n]
        tmp = alpha * enk[n] * pk[k] + beta * tao * ln[n, k]
        ret.append(tmp)
    return np.array(ret) + punnishs


def find_pareto_frontier(fitnesses):
    """
    得到帕累托前沿，即不被支配的一组解

    :param fitnesses: 种群适应值
    :return: 帕累托前沿
    """
    pareto_frontier_idx = []
    for i, solution in enumerate(fitnesses):
        is_pareto = True
        for j, other_solution in enumerate(fitnesses):
            if i != j and is_dominate(other_solution, solution):
                is_pareto = False
                break
        if is_pareto:
            pareto_frontier_idx.append(i)
    return pareto_frontier_idx


def objective(x):
    return np.sum(f(x))


def random_search():
    best_x = np.zeros(N)
    with open("init_strategy.json", "r", encoding="utf-8") as file:
        content = json.load(file)
    for k, v in content.items():
        st = random.choice(v)
        best_x[int(k)] = cum_Jk[st[0]] + st[1]
    # best_fit = 1e10
    # for i in tqdm(range(1000000)):
    #     x = np.random.choice(np.arange(0, sum_Jk), N, replace=False, p=SoC_init / np.sum(SoC_init))
    #     fit = objective(x)
    #     if fit < best_fit:
    #         best_x = x.copy()
    #         best_fit = fit.copy()
    return f(np.array(best_x, dtype=np.int32))


def DE(pop_size=50, maxFEs=1e6, F=0.5, CR=0.5):
    """
    进化算法
    :param pop_size: 种群大小
    :param maxFEs: 最大评估数
    :param F: 因子F
    :param CR: 因子CR
    :return: 最优评估，最优解
    """
    population = np.array([np.random.choice(np.arange(0, sum_Jk), N, replace=False) for _ in range(pop_size)])
    fits = np.array([objective(x) for x in population])
    set_all = set(np.arange(np.sum(Jk)))
    FEs = pop_size
    ret = []
    while FEs < maxFEs:
        for i in range(N):
            # 交叉
            r1, r2, r3 = random.sample(list(range(i)) + list(range(i + 1, N)), 3)
            v = population[r1] + F * (population[r2] - population[r3])
            v = np.clip(v, 0, sum_Jk - 1)
            # 变异
            u = population[i].copy()
            jrand = random.randint(0, N - 1)
            for j in range(N):
                if random.random() <= CR:
                    u[j] = v[j]
            u[jrand] = v[jrand]
            # 无重复约束
            u = np.array(u, dtype=np.int32)
            set_new_x = set(u)
            replace = list(set_all - set_new_x)
            random.shuffle(replace)
            dic = defaultdict(list)
            for idx, bat in enumerate(u):
                dic[bat].append(idx)
            count = 0
            for k, v in dic.items():
                chs = random.sample(v, len(v) - 1)
                for c in chs:
                    u[c] = replace[count]
                    count += 1
            cur_fit = fits[i]
            new_fit = objective(u)
            if new_fit < cur_fit:
                population[i] = u.copy()
                fits[i] = new_fit.copy()
        FEs += pop_size
        print(f'[{FEs}/{maxFEs}],min(fits):{min(fits)}')
        idx = np.argmin(fits)
        Xs = f(population[idx])
        ret.append(Xs)
    return np.array(ret).T


def DE_MOP(pop_size=50, maxFEs=2e5, F=0.5, CR=0.5):
    """
    进化算法
    :param pop_size: 种群大小
    :param maxFEs: 最大评估数
    :param F: 因子F
    :param CR: 因子CR
    :return: 最优评估，最优解
    """
    population = np.array([np.random.choice(np.arange(0, sum_Jk), N, replace=False) for _ in range(pop_size)])
    fits = np.array([f(x) for x in population])
    set_all = set(np.arange(np.sum(Jk)))
    FEs = pop_size
    while FEs < maxFEs:
        for i in range(N):
            # 交叉
            r1, r2, r3 = random.sample(list(range(i)) + list(range(i + 1, N)), 3)
            v = population[r1] + F * (population[r2] - population[r3])
            v = np.clip(v, 0, sum_Jk - 1)
            # 变异
            u = population[i].copy()
            jrand = random.randint(0, N - 1)
            for j in range(N):
                if random.random() <= CR:
                    u[j] = v[j]
            u[jrand] = v[jrand]
            # 无重复约束
            u = np.array(u, dtype=np.int32)
            set_new_x = set(u)
            replace = list(set_all - set_new_x)
            random.shuffle(replace)
            dic = defaultdict(list)
            for idx, bat in enumerate(u):
                dic[bat].append(idx)
            count = 0
            for k, v in dic.items():
                chs = random.sample(v, len(v) - 1)
                for c in chs:
                    u[c] = replace[count]
                    count += 1
            cur_fit = fits[i]
            new_fit = f(u)
            if is_dominate(new_fit, cur_fit):
                population[i] = u.copy()
                fits[i] = new_fit.copy()
        FEs += pop_size
        print(f'[{FEs}/{maxFEs}],min(fits):{find_pareto_frontier(fits)}')
    ret = find_pareto_frontier(fits)
    # ret.append(Xs)
    return ret


if __name__ == "__main__":
    np.random.seed(None)
    costs = DE()
    for cost in costs:
        plt.plot(cost)
    plt.xlabel("iteration")
    plt.ylabel("cost")
    plt.show()
    # print(DE())

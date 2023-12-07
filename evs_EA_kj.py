import collections
import copy
import json
import random
from bisect import bisect_left
from collections import defaultdict

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from tqdm import tqdm

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
batterys = []
for i in range(len(Jk)):
    for j in range(Jk[i]):
        batterys.append((i, j))
cum_Jk = np.append(0, np.cumsum(Jk))
sa = 50
delta_yita_a = 7.344 * 1e-5 * sa ** 2 - 7.656 * 1e-3 * sa + 0.3536
SoC_init = [[0.92, 1, 1, 1, 0.8], [0.52, 0.8, 0.6, 0.92, 0.52, 0.92, 0.62, 1, 0.6, 0.8, 1, 0.87, 0.93],
            [1, 0.93, 1, 1, 0.73], [0.6, 0.73, 0.93, 1, 0.87, 0.67, 1, 0.8, 0.73, 0.87, 0.67, 0.6, 0.53],
            [0.73, 0.67, 1, 0.6, 0.53, 0.67, 0.87, 0.8, 0.73, 0.87, 0.6, 0.93, 0.53]]
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
Rk = 12.5 * Jk  # rho * delta_t=1.25*10
p_grid = 0.85
Lkavi_pre = np.array([np.sum(SoC_init[k] * En) for k in range(K)])
bts = []
for i in range(len(Jk)):
    for j in range(Jk[i]):
        bts.append((i, j))
set_all = set(bts)


def init_pop(pop_size=100):
    pop = []
    for g in range(10000):
        if len(pop) == pop_size:
            break
        bts_copy = copy.deepcopy(bts)
        random.shuffle(bts_copy)
        sq = []
        for ev in range(N):
            for bt in bts_copy:
                k, j = bt
                Soc_nk = SoC_zeros[ev] - ln[ev, k] * delta_yita_a / En
                if Soc_nk >= SoC_hats[ev] and SoC_init[k][j] >= SoC_thrs[ev]:
                    sq.append(bt)
                    bts.remove(bt)
                    break
            else:
                print(ev)
                break
        if len(sq) == N:
            pop.append(sq)
    return np.array(pop)


def f(x):  # 目标函数
    Lkavi_cur = Lkavi_pre + Rk
    enk = np.zeros(N)
    punnishs = np.zeros(N)
    for n in range(N):
        # 到车站约束 Soc_nk>=SoC_hats
        k, j = x[n]
        Soc_nk = SoC_zeros[n] - ln[n, k] * delta_yita_a / En
        # 到目的地约束 SoC_init[x[i]]>=SoC_thrs
        en = (SoC_init[k][j] - Soc_nk) * En
        Lkavi_cur[k] -= en
        enk[n] = en
        punnishs[n] = 0 if Soc_nk >= SoC_hats[n] and SoC_init[k][j] >= SoC_thrs[n] else 1e4
    pk = p_grid + p_grid * (1 - Lkavi_cur / Lk_max)
    ret = []
    for n in range(N):
        k, j = x[n]
        tmp = alpha * enk[n] * pk[k] + beta * tao * ln[n, k]
        ret.append(tmp)
    return np.array(ret), punnishs


def objective(x):
    return 1 / np.sum(f(x))


def rank(pop, fitness):
    pop_rank = np.argsort(-fitness)
    rank_pop = [pop[r] for r in pop_rank]
    rank_fit = fitness[pop_rank]
    return rank_pop, rank_fit


def select(pop, fitness, eliteSize, pop_size):
    fit_p = fitness / np.sum(fitness)
    select_pop = pop[:eliteSize] + random.choices(pop, weights=fit_p, k=pop_size - eliteSize)
    return select_pop


def cross(select_pop, eliteSize, pop_size):
    cross_pop = select_pop[:eliteSize]
    for i in range(pop_size - eliteSize):
        # 交叉
        r1, r2 = np.random.choice(np.arange(pop_size), 2, False)
        start, end = np.sort(np.random.choice(np.arange(N), 2, False))
        child = select_pop[r1][start:end]
        # for gene in np.append(select_pop[r2][end:], select_pop[r2][:end]):
        for gene in select_pop[r2]:
            if gene not in child:
                child.append(gene)
            if len(child) == N:
                break
        cross_pop.append(child)
    return cross_pop


def mutate(cross_pop, pop_size, mutationRate):
    for i in range(pop_size):
        if i == 0:
            continue
        set_new_x = set(cross_pop[i])
        replace = list(set_all - set_new_x)
        random.shuffle(replace)
        c = 0
        for j in range(N):
            if random.random() < mutationRate:
                cross_pop[i][j] = replace[c]
                c += 1
            if random.random() < mutationRate:
                a = random.randint(0, N - 1)
                cross_pop[i][a], cross_pop[i][j] = cross_pop[i][j], cross_pop[i][a]
    return cross_pop


def EA(pop_size=100, gmax=1000, eliteSize=10):
    pop = [random.sample(bts, N) for _ in range(pop_size)]
    # pop = init_pop()
    fitness = np.array([objective(x) for x in pop])
    mutationRate = 0.01
    ret = []
    for g in range(gmax):
        rank_pop, rank_fit = rank(pop, fitness)
        select_pop = select(rank_pop, rank_fit, eliteSize, pop_size)
        cross_pop = cross(select_pop, eliteSize, pop_size)
        pop = mutate(cross_pop, pop_size, mutationRate)
        fitness = np.array([objective(x) for x in pop])
        best_idx = np.argmax(fitness)
        print(f"g:{g},fit:{1 / fitness[best_idx]}, x:{pop[best_idx]}")
        # return fitness[best_idx], pop[best_idx]
        Xs = pop[best_idx]
        cost, _ = f(Xs)
        ret.append(cost)
    json.dump(Xs, open("bts.json", 'w'))
    return np.array(ret).T


if __name__ == "__main__":
    # np.random.seed(None)
    # pop = init_pop(pop_size=100)
    costs = EA()
    for cost in costs:
        plt.plot(cost)
    data = pd.DataFrame([[c[-1] for c in costs]])
    data.to_excel('EA.xlsx')
    plt.title(f'evs_GA-sum_cost={sum([c[-1] for c in costs])}')
    plt.xlabel("iteration")
    plt.ylabel("cost")
    plt.ylim([10,50])
    plt.show()
    # print(fit)
    # print(x)

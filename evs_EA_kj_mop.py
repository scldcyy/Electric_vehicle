import collections
import copy
import random
from bisect import bisect_left
from collections import defaultdict
from pymoo.indicators.hv import HV
import numpy as np
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
ref_point = [50 for _ in range(N)]
hv = HV(ref_point=ref_point)


def is_dominate(arr1, arr2):
    """
    判断arr1是否支配arr2。这里arr表示个体的多个目标函数适应值组成的向量

    :param arr1: 向量1
    :param arr2: 向量2
    :return: 是否支配
    """
    return np.all(arr1 <= arr2) and np.any(arr1 != arr2)


def Top_K_Arg(fitnesses, K):
    """
    根据非支配关系与拥挤距离排序策略对种群排序

    :param fitnesses: 种群适应值
    :param K: 选取数量
    :return: 前K个数量索引
    """
    ranks = sorting_fits(fitnesses)
    crowding_distances = crowding_distance(fitnesses)
    args = np.array([ranks, crowding_distances]).T
    top_K = sorted(range(len(fitnesses)), key=lambda k: (args[k][0], -args[k][1]))[:K]
    # top_K = np.argsort(crowding_distances)[:K]
    return top_K


def crowding_distance(fitnesses):
    """
    计算种群拥挤距离

    :param fitnesses: 种群适应值
    :return: 拥挤距离
    """
    num_individuals = len(fitnesses)
    num_objectives = len(fitnesses[0])
    crowding_distances = [0] * num_individuals
    for obj_index in range(num_objectives):
        sorted_indices = sorted(range(num_individuals), key=lambda i: fitnesses[i][obj_index])
        crowding_distances[sorted_indices[0]] = float('inf')
        crowding_distances[sorted_indices[-1]] = float('inf')
        obj_min = fitnesses[sorted_indices[0]][obj_index]
        obj_max = fitnesses[sorted_indices[-1]][obj_index]
        if obj_max - obj_min == 0:
            continue
        for i in range(1, num_individuals - 1):
            crowding_distances[sorted_indices[i]] += (fitnesses[sorted_indices[i + 1]][obj_index] -
                                                      fitnesses[sorted_indices[i - 1]][obj_index]) / (
                                                             obj_max - obj_min)
    return crowding_distances


def sorting_fits(fitnesses):
    """
    根据非支配关系对种群排序
    :param fitnesses: 种群适应值
    :return: 等级
    """
    ranks = [0] * len(fitnesses)
    # domination_mat = np.array(
    #     [[is_dominate(solutions[i], solutions[j]) for i in range(len(solutions))] for j in range(len(solutions))])
    domination_count = [0] * len(fitnesses)
    dominated_solutions = [[] for _ in range(len(fitnesses))]
    for i in range(len(fitnesses)):
        for j in range(i + 1, len(fitnesses)):
            if is_dominate(fitnesses[i], fitnesses[j]):
                domination_count[j] += 1
                dominated_solutions[i].append(j)
            elif is_dominate(fitnesses[j], fitnesses[i]):
                domination_count[i] += 1
                dominated_solutions[j].append(i)
    front = 0
    current_front = []
    for index, count in enumerate(domination_count):
        if count == 0:
            current_front.append(index)
    while current_front:
        next_front = []
        for index in current_front:
            for d in dominated_solutions[index]:
                domination_count[d] -= 1
                if domination_count[d] == 0:
                    ranks[d] = front + 1
                    next_front.append(d)
        front += 1
        current_front = next_front
    return ranks


def tournament_selection(fitnesses, tournament_size=10):
    """
    锦标赛策略，从种群中随机选择tournament_size个个体竞争，赢家胜出

    :param fitnesses: 种群适应值
    :param tournament_size: 一次锦标赛的个体数
    :return: 3次比赛的获胜者
    """
    selected = []
    under_selected = list(range(len(fitnesses)))
    for _ in range(3):
        tournament = random.sample(under_selected, tournament_size)
        winner = Top_K_Arg(fitnesses[tournament], 1)[0]
        selected.append(tournament[winner])
        under_selected.remove(tournament[winner])
    return selected


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
                    bts_copy.remove(bt)
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
    fn, punnishs = f(x)
    fit = np.sum(fn+punnishs)
    hv_ind = hv(fn)
    return fit, 200 / np.log(hv_ind)


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
    # pop = [random.sample(bts, N) for _ in range(pop_size)]
    pop = init_pop(pop_size)
    pop=[list(map(tuple,(x.tolist()))) for x in pop]
    fitness = np.array([objective(x) for x in pop])
    mutationRate = 0.01
    ret = []
    for g in range(gmax):
        rank_pop_idx = Top_K_Arg(fitness, pop_size)
        rank_pop = [pop[i] for i in rank_pop_idx]
        rank_sort = np.array(sorting_fits([fitness[i] for i in rank_pop_idx]))
        rank_sort = max(rank_sort) - rank_sort + 1
        select_pop = select(rank_pop, rank_sort, eliteSize, pop_size)
        cross_pop = cross(select_pop, eliteSize, pop_size)
        pop = mutate(cross_pop, pop_size, mutationRate)
        fitness = np.array([objective(x) for x in pop])
        best_idx = Top_K_Arg(fitness, 1)[0]
        print(f"g:{g},fit:{fitness[best_idx]}, x:{pop[best_idx]}")
        # return fitness[best_idx], pop[best_idx]
        Xs = pop[best_idx]
        cost, _ = f(Xs)
        ret.append(cost)
    return np.array(ret).T


if __name__ == "__main__":
    # np.random.seed(None)
    # pop = init_pop(pop_size=100)
    costs = EA()
    for cost in costs:
        plt.plot(cost)
    plt.xlabel("iteration")
    plt.ylabel("cost")
    plt.show()
    # print(fit)
    # print(x)

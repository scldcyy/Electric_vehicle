import copy
import json
import os
import random
import time

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

import CEC2009


def is_dominate(arr1, arr2):
    """
    判断arr1是否支配arr2。这里arr表示个体的多个目标函数适应值组成的向量

    :param arr1: 向量1
    :param arr2: 向量2
    :return: 是否支配
    """
    return np.all(arr1 <= arr2) and np.any(arr1 != arr2)


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


def calculate_distance(point_a, point_b):
    """
    计算两个点之间的欧氏距离

    :param point_a: 点a
    :param point_b: 点b
    :return: 欧氏距离
    """
    return np.linalg.norm(point_a - point_b)


def calculate_igd(approximate_front, true_front):
    """
    计算igd，用于衡量近似Pareto前沿与真实Pareto前沿之间的距离，评估算法性能。

    :param approximate_front: 算法得到的近似前沿
    :param true_front: 真实前沿
    :return: igd
    """
    total_distance = 0.0
    # 对于真实Pareto前沿中的每个解b
    for point_b in true_front:
        min_distance = float('inf')
        # 计算b与近似集合A中每个解的距离，并选择最小距离
        for point_a in approximate_front:
            distance = calculate_distance(point_a, point_b)
            min_distance = min(min_distance, distance)
        # 将所有选择的最小距离累加
        total_distance += min_distance
    # 计算平均距离，即IGD值
    igd = total_distance / len(true_front)
    return igd


def updateA(A, M, D, N, swarms, lb, ub, fitness_tuple, NA, FEs):
    S = []
    for i in range(M):
        for j in range(N):
            S.append(swarms[i][j])
    S += A
    # ELS
    A_new = copy.deepcopy(A)
    for i in range(len(A_new)):
        solution = A_new[i]
        d = random.randint(0, D - 1)
        tmp = solution.x[d]
        tmp += (lb[d] - ub[d]) * random.gauss(0, 1)
        # 防止越界
        tmp = min(tmp, ub[d])
        tmp = max(tmp, lb[d])
        solution.x[d] = tmp
        fit = fitness_tuple[solution.obj](solution.x)
        FEs += 1
        if fit < solution.fitBest:
            solution.fitBest = fit
            solution.pBest = solution.x
    S += A_new
    S_fit = np.array([[fitness_tuple[m](solution.x) for m in range(M)] for solution in S])
    FEs += M * len(S)
    R_idx = find_pareto_frontier(S_fit)
    R = [S[idx] for idx in R_idx]
    if len(R_idx) <= NA:
        A = R
    else:
        # 拥挤程度决定
        R_fit = S_fit[R_idx]
        dist = crowding_distance(R_fit)
        dist_idx = sorted(range(len(dist)), key=lambda i: -dist[i])[:NA]
        A = [S[idx] for idx in dist_idx]
    return A, FEs


class Particle:
    def __init__(self, x, v, obj, pBest=None, fitBest=None):
        self.x = x
        self.v = v
        self.obj = obj
        self.pBest = pBest
        self.fitBest = fitBest


def CMPSO(lb=np.array([0] + [-1] * 29), ub=np.array([1] * 30), NA=1000, maxFEs=3e6, N=200,
          fitness_tuple=(CEC2009.ZDT1_OBJ1, CEC2009.ZDT1_OBJ2), w=(0.4, 0.9), c1=4 / 3, c2=4 / 3, c3=4 / 3):
    """
    CMPSO流程

    :param lb:
    :param ub:
    :param NA:
    :param maxFEs:
    :param N:
    :param M:
    :param fitness_tuple:
    :param w:
    :param c1:
    :param c2:
    :param c3:
    :return:
    """
    D = len(lb)
    M = len(fitness_tuple)
    particle_min_v = -(ub - lb) * 0.2
    particle_max_v = (ub - lb) * 0.2
    A = []
    swarms = []
    FEs = 0
    gBest = [{"x": np.zeros(D), "fit": np.inf}] * M
    for i in range(M):
        particle_group = []
        for j in range(N):
            px = np.random.uniform(lb, ub)
            pv = np.random.uniform(particle_min_v, particle_max_v)
            particle = Particle(x=px, v=pv, obj=i, pBest=px, fitBest=fitness_tuple[i](px))
            particle_group.append(particle)
            if particle.fitBest < gBest[i]['fit']:
                gBest[i]['fit'] = particle.fitBest
                gBest[i]['x'] = particle.pBest
        FEs += N
        swarms.append(particle_group)
    # update A
    A, FEs = updateA(A, M, D, N, swarms, lb, ub, fitness_tuple, NA, FEs)
    while FEs <= maxFEs:
        w0 = w[1] - (FEs / maxFEs) * (w[1] - w[0])
        for i in range(M):
            for j in range(N):
                if len(A) == 0:
                    A_selected = gBest[random.choice(list(range(i)) + list(range(i + 1, M)))]['x'].copy()
                else:
                    A_selected = random.choice(A)
                particle = swarms[i][j]
                new_v = w0 * particle.v + c1 * np.random.random(size=D) * (
                        particle.pBest - particle.x) + c2 * np.random.random(size=D) * (
                                gBest[i]['x'] - particle.x) + c3 * np.random.random(size=D) * (
                                A_selected.x - particle.x)
                # 防止越界
                idx = np.where(new_v > ub)
                new_v[idx] = ub[idx] - 0.05 * np.random.uniform(0, 1, len(idx)) * (ub[idx] - lb[idx])
                idx = np.where(new_v < lb)
                new_v[idx] = lb[idx] + 0.05 * np.random.uniform(0, 1, len(idx)) * (ub[idx] - lb[idx])
                particle.v = new_v

                # update_x(i)
                new_x = particle.x + particle.v
                # 防止越界
                idx = np.where(new_x > ub)
                new_x[idx] = ub[idx] - 0.05 * np.random.uniform(0, 1, len(idx)) * (ub[idx] - lb[idx])
                idx = np.where(new_x < lb)
                new_x[idx] = lb[idx] + 0.05 * np.random.uniform(0, 1, len(idx)) * (ub[idx] - lb[idx])
                particle.x = new_x

                # update_fit(i)
                particle_fit = fitness_tuple[i](particle.x)
                if particle_fit < particle.fitBest:
                    particle.fitBest = particle_fit
                if particle_fit < gBest[i]['fit']:
                    gBest[i]['fit'] = particle.fitBest
                FEs += 1
        A, FEs = updateA(A, M, D, N, swarms, lb, ub, fitness_tuple, NA, FEs)
        print("FEs:", FEs)
    return A


def sq(x):
    return np.sum(x * x)


if __name__ == "__main__":
    # v_2
    fitness_tuple = (CEC2009.ZDT1_OBJ1, CEC2009.ZDT1_OBJ2)
    M = len(fitness_tuple)
    A = CMPSO()
    fitnesses = np.array([[fitness_tuple[m](solution.x) for m in range(2)] for solution in A])
    pareto_frontier_idx = find_pareto_frontier(fitnesses)
    pareto_frontier = np.array([fitnesses[idx] for idx in pareto_frontier_idx])
    plt.scatter(pareto_frontier[:, 0], pareto_frontier[:, 1])
    x = np.linspace(0, 1, 100)
    y = 1 - np.sqrt(x)
    IGD = calculate_igd(pareto_frontier, np.concatenate((x[:, None], y[:, None]), axis=1))
    print(f"IGD={IGD}")
    plt.plot(x, y)
    plt.savefig('pareto_frontier.png')
    plt.show()
    pass

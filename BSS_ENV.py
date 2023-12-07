import copy
import random
from bisect import bisect_left

import numpy as np

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
# Actions_Set = np.array([[-1, 0], [0, -1], [0, 0], [0, 1], [1, 0],[2,0],[-2,0],[0,2],[0,-2]])
Actions_Set = np.array([[-1, 0], [0, -1], [0, 0], [0, 1], [1, 0]])
bts = []
for i in range(len(JK)):
    for j in range(JK[i]):
        bts.append((i, j))


class BSS_ENV:
    def __init__(self):
        self.agents = [{} for _ in range(N)]
        self.action_space = [len(Actions_Set) for _ in range(N)]
        self.observation_space = [2 for _ in range(N)]

    def reset(self):
        bts_copy = copy.deepcopy(bts)
        random.shuffle(bts_copy)
        for ev in range(N):
            for bt in bts_copy:
                k, j = bt
                Soc_nk = EV_SoC[ev][0] - l_n_k[ev][k] * delta_yita_a / E_n
                if Soc_nk >= EV_SoC[ev][1] and Inital_battery_Soc[k][j] >= EV_SoC[ev][2]:
                    self.agents[ev]["state"] = [k, j]
                    self.agents[ev]["enk"] = (Inital_battery_Soc[k][j] - Soc_nk) * E_n
                    bts_copy.remove(bt)
                    break
        Lkavi_cur = Lkavi_pre + Rk
        for n in range(N):
            # 到车站约束 Soc_nk>=SoC_hats
            k, j = self.agents[n]["state"]
            en = self.agents[n]["enk"]
            Lkavi_cur[k] -= en
        pk = p_grid + p_grid * (1 - Lkavi_cur / Lk_max)
        for n in range(N):
            k, j = self.agents[n]["state"]
            self.agents[n]["f_n"] = alpha * self.agents[n]["enk"] * pk[k] + belta * tao * l_n_k[n][k]
        return [np.array(self.agents[n]["state"]) for n in range(N)]

    def step(self, actions):
        act_idx = np.argmax(actions, axis=-1)
        done = [False] * N
        if np.all(act_idx == 2):
            done = [True] * N
        act_vec = [Actions_Set[idx] for idx in act_idx]
        next_state = [self.agents[n]["state"] for n in range(N)]
        reward = [0 for _ in range(N)]

        for n in range(N):
            next_k, next_j = self.agents[n]["state"] + act_vec[n]
            next_k = np.clip(next_k, 0, K - 1)
            next_j = np.clip(next_j, 0, JK[next_k] - 1)
            next_state_n = [next_k, next_j]
            Soc_nk = EV_SoC[n][0] - l_n_k[n][next_k] * delta_yita_a / E_n
            if next_state_n == next_state[n]:  # 状态没变
                reward[n] = -10
            elif next_state_n not in next_state and Soc_nk >= EV_SoC[n][1] and Inital_battery_Soc[next_k][next_j] >= \
                    EV_SoC[n][2]:  # 未选择重复电站且满足约束
                self.agents[n]["state"] = next_state[n] = [next_k, next_j]
                self.agents[n]["enk"] = (Inital_battery_Soc[next_k][next_j] - Soc_nk) * E_n
            else:  # 选择重复电站或未满足约束
                reward[n] = -100

        Lkavi_cur = Lkavi_pre + Rk
        for n in range(N):
            k, j = self.agents[n]["state"]
            en = self.agents[n]["enk"]
            Lkavi_cur[k] -= en
        pk = p_grid + p_grid * (1 - Lkavi_cur / Lk_max)
        for n in range(N):
            k, j = self.agents[n]["state"]
            new_f_n = alpha * self.agents[n]["enk"] * pk[k] + belta * tao * l_n_k[n][k]
            if reward[n] == 0:
                # reward[n] = self.agents[n]["f_n"] - new_f_n
                reward[n] = self.agents[n]["f_n"] - new_f_n
                if reward[n] > 0:
                    reward[n] *= 50
            self.agents[n]["f_n"] = new_f_n
        return np.array(next_state), np.array(reward), done


class BSS_ENV2:
    def __init__(self):
        self.agents = [{} for _ in range(N)]
        self.action_space = [len(bts) for _ in range(N)]
        self.observation_space = [2 for _ in range(N)]

    def reset(self, Q=None):
        bts_copy = copy.deepcopy(bts)
        random.shuffle(bts_copy)
        if Q:
            bts_copy = [[q[1], q[2]] for q in Q]
        for ev in range(N):
            for bt in bts_copy:
                k, j = bt
                Soc_nk = EV_SoC[ev][0] - l_n_k[ev][k] * delta_yita_a / E_n
                if Soc_nk >= EV_SoC[ev][1] and Inital_battery_Soc[k][j] >= EV_SoC[ev][2]:
                    self.agents[ev]["state"] = [k, j]
                    self.agents[ev]["enk"] = (Inital_battery_Soc[k][j] - Soc_nk) * E_n
                    bts_copy.remove(bt)
                    break
        Lkavi_cur = Lkavi_pre + Rk
        for n in range(N):
            # 到车站约束 Soc_nk>=SoC_hats
            k, j = self.agents[n]["state"]
            en = self.agents[n]["enk"]
            Lkavi_cur[k] -= en
        pk = p_grid + p_grid * (1 - Lkavi_cur / Lk_max)
        for n in range(N):
            k, j = self.agents[n]["state"]
            self.agents[n]["f_n"] = alpha * self.agents[n]["enk"] * pk[k] + belta * tao * l_n_k[n][k]
        return [np.array(self.agents[n]["state"]) for n in range(N)]

    def step(self, actions):
        act_idx = np.argmax(actions, axis=-1)
        done = [False] * N
        act_vec = [bts[idx] for idx in act_idx]
        next_state = [self.agents[n]["state"] for n in range(N)]
        reward = [0 for _ in range(N)]

        for n in range(N):
            next_k, next_j = act_vec[n]
            next_state_n = [next_k, next_j]
            Soc_nk = EV_SoC[n][0] - l_n_k[n][next_k] * delta_yita_a / E_n
            if next_state_n == next_state[n]:  # 状态没变
                reward[n] = -10
            elif next_state_n not in next_state and Soc_nk >= EV_SoC[n][1] and Inital_battery_Soc[next_k][next_j] >= \
                    EV_SoC[n][2]:  # 未选择重复电站且满足约束
                self.agents[n]["state"] = next_state[n] = [next_k, next_j]
                self.agents[n]["enk"] = (Inital_battery_Soc[next_k][next_j] - Soc_nk) * E_n
            else:  # 选择重复电站或未满足约束
                reward[n] = -100

        Lkavi_cur = Lkavi_pre + Rk
        for n in range(N):
            k, j = self.agents[n]["state"]
            en = self.agents[n]["enk"]
            Lkavi_cur[k] -= en
        pk = p_grid + p_grid * (1 - Lkavi_cur / Lk_max)
        for n in range(N):
            k, j = self.agents[n]["state"]
            new_f_n = alpha * self.agents[n]["enk"] * pk[k] + belta * tao * l_n_k[n][k]
            if reward[n] == 0:
                # reward[n] = self.agents[n]["f_n"] - new_f_n
                reward[n] = self.agents[n]["f_n"] - new_f_n
                if reward[n] > 0:
                    reward[n] *= 50
            self.agents[n]["f_n"] = new_f_n
        return np.array(next_state), np.array(reward), done


def split_kj(kj):
    kj += 1
    k = bisect_left(cum_Jk, kj) - 1
    j = kj - cum_Jk[k] - 1
    return k, j


Actions_Set3 = list(range(-15, 15))


class BSS_ENV3:
    def __init__(self):
        self.agents = [{} for _ in range(N)]
        self.action_space = [len(Actions_Set3) for _ in range(N)]
        self.observation_space = [1 for _ in range(N)]

    def reset(self,Q=None):
        bts_copy = list(range(sum(JK)))
        random.shuffle(bts_copy)
        if Q:
            bts_copy = [cum_Jk[q[1]] + q[2] for q in Q]
        for ev in range(N):
            for bt in bts_copy:
                k, j = split_kj(bt)
                Soc_nk = EV_SoC[ev][0] - l_n_k[ev][k] * delta_yita_a / E_n
                if Soc_nk >= EV_SoC[ev][1] and Inital_battery_Soc[k][j] >= EV_SoC[ev][2]:
                    self.agents[ev]["state"] = bt
                    self.agents[ev]["enk"] = (Inital_battery_Soc[k][j] - Soc_nk) * E_n
                    bts_copy.remove(bt)
                    break
        Lkavi_cur = Lkavi_pre + Rk
        for n in range(N):
            # 到车站约束 Soc_nk>=SoC_hats
            k, j = split_kj(self.agents[n]["state"])
            en = self.agents[n]["enk"]
            Lkavi_cur[k] -= en
        pk = p_grid + p_grid * (1 - Lkavi_cur / Lk_max)
        for n in range(N):
            k, j = split_kj(self.agents[n]["state"])
            self.agents[n]["f_n"] = alpha * self.agents[n]["enk"] * pk[k] + belta * tao * l_n_k[n][k]
        return [np.array([self.agents[n]["state"]]) for n in range(N)]

    def step(self, actions):
        act_idx = np.argmax(actions, axis=-1)
        done = [False] * N
        if np.all(act_idx == 15):
            done = [True] * N
        act_vec = [Actions_Set3[idx] for idx in act_idx]
        next_state = [self.agents[n]["state"] for n in range(N)]
        reward = [0 for _ in range(N)]

        for n in range(N):
            next_kj = (self.agents[n]["state"] + act_vec[n]) % np.sum(JK)
            next_k, next_j = split_kj(next_kj)
            next_state_n = next_kj
            Soc_nk = EV_SoC[n][0] - l_n_k[n][next_k] * delta_yita_a / E_n
            if next_state_n == next_state[n]:  # 状态没变
                reward[n] =0
            elif next_state_n not in next_state and Soc_nk >= EV_SoC[n][1] and Inital_battery_Soc[next_k][next_j] >= \
                    EV_SoC[n][2]:  # 未选择重复电站且满足约束
                self.agents[n]["state"] = next_state[n] = next_kj
                self.agents[n]["enk"] = (Inital_battery_Soc[next_k][next_j] - Soc_nk) * E_n
            else:  # 选择重复电站或未满足约束
                reward[n] = -100

        Lkavi_cur = Lkavi_pre + Rk
        for n in range(N):
            k, j = split_kj(self.agents[n]["state"])
            en = self.agents[n]["enk"]
            Lkavi_cur[k] -= en
        pk = p_grid + p_grid * (1 - Lkavi_cur / Lk_max)
        for n in range(N):
            k, j = split_kj(self.agents[n]["state"])
            new_f_n = alpha * self.agents[n]["enk"] * pk[k] + belta * tao * l_n_k[n][k]
            if reward[n] == 0:
                # reward[n] = self.agents[n]["f_n"] - new_f_n
                reward[n] = 1000 / new_f_n
            self.agents[n]["f_n"] = new_f_n
        # return np.array(next_state), np.array(reward), done
        return [np.array([n]) for n in next_state], np.array(reward), done


if __name__ == "__main__":
    env = BSS_ENV()
    states = env.reset()

    for i in range(10):
        actions = np.zeros((N, len(Actions_Set)))
        for j in range(N):
            actions[j][random.randint(0, len(Actions_Set) - 1)] = 1
        next_state, reward, done = env.step(actions)
        print(next_state, reward, done)

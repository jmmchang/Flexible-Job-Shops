import random
import numpy as np
import copy
from functions import decode_schedule
from collections import defaultdict

class SimulatedAnnealing:
    def __init__(self, problem, initial_temperature = 1000, min_temperature = 0.01,
                 max_iteration = 100, steps = 50, cooling_parameter = 0.98):

        self.init_temp = initial_temperature
        self.problem = problem
        self.min_temp = min_temperature
        self.max_iter = max_iteration
        self.steps = steps
        self.cool = cooling_parameter

    def evaluate(self, target):
        return self.problem.decode(target[0], target[1])

    def random_swap(self, target):
        new_target = copy.deepcopy(target)
        new_assign, new_prio = new_target

        schedule = decode_schedule(
            self.problem.jobs_data,
            self.problem.release,
            self.problem.setup_times,
            new_assign,
            new_prio)

        by_machine = defaultdict(list)
        for j, ops in schedule.items():
            for o, m, st, _ in ops:
                by_machine[m].append((st, j, o))

        # 2. 隨機挑一台有 >=2 道工序的機台
        machines = [m for m, lst in by_machine.items() if len(lst) > 1]
        if not machines:
            return new_assign, new_prio
        m = random.choice(machines)

        # 3. 排序後隨機選一個「當前」索引 idx，與 idx-1 的工序做交換
        ops = sorted(by_machine[m], key = lambda x: x[0])
        idx = random.randint(1, len(ops) - 1)
        _, j_cur, o_cur = ops[idx]
        _, j_pre, o_pre = ops[idx - 1]
        if j_cur != j_pre:
            new_prio[(j_cur, o_cur)], new_prio[(j_pre, o_pre)] = new_prio[(j_pre, o_pre)], new_prio[(j_cur, o_cur)]

        for j in (j_pre, j_cur):
            ops_count = len(self.problem.jobs_data[j])
            # 收集所有 op 的 priority，排序
            sorted_vals = sorted(new_prio[(j, o)] for o in range(ops_count))
            # 依序寫回，保證 op0 < op1 < ...
            for o, val in enumerate(sorted_vals):
                new_prio[(j, o)] = val

        return new_target

    def run(self, seed_solution):
        t = self.init_temp
        k = 0
        curr = seed_solution
        curr_obj = self.evaluate(curr)
        best = curr
        best_obj = curr_obj

        while t > self.min_temp and k < self.max_iter:
            for _ in range(self.steps):
                new = self.random_swap(curr)
                new_obj = self.evaluate(new)
                diff = new_obj - curr_obj
                prob = min(1, np.exp(-diff / t))

                if prob >= random.uniform(0, 1):
                    curr = new
                    curr_obj = new_obj

                if new_obj < best_obj:
                    best, best_obj = new, new_obj

            t = t * self.cool
            k += 1

        return best_obj, best
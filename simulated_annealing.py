import random
import numpy as np
import copy
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
        return self.problem.decode(target[1])

    def random_swap(self, target):
        new_target = copy.deepcopy(target)
        new_assign, new_times = new_target
        schedule = self.problem.generate_schedule(new_assign, new_times)

        by_machine = defaultdict(list)
        for j, ops in schedule.items():
            for o, m, st, end in ops:
                by_machine[m].append((st, j, o, end))

        machines = [m for m, lst in by_machine.items() if len(lst) > 1]
        if not machines:
            return new_assign, new_times

        m = random.choice(machines)
        ops = sorted(by_machine[m], key = lambda x: x[0])
        idx = random.randint(1, len(ops) - 1)
        st_cur, j_cur, o_cur, end_cur = ops[idx]
        st_pre, j_pre, o_pre, end_pre = ops[idx - 1]
        j_temp, o_temp = None, None
        if idx >= 2:
            _, j_temp, o_temp, _ = ops[idx - 2]

        if j_cur != j_pre:
            if j_temp is not None:
                temp = st_pre + self.problem.setup_times[m[:2]][((j_temp, o_temp), (j_cur, o_cur))]
                new_times[(j_cur, o_cur)] = (temp, temp + end_cur - st_cur)
            else:
                new_times[(j_cur, o_cur)] = (st_pre, st_pre + end_cur - st_cur)

            s = new_times[(j_cur, o_cur)][1] + self.problem.setup_times[m[:2]][((j_cur, o_cur), (j_pre, o_pre))]
            new_times[(j_pre, o_pre)] = (s, s + end_pre - st_pre)

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
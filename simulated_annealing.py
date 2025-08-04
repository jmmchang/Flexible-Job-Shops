import random
import numpy as np
import copy
from collections import defaultdict, deque

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
        new_assign, new_times = target
        priority = []
        schedule = self.problem.generate_schedule(new_assign, new_times)
        by_machine = defaultdict(list)

        for j, ops in schedule.items():
            for o, m, st, end in ops:
                priority.append((j, o, st))
                by_machine[m].append((j, o, st))

        machines = [m for m, lst in by_machine.items() if len(lst) > 1]
        if not machines:
            return new_assign, new_times

        m = random.choice(machines)
        priority.sort(key = lambda x: x[2])
        ops_m = sorted(by_machine[m], key = lambda x: x[2])
        idx = random.randint(1, len(ops_m) - 1)
        j_cur, o_cur, st_cur = ops_m[idx]
        j_pre, o_pre, st_pre = ops_m[idx - 1]
        priority[priority.index((j_cur,o_cur,st_cur))], priority[priority.index((j_pre,o_pre,st_pre))] = (j_pre, o_pre, st_pre), (j_cur, o_cur, st_cur)

        groups = defaultdict(list)
        for (x, y, _) in priority:
            groups[x].append((x, y))

        for x in groups:
            groups[x].sort(key = lambda t: t[1])
            groups[x] = deque(groups[x])

        grouped_priority = []
        for x, _, _ in priority:
            grouped_priority.append(groups[x].popleft())

        new_machine, new_times = self.problem.encode(priority = grouped_priority)
        new_target = [new_machine, new_times]

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
import random
import copy
from collections import defaultdict, deque

class TabuSearch:
    def __init__(self, problem, steps = 500, tabu_size = 20):
        self.problem = problem
        self.size = tabu_size
        self.steps = steps
        self.tabu_list = deque()

    def evaluate(self, target):
        return self.problem.decode(target[1])

    def tabu_swap(self, target):
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
        best, best_obj = None, float('inf')
        j_cur_best, o_cur_best = None, None
        j_pre_best, o_pre_best = None, None

        for idx in range(1, len(ops)):
            curr_assign, curr_times = copy.deepcopy(new_assign), copy.deepcopy(new_times)
            st_cur, j_cur, o_cur, end_cur = ops[idx]
            st_pre, j_pre, o_pre, end_pre = ops[idx - 1]

            if sorted([(j_cur, o_cur), (j_pre, o_pre)]) not in self.tabu_list:
                j_temp, o_temp = None, None
                if idx >= 2:
                    _, j_temp, o_temp, _ = ops[idx - 2]

                if j_cur != j_pre:
                    if j_temp is not None:
                        temp = st_pre + self.problem.setup_times[m[:2]][((j_temp, o_temp), (j_cur, o_cur))]
                        curr_times[(j_cur, o_cur)] = (temp, temp + end_cur - st_cur)
                    else:
                        curr_times[(j_cur, o_cur)] = (st_pre, st_pre + end_cur - st_cur)

                    s = curr_times[(j_cur, o_cur)][1] + self.problem.setup_times[m[:2]][((j_cur, o_cur), (j_pre, o_pre))]
                    curr_times[(j_pre, o_pre)] = (s, s + end_pre - st_pre)

                curr = [curr_assign, curr_times]
                curr_obj = self.evaluate(curr)
                if curr_obj < best_obj:
                    best, best_obj = curr, curr_obj
                    j_cur_best, o_cur_best = j_cur, o_cur
                    j_pre_best, o_pre_best = j_pre, o_pre

        if len(self.tabu_list) > self.size:
            self.tabu_list.popleft()
        if best is not None:
            self.tabu_list.append(sorted([(j_cur_best, o_cur_best), (j_pre_best, o_pre_best)], key = lambda x: x[0]))
            new_target = best

        return new_target

    def run(self, seed_solution):
        curr = seed_solution
        curr_obj = self.evaluate(curr)
        best = curr
        best_obj = curr_obj

        for _ in range(self.steps):
            curr = self.tabu_swap(curr)
            curr_obj = self.evaluate(curr)
            if curr_obj < best_obj:
                best, best_obj = curr, curr_obj

        return best_obj, best
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
        best = None
        best_obj = float('inf')
        new_target = copy.deepcopy(target)
        new_assign, new_times = new_target
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
        j_cur_best, o_cur_best, j_pre_best, o_pre_best = None, None, None, None

        for idx in range(1, len(ops_m)):
            curr_assign, curr_times, curr_priority = copy.deepcopy(new_assign), copy.deepcopy(new_times), copy.deepcopy(priority)
            j_cur, o_cur, st_cur = ops_m[idx]
            j_pre, o_pre, st_pre = ops_m[idx - 1]

            if sorted([(j_cur, o_cur), (j_pre, o_pre)]) not in self.tabu_list:
                curr_priority[priority.index((j_cur, o_cur, st_cur))], curr_priority[priority.index((j_pre, o_pre, st_pre))] = (
                    j_pre, o_pre, st_pre), (j_cur, o_cur, st_cur)

                groups = defaultdict(list)
                for (x, y, _) in curr_priority:
                    groups[x].append((x, y))

                for x in groups:
                    groups[x].sort(key = lambda t: t[1])
                    groups[x] = deque(groups[x])

                grouped_priority = []
                for x, _, _ in curr_priority:
                    grouped_priority.append(groups[x].popleft())

                curr_assign, curr_times = self.problem.encode(priority = grouped_priority)
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
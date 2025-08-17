import random
import copy
from collections import defaultdict, deque

class TabuSearch:
    """
    A Tabu Search template for scheduling problems.
    The Tabu Search explores the neighborhood of the current solution by
    swapping adjacent operations on a single machine. It uses a tabu list
    to forbid reversing recent moves, guiding the search away from cycles.

    Attributes:
        problem:   Problem interface providing
                       - encode(assign=None, priority=None) -> (machine_assign, time_dict)
                       - decode(time_dict) -> objective value (to minimize)
                       - generate_schedule(assign, time_dict) -> detailed schedule
        steps:     Maximum number of iterations
        tabu_size: Maximum length of the tabu list
        tabu_list: deque storing recent moves as sorted lists of op-tuples (FIFO)
    """

    def __init__(self, problem, steps = 500, tabu_size = 20):
        self.problem = problem
        self.size = tabu_size
        self.steps = steps
        self.tabu_list = deque()

    def evaluate(self, target):
        """
        Compute the objective value of a solution.
        """

        return self.problem.decode(target[1])

    def tabu_swap(self, target, prob = 0.2):
        """
        Explore neighborhood by swapping adjacent ops on one random machine,
        respecting the tabu list, and return the best admissible neighbor.
        Steps:
        1. Decode current schedule to get start times per operation.
        2. Reassign machines with small probability.
        3. Pick a random machine with at least 2 ops.
        4. For each adjacent pair on that machine:
           a. If the swap move is not in the tabu list, apply the swap.
           b. Re-encode and evaluate.
           c. Track the best neighbor and its defining move.
        5. If the best non-tabu neighbor is found, push it into the tabu_list.
        """

        best = None
        best_obj = float('inf')
        new_target = copy.deepcopy(target)
        new_assign, new_times = new_target

        if random.random() < prob:
            new_assign, new_times = self.problem.encode()

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
        """
        Execute the Tabu Search loop.
        """

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
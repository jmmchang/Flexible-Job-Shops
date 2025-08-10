from itertools import combinations, permutations
from collections import namedtuple
from ortools.sat.python import cp_model
from numpy import floor
import random
import matplotlib.pyplot as plt

def solve_fjs_with_parallel_machines(jobs_data, release_dates, due_dates,
                                     weights, setup_times, center_caps, alpha = 0.5):
    """
    Solves a Flexible Job Shop Scheduling (FJS) problem with parallel machines and sequence-dependent setup times
    using Google OR-Tools CP-SAT solver.

    Args:
        jobs_data: Dictionary mapping job ID to a list of operations, each as (duration, center).
        release_dates: Dictionary mapping job ID to its release time.
        due_dates: Dictionary mapping job ID to its due date.
        weights: Dictionary mapping job ID to its tardiness weight.
        setup_times: Dictionary mapping machine name to a dictionary of setup times between operation pairs.
        center_caps: Dictionary mapping center name to its number of machines.
        alpha: Weighting factor between makespan and total weighted tardiness (0 < alpha < 1).

    Returns:
        schedule: Dictionary mapping job ID to a list of scheduled operations as (op_index, machine, start, end).
        objective_value: The value of the objective function.
        If no feasible solution is found, returns (None, None).
    """

    model = cp_model.CpModel()
    # Create machines based on center capacities
    machines, center_of = [], {}
    for p, cap in center_caps.items():
        for k in range(cap):
            m = f"{p}_{k}"
            machines.append(m)
            center_of[m] = p

    # Decision variables
    horizon = sum(d for ops in jobs_data.values() for d, _ in ops) + max(release_dates.values())
    assign, tardiness = {}, {}

    # Named tuple to store information about created variables.
    task_type = namedtuple("task_type", "start end interval")
    # Creates job intervals and add to the corresponding machine lists.
    all_tasks = {}

    # Create interval variables for each operation and machine assignment booleans
    for j, ops in jobs_data.items():
        for o, (duration, center) in enumerate(ops):
            suffix = f"s_j{j}o{o}"
            start_var = model.new_int_var(release_dates[j], horizon, "start" + suffix)
            end_var = model.new_int_var(release_dates[j], horizon, "end" + suffix)
            valid = [m for m in machines if center_of[m] in center]
            bools = []
            for m in valid:
                b = model.new_bool_var(f"a_j{j}o{o}_m{m}")
                assign[j, o, m] = b
                bools.append(b)

            model.add_exactly_one(bools)
            interval_var = model.new_interval_var(start_var, duration, end_var, "interval" + suffix)
            all_tasks[j, o] = task_type(start = start_var, end = end_var, interval = interval_var)

    # Enforce operations order
    for j, ops in jobs_data.items():
        for o in range(len(ops) - 1):
            model.add(all_tasks[j, o + 1].start >= all_tasks[j, o].end)

    # Machine mutual exclusion and sequence-dependent setup constraints
    for m in machines:
        tasks = [(j, o) for j, ops in jobs_data.items()
                        for o, (_, centers) in enumerate(ops)
                        if center_of[m] in centers]
        for (j1, o1), (j2, o2) in combinations(tasks, 2):
            b = model.new_bool_var(f"ord_{j1}o{o1}_{j2}o{o2}_on_{m}")
            s12 = setup_times[m][((j1, o1), (j2, o2))]
            s21 = setup_times[m][((j2, o2), (j1, o1))]
            model.add(all_tasks[j2, o2].start >= all_tasks[j1, o1].end + s12).only_enforce_if([assign[j1, o1, m], assign[j2, o2, m], b])
            model.add(all_tasks[j1, o1].start >= all_tasks[j2, o2].end + s21).only_enforce_if([assign[j1, o1, m], assign[j2, o2, m], b.Not()])

    # Tardiness
    for j, ops in jobs_data.items():
        last = len(ops) - 1
        tardiness[j] = model.new_int_var(0, horizon, f"T_j{j}")
        model.add_max_equality(tardiness[j],[0, all_tasks[j, last].end - due_dates[j]])

    # Makespan
    makespan = model.new_int_var(0, horizon, "makespan")
    model.add_max_equality(
        makespan,
        [all_tasks[j, len(jobs_data[j]) - 1].end for j in range(len(jobs_data))],
    )
    model.minimize(alpha * makespan + (1 - alpha) * sum(weights[j] * tardiness[j] for j in jobs_data))

    # Solve the model
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 60
    status = solver.solve(model)

    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        schedule = {}
        for j, ops in jobs_data.items():
            schedule[j] = []
            for o in range(len(ops)):
                m_assigned = next(
                    m for m in machines
                    if (j, o, m) in assign and solver.value(assign[j, o, m]) == 1
                )
                schedule[j].append((o, m_assigned, solver.value(all_tasks[j, o].start), solver.value(all_tasks[j, o].end)))

        return schedule, solver.objective_value

    return None, None

def generate_random_instance(num_jobs = 30, centers = ('C1','C2','C3',"C4","C5","C6"),
                             center_caps = {'C1':3,'C2':2,'C3':3,"C4":2,"C5":3,"C6":2}):
    """
    Generates a random FJSP instance with multiple jobs, centers, and sequence-dependent setup times.

    Args:
        num_jobs: Number of jobs to generate.
        centers: Tuple of center names.
        center_caps: Dictionary mapping each center to its machine capacity.

    Returns:
        jobs_data: Dictionary mapping job ID to a list of operations (duration, center).
        release_dates: Dictionary mapping job ID to its release time.
        due_dates: Dictionary mapping job ID to its due date.
        weights: Dictionary mapping job ID to its tardiness weight.
        setup_times: Dictionary mapping machine name to setup times between operation pairs.
        center_caps: The same center capacity dictionary passed in.
    """

    jobs_data, release_dates, due_dates, weights = {}, {}, {}, {}
    machines = []

    for p, cap in center_caps.items():
        for k in range(cap):
            m = f"{p}_{k}"
            machines.append(m)

    setup_times = {m:{} for m in machines}

    for j in range(num_jobs):
        ops = []
        total_dur = 0
        sample = sorted(random.sample(centers, 4), key = lambda s: int(s[1]))
        for c in sample:
            dur = random.randint(20, 50)
            total_dur += dur
            ops.append((dur, c))

        jobs_data[j] = ops
        due_dates[j]     = random.randint(floor(total_dur * 2), floor(total_dur * 5))
        release_dates[j] = random.randint(0, (due_dates[j] - total_dur) // 2)
        weights[j]       = random.randint(1, 5)

    for m in machines:
        all_ops = [(j, o) for j, ops in jobs_data.items()
                          for o in range(len(ops))
                          if m.split('_')[0] in ops[o][1]]

        for (j1,o1), (j2,o2) in permutations(all_ops, 2):
            setup_times[m][((j1,o1),(j2,o2))] = random.randint(5, 10)

    return jobs_data, release_dates, due_dates, weights, setup_times, center_caps

def plot_gantt(schedule, title="Gantt Chart"):
    """
    Plots a Gantt chart for a given job schedule using matplotlib.

    Args:
        schedule: Dictionary mapping job ID to a list of operations.
                  Each operation is a tuple: (operation index, machine name, start time, end time).
        title: Title of the chart.

    Returns:
        None. Displays the Gantt chart.
    """

    # Prepare y-axis: list of machines and their vertical positions
    machines = sorted({ m for ops in schedule.values() for _, m, _, _ in ops })
    y_pos = { m: i for i, m in enumerate(machines) }

    # Color mapping: assign a unique color to each job ID
    cmap = plt.get_cmap("tab20")
    job_ids = sorted(schedule)
    color_map = { j: cmap(i % 20) for i, j in enumerate(job_ids) }

    # Plot each operation as a horizontal bar
    fig, ax = plt.subplots(figsize=(12, len(machines)*0.6+1))
    for j, ops in schedule.items():
        for op_idx, m, st, en in ops:
            ax.barh(y_pos[m],
                    en - st,
                    left=st,
                    height=0.4,
                    color=color_map[j],
                    edgecolor="black")
            ax.text(st + (en-st)/2,
                    y_pos[m],
                    f"J{j}O{op_idx}",
                    ha="center",
                    va="center",
                    color="white",
                    fontsize=8)

    ax.set_yticks(list(y_pos.values()))
    ax.set_yticklabels(machines)
    ax.set_xlabel("Time")
    ax.set_title(title)
    ax.grid(axis="x", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()

class SchedulingProblem:
    def __init__(self, jobs_data, release_dates, due_dates,
                 weights, setup_times, center_caps, alpha = 0.5):
        self.jobs_data   = jobs_data
        self.release     = release_dates
        self.due_dates   = due_dates
        self.weights     = weights
        self.setup_times = setup_times
        self.center_caps = center_caps
        self.alpha       = alpha

    def generate_times(self, machine_assign, priority = None):
        """
        Simulates scheduling based on machine assignments and optional operation priority.

        Args:
            machine_assign: Mapping from (job, operation) to assigned machine.
            priority: Optional list of operations to schedule in order.

        Returns:
            doc_times: Mapping from (job, operation) to (start, end) times.
        """

        machines = [f"{c}_{k}"
                    for c, cap in self.center_caps.items()
                    for k in range(cap)]
        mc_timeline = {m: [] for m in machines}
        doc_times = {}
        unscheduled = set(machine_assign.keys())

        if not priority:
            while unscheduled:
                ready = [(j, o) for (j, o) in unscheduled if o == 0 or (j, o-1) in doc_times]
                # Greedy scheduling based on readiness
                earliest = {}
                for (j, o) in ready:
                    m = machine_assign[(j, o)]
                    p = m.split('_')[0]
                    dur, _ = self.jobs_data[j][o]
                    t0 = self.release[j]
                    if o > 0:
                        t0 = max(t0, doc_times[(j, o - 1)][1])

                    t = t0
                    for prev_end, prev_op in sorted(mc_timeline[m], key=lambda x: x[0]):
                        s = self.setup_times.get(p, {}).get((prev_op, (j, o)), 0)
                        t = max(t, prev_end + s)

                    earliest[(j, o)] = t

                sel = min(earliest, key=earliest.get)
                st = earliest[sel]
                dur, _ = self.jobs_data[sel[0]][sel[1]]
                end = st + dur
                mc_timeline[machine_assign[sel]].append((end, sel))
                doc_times[sel] = (st, end)
                unscheduled.remove(sel)
        else:
            # Schedule based on provided priority list
            ready = priority
            for (j, o) in ready:
                m = machine_assign[(j, o)]
                p = m.split('_')[0]
                dur, _ = self.jobs_data[j][o]
                t0 = self.release[j]
                if o > 0:
                    t0 = max(t0, doc_times[(j, o-1)][1])

                st = t0
                for prev_end, prev_op in sorted(mc_timeline[m], key = lambda x: x[0]):
                    s = self.setup_times.get(p, {}).get((prev_op, (j, o)), 0)
                    st = max(st, prev_end + s)

                dur, _ = self.jobs_data[j][o]
                end = st + dur

                mc_timeline[machine_assign[(j,o)]].append((end, (j,o)))
                doc_times[(j,o)] = (st, end)

        return doc_times

    def encode(self, seed_solution = None, priority = None):
        """
        Generates a machine assignment and corresponding operation times.

        Args:
            seed_solution: Optional seed machine assignment and times.
            priority: Optional scheduling priority list.

        Returns:
            machine_assign: Mapping from (job, operation) to machine.
            times: Mapping from (job, operation) to (start, end) times.
        """

        if seed_solution:
            machine_assign = seed_solution[0]
        else:
            machine_assign = {}
            for j, ops in self.jobs_data.items():
                for o in range(len(ops)):
                    centers = ops[o][1]
                    p = random.choice([centers])
                    k = random.randrange(self.center_caps[p])
                    machine_assign[(j, o)] = f"{p}_{k}"

        times = self.generate_times(machine_assign, priority)

        return machine_assign, times

    def decode(self, times):
        """
        Computes the objective value based on operation times.

        Args:
            times: Mapping from (job, operation) to (start, end) times.

        Returns:
            Objective value: alpha * makespan + (1 - alpha) * total weighted tardiness.
        """

        makespan = max(end for _, end in times.values())
        tardiness = [0] * len(self.jobs_data)
        for j in range(len(self.jobs_data)):
            tardiness[j] = max(0, times[(j, len(self.jobs_data[j]) - 1)][1] - self.due_dates[j])

        obj = 0
        for j in range(len(self.jobs_data)):
            obj += self.weights[j] * tardiness[j]

        obj *= 1 - self.alpha
        obj += self.alpha * makespan

        return obj

    @staticmethod
    def generate_schedule(machine_assign, times):
        """
        Converts raw machine assignments and operation times into a structured schedule.

        Args:
            machine_assign: Mapping from (job, operation) to assigned machine.
            times: Mapping from (job, operation) to (start, end) times.

        Returns:
            schedule: Dictionary mapping job ID to a list of scheduled operations,
                      each as (operation index, machine, start time, end time).
        """

        schedule = {}
        for (j,o), (st,end) in times.items():
            m = machine_assign[(j,o)]
            schedule.setdefault(j, []).append((o, m, st, end))

        return schedule
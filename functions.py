import itertools
import collections
from ortools.sat.python import cp_model
import random
import numpy as np
import matplotlib.pyplot as plt

def solve_fjs_with_parallel_machines(jobs_data, release_dates, due_dates,
                                     weights, setup_times, center_caps, alpha = 0.5):
    model = cp_model.CpModel()
    # 1. 生成實體機台
    machines, center_of = [], {}
    for p, cap in center_caps.items():
        for k in range(cap):
            m = f"{p}_{k}"
            machines.append(m)
            center_of[m] = p

    # 2. 決策變數
    horizon = sum(d for ops in jobs_data.values() for d, _ in ops) + max(release_dates.values())
    start, end, assign, tardiness = {}, {}, {}, {}

    # Named tuple to store information about created variables.
    task_type = collections.namedtuple("task_type", "start end interval")
    # Creates job intervals and add to the corresponding machine lists.
    all_tasks = {}

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

    # 3. 工序順序
    for j, ops in jobs_data.items():
        for o in range(len(ops) - 1):
            model.add(all_tasks[j, o + 1].start >= all_tasks[j, o].end)

    # 4. 機台互斥 + 序依換線
    for m in machines:
        tasks = [(j, o) for j, ops in jobs_data.items()
                        for o, (_, centers) in enumerate(ops)
                        if center_of[m] in centers]
        for (j1, o1), (j2, o2) in itertools.combinations(tasks, 2):
            b = model.new_bool_var(f"ord_{j1}o{o1}_{j2}o{o2}_on_{m}")
            p = center_of[m]
            s12 = setup_times[p][((j1, o1), (j2, o2))]
            s21 = setup_times[p][((j2, o2), (j1, o1))]
            model.add(all_tasks[j2, o2].start >= all_tasks[j1, o1].end + s12).only_enforce_if([assign[j1, o1, m], assign[j2, o2, m], b])
            model.add(all_tasks[j1, o1].start >= all_tasks[j2, o2].end + s21).only_enforce_if([assign[j1, o1, m], assign[j2, o2, m], b.Not()])

    # 5. 拖期與目標
    for j, ops in jobs_data.items():
        last = len(ops) - 1
        tardiness[j] = model.new_int_var(0, horizon, f"T_j{j}")
        model.add_max_equality(tardiness[j],[0, all_tasks[j, last].end - due_dates[j]])

    makespan = model.new_int_var(0, horizon, "makespan")
    model.add_max_equality(
        makespan,
        [all_tasks[j, len(jobs_data[j]) - 1].end for j in range(len(jobs_data))],
    )
    model.minimize(alpha * makespan + (1 - alpha) * sum(weights[j] * tardiness[j] for j in jobs_data))

    # 6. 求解
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 1
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

def generate_random_instance(num_jobs = 50, centers = ('C1','C2','C3',"C4","C5","C6"),
                             center_caps = {'C1':3,'C2':2,'C3':3,"C4":2,"C5":3,"C6":2},
                             num_ops = {'C1':1,'C2':1,'C3':1,"C4":1,"C5":1,"C6":1}):

    jobs_data, release_dates, due_dates, weights = {}, {}, {}, {}
    setup_times = {p:{} for p in centers}

    for j in range(num_jobs):
        ops = []
        total_dur = 0
        sample = sorted(random.sample(centers, 4), key = lambda s: int(s[1]))
        for c in sample:
            for _ in range(num_ops[c]):
                dur = random.randint(20, 50)
                total_dur += dur
                ops.append((dur, c))

        jobs_data[j] = ops
        due_dates[j]     = random.randint(np.floor(total_dur * 2), np.floor(total_dur * 4))
        release_dates[j] = random.randint(0, (due_dates[j] - total_dur) // 2)
        weights[j]       = random.randint(1, 5)

    # 隨機設定序依換線時間
    for p in centers:
        all_ops = [(j, o) for j, ops in jobs_data.items()
                          for o in range(len(ops))
                          if p in ops[o][1]]

        for (j1,o1), (j2,o2) in itertools.permutations(all_ops, 2):
            setup_times[p][((j1,o1),(j2,o2))] = random.randint(5, 10)

    return jobs_data, release_dates, due_dates, weights, setup_times, center_caps

def plot_gantt(schedule, title="Gantt Chart"):
    # 1. 準備 y 軸：機台列表與對應座標
    machines = sorted({ m for ops in schedule.values() for _, m, _, _ in ops })
    y_pos = { m: i for i, m in enumerate(machines) }

    # 2. 顏色映射 (以 job_id 分組上色)
    cmap = plt.get_cmap("tab20")
    job_ids = sorted(schedule)
    color_map = { j: cmap(i % 20) for i, j in enumerate(job_ids) }

    # 3. 繪圖
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

    # 4. 美化
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
        # 準備所有機台與時線
        machines = [f"{c}_{k}"
                    for c, cap in self.center_caps.items()
                    for k in range(cap)]
        mc_timeline = {m: [] for m in machines}

        # 記錄每道工序 (j,o) 的 (start, end)
        doc_times = {}
        unscheduled = set(machine_assign.keys())

        if not priority:
            while unscheduled:
                ready = [(j, o) for (j, o) in unscheduled if o == 0 or (j, o-1) in doc_times]
                # 計算每個 ready 工序的最早啟動
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

                    # 選擇啟動時間最小者
                sel = min(earliest, key=earliest.get)
                st = earliest[sel]
                dur, _ = self.jobs_data[sel[0]][sel[1]]
                end = st + dur

                # 更新機台時線、doc_times、unscheduled
                mc_timeline[machine_assign[sel]].append((end, sel))
                doc_times[sel] = (st, end)
                unscheduled.remove(sel)
        else:
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
        schedule = {}
        for (j,o), (st,end) in times.items():
            m = machine_assign[(j,o)]
            schedule.setdefault(j, []).append((o, m, st, end))

        return schedule
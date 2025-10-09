from typing import Dict, List, Tuple, Optional, Any
from ortools.sat.python import cp_model
import math
import pprint

from helpers import (
    compute_shift_hours,
    aligned_week_windows,
    effective_weekly_cap,
    build_shift_skill_demands,
)

SKILLS = ["MD", "N", "D"]

TEAM_REQUIREMENTS: Dict[str, Dict[str, int]] = {
    "ADV": {"MD": 1, "N": 1, "D": 1},
    "BAS": {"N": 1, "D": 1},
    "MOTO": {"N": 2},
}


def solve_max_covered_shifts(
        worker_list: List[Tuple[str, int]],
        days: int = 30,
        teams_per_day_shift: Dict[str, int] = {"ADV": 7, "BAS": 30, "MOTO": 11},
        teams_per_night_shift: Dict[str, int] = {"ADV": 7, "BAS": 30},
        # Weekly flexibility:
        weekly_soft_overage: int = 2,  # e.g., +10% per week: 44 when cap=40
        rolling_weeks_for_soft: int = 4,  # total over any K aligned weeks <= K * cap (e.g., 160 for K=4)
        # Balance constraint:
        max_shift_imbalance: Optional[int] = None,  # max difference in met demand between any two shifts
        # Solve options:
        time_limit: float = 60.0,
        num_search_workers: int = 8,
        use_tiebreak_fill_positions: bool = True,
) -> Dict[str, Any]:
    """
    Maximize the number of fully covered shifts over a horizon with a fixed workforce,
    provide granular team coverage accounting, and enforce the MOTO weekend rule.

    Args:
      worker_list: list of tuples (skill, weekly_hour_cap).
      days: planning horizon (default 30).
      teams_per_day_shift: team counts for morning/afternoon shifts (default fixed EMS mix).
      teams_per_night_shift: team counts for night shifts (default fixed EMS mix).
      weekly_soft_overage: per-week overage allowed (e.g., 2 allows 42 when cap=40).
      rolling_weeks_for_soft: rolling aligned K-week window cap K * personal_weekly_cap.
      max_shift_imbalance: maximum allowed difference in met demand between any two shifts (None to disable).
      time_limit: CP-SAT time limit in seconds.
      num_search_workers: CP-SAT parallel workers.
      use_tiebreak_fill_positions: tie-break objective prefers more filled slots after maximizing shifts.

    """

    # Shifts and demands (with MOTO weekend rule)
    shifts = days * 3
    shift_hours = compute_shift_hours(days)
    demand_by_shift = build_shift_skill_demands(
        days, teams_per_day_shift, teams_per_night_shift, SKILLS, TEAM_REQUIREMENTS
    )
    windows = aligned_week_windows(shifts)
    weeks = math.ceil(shifts / 21)

    # Group workers by skill and collect capacities
    workers_by_skill: Dict[str, List[int]] = {skill: [] for skill in SKILLS}
    for (skill, cap) in worker_list:
        workers_by_skill[skill].append(int(cap))

    # Per-worker caps (effective per week, soft weekly cap, K-week cap)
    eff_cap_by_skill: Dict[str, List[int]] = {}
    weekly_soft_cap_nom_by_skill: Dict[str, List[int]] = {}
    rolling_cap_nom_by_skill: Dict[str, List[int]] = {}
    workforce_count_by_skill: Dict[str, int] = {}
    for skill in SKILLS:
        eff_cap_by_skill[skill] = []
        weekly_soft_cap_nom_by_skill[skill] = []
        rolling_cap_nom_by_skill[skill] = []
        workforce_count_by_skill[skill] = len(workers_by_skill[skill])
        for cap in workers_by_skill[skill]:
            eff_cap_by_skill[skill].append(effective_weekly_cap(cap))
            weekly_soft_cap_nom_by_skill[skill].append(int(cap + weekly_soft_overage))
            rolling_cap_nom_by_skill[skill].append(int(cap * rolling_weeks_for_soft))

    # Model
    model = cp_model.CpModel()

    # Decision variables
    workers_assigned = {}  # (skill, i, s) -> BoolVar: assign worker i (skill) to shift s
    hours_var = {}  # (skill, i) -> IntVar: total hours across horizon
    full_coverage = []  # per-shift "fully covered" indicator

    # Create variables and constraints
    for skill in SKILLS:
        n = workforce_count_by_skill[skill]
        for i in range(n):
            # Tight Big-M for horizon hours: weeks * effective cap of this worker
            M_i = weeks * eff_cap_by_skill[skill][i] if weeks > 0 else 0
            hours_var[(skill, i)] = model.NewIntVar(0, M_i, f"h_{skill}_{i}")
            for shift in range(shifts):
                workers_assigned[(skill, i, shift)] = model.NewBoolVar(f"workers_assigned{skill}_{i}_{shift}")

            # Total hours accumulation
            model.Add(hours_var[(skill, i)] == sum(
                workers_assigned[(skill, i, shift)] * shift_hours[shift] for shift in range(shifts)))

            # No 3 consecutive shifts (rest 6h after 18h)
            for shift in range(shifts - 2):
                model.Add(
                    workers_assigned[(skill, i, shift)] + workers_assigned[(skill, i, shift + 1)] + workers_assigned[
                        (skill, i, shift + 2)] <= 2)

            # Weekly soft cap per aligned week window
            for (start, end) in windows:
                model.Add(
                    sum(workers_assigned[(skill, i, shift)] * shift_hours[shift] for shift in range(start, end))
                    <= weekly_soft_cap_nom_by_skill[skill][i]
                )

            # Rolling K-week cap (aligned)
            if len(windows) >= rolling_weeks_for_soft:
                for w in range(len(windows) - rolling_weeks_for_soft + 1):
                    start = windows[w][0]
                    end = windows[w + rolling_weeks_for_soft - 1][1]
                    model.Add(
                        sum(workers_assigned[(skill, i, shift)] * shift_hours[shift] for shift in range(start, end))
                        <= rolling_cap_nom_by_skill[skill][i]
                    )

    for shift in range(shifts):
        # Shift fully covered indicator via skill demands:
        # full_coverage[s] = 1 -> for all skills with positive demand: sum y == demand (via >= and <= together)        
        c_s = model.NewBoolVar(f"full_coverage_{shift}")
        full_coverage.append(c_s)
        for skill in SKILLS:
            demand = demand_by_shift[shift].get(skill, 0)
            if demand > 0:
                model.Add(
                    sum(workers_assigned[(skill, i, shift)] for i in range(workforce_count_by_skill[skill]))
                    >= demand * c_s
                )

            # Coverage constraints per shift and skill: do not exceed demand
            if demand == 0:
                for i in range(workforce_count_by_skill[skill]):
                    model.Add(workers_assigned[(skill, i, shift)] == 0)
            else:
                model.Add(
                    sum(workers_assigned[(skill, i, shift)] for i in range(workforce_count_by_skill[skill])) <= demand)

    # Balance constraint: limit difference in met demand between shifts
    if max_shift_imbalance is not None:
        # Calculate total demand per shift (for upper bound)
        max_demand_per_shift = max(sum(demand_by_shift[s].get(skill, 0) for skill in SKILLS) for s in range(shifts))

        # Create variables for met demand per shift
        met_demand = []
        for shift in range(shifts):
            met_demand_s = model.NewIntVar(0, max_demand_per_shift, f"met_demand_{shift}")
            model.Add(
                met_demand_s == sum(
                    workers_assigned[(skill, i, shift)]
                    for skill in SKILLS
                    for i in range(workforce_count_by_skill[skill])
                )
            )
            met_demand.append(met_demand_s)

        # Create min and max met demand variables
        min_met_demand = model.NewIntVar(0, max_demand_per_shift, "min_met_demand")
        max_met_demand = model.NewIntVar(0, max_demand_per_shift, "max_met_demand")

        # Constrain min and max
        model.AddMinEquality(min_met_demand, met_demand)
        model.AddMaxEquality(max_met_demand, met_demand)

        # Enforce balance constraint: max - min <= max_shift_imbalance
        model.Add(max_met_demand - min_met_demand <= max_shift_imbalance)

    # Objective: maximize number of fully covered shifts (c),
    # tie-break: maximize filled positions (sum y without exceeding demands).

    total_positions_demand = sum(
        sum(demand_by_shift[shift].get(skill, 0) for skill in SKILLS) for shift in range(shifts))
    if use_tiebreak_fill_positions:
        big_W = total_positions_demand + 1  # ensures 1 more full shift dominates any change in filled slots
        model.Maximize(
            big_W * sum(full_coverage[shift] for shift in range(shifts)) +
            sum(workers_assigned[(skill, i, s)] for skill in SKILLS
                for i in range(workforce_count_by_skill[skill])
                for s in range(shifts))
        )
    else:
        model.Maximize(sum(full_coverage[shift] for shift in range(shifts)))

    # Solve
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = float(time_limit)
    solver.parameters.num_search_workers = int(num_search_workers)
    # solver.parameters.log_search_progress = True  # Uncomment for debug
    solver.Solve(model)

    return solver, workforce_count_by_skill, total_positions_demand, full_coverage, workers_assigned

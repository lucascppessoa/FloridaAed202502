#!/usr/bin/env python3
"""
EMS staffing optimization (max coverage with granular team accounting) using OR-Tools CP-SAT.

Scenario:
- Ephemeral teams per shift; exact skill totals ensure feasible team formation.
- Three shifts per day with hours [6, 6, 12] repeating (Morning, Afternoon, Night).
- You supply a fixed workforce as a list of tuples: ('skill', weekly_hour_cap).
  Example: ('MD', 10) is a medical doctor with a 10h/week cap.
- Objective:
    Maximize the number of fully covered shifts (all team slots present for that shift).
    Tie-breaker: maximize filled positions (sum of y).
- Granular team coverage in the summary:
    For each shift, compute how many teams of each type (ADV/BAS/MOTO) are fully formable
    from the assigned skills; report totals (e.g., 41/42 rather than 0/1).

Rules/constraints (as before, adapted for fixed workforce) AND new MOTO weekend rule:
- Coverage by skill per shift cannot exceed demand.
- A shift is "fully covered" if all team slots of that shift are formable (accounting teams per type).
- Rest rule: no worker may work three consecutive shifts anywhere in the sequence.
- Weekly hours flexibility:
    * In each aligned week (21-shift window aligned to Sunday 06:00),
      a worker may go up to weekly_soft_cap = (overage + personal_weekly_cap).
      Given 6/12h shifts, realizable totals are multiples of 6;
    * Over any rolling K-week window (aligned, default K=4), the total must be ≤ K * personal_weekly_cap.

Notes:
- We do NOT build explicit teams in the model. After solving, we compute the maximum number
  of teams per type that can be formed from the assigned skill counts in each shift via a small
  integer calculation, and report those team counts.
- Weeks are aligned to Sunday 06:00 with shift index 0 corresponding to Sunday morning.

Run:
  pip install ortools
  python ems_cp_sat_max_coverage_with_moto_weekend.py
"""

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
from summary import (print_summary, generate_summary)

SKILLS = ["MD", "N", "NA", "D"]

TEAM_REQUIREMENTS: Dict[str, Dict[str, int]] = {
    "ADV": {"MD": 1, "N": 1, "D": 1},
    "BAS": {"NA": 1, "D": 1},
    "MOTO": {"N": 1, "NA": 1},
}


def solve_max_covered_shifts(
    worker_list: List[Tuple[str, int]],
    days: int = 30,
    teams_per_day_shift: Dict[str, int] = {"ADV": 7, "BAS": 30, "MOTO": 11},
    teams_per_night_shift: Dict[str, int] = {"ADV": 7, "BAS": 30},
    # Weekly flexibility:
    weekly_soft_overage: int = 2,  # e.g., +10% per week: 44 when cap=40
    rolling_weeks_for_soft: int = 4,    # total over any K aligned weeks <= K * cap (e.g., 160 for K=4)
    # Balance constraint:
    max_shift_imbalance: Optional[int] = None,  # max difference in met demand between any two shifts
    # Solve options:
    time_limit: Optional[float] = 60.0,
    num_search_workers: Optional[int] = None,
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

    # A summary of day/night demands considering the weekend rule
    # Day (weekday): team mix as provided
    day_weekday_demand = {skill: 0 for skill in SKILLS}
    for team, cnt in teams_per_day_shift.items():
        if team not in TEAM_REQUIREMENTS:
            continue
        for skill, req in TEAM_REQUIREMENTS[team].items():
            day_weekday_demand[skill] += cnt * req
    # Day (weekend): MOTO=0
    day_weekend_demand = {skill: 0 for skill in SKILLS}
    for team, cnt in {"ADV": teams_per_day_shift.get("ADV", 0),
                      "BAS": teams_per_day_shift.get("BAS", 0),
                      "MOTO": 0}.items():
        if team not in TEAM_REQUIREMENTS:
            continue
        for skill, req in TEAM_REQUIREMENTS[team].items():
            day_weekend_demand[skill] += cnt * req
    # Night (no MOTO)
    night_demand = {skill: 0 for skill in SKILLS}
    for team, cnt in teams_per_night_shift.items():
        if team not in TEAM_REQUIREMENTS:
            continue
        for skill, req in TEAM_REQUIREMENTS[team].items():
            night_demand[skill] += cnt * req
    demand_per_shift_summary = {
        "day_weekday": day_weekday_demand,
        "day_weekend": day_weekend_demand,
        "night": night_demand,
    }

    # Total positions demanded (across all skills/shifts)
    total_positions_demand = sum(sum(demand_by_shift[s].get(skill, 0) for skill in SKILLS) for s in range(shifts))

    # Group workers by skill and collect capacities
    workers_by_skill: Dict[str, List[int]] = {skill: [] for skill in SKILLS}
    for (skill, cap) in worker_list:
        if skill not in workers_by_skill:
            raise ValueError(f"Unknown skill {skill}. Allowed: {SKILLS}")
        if cap < 0:
            raise ValueError(f"Weekly cap must be non-negative for worker ({skill}, {cap}).")
        workers_by_skill[skill].append(int(cap))

    workforce_count_by_skill = {skill: len(workers_by_skill[skill]) for skill in SKILLS}

    # Per-worker caps (effective per week, soft weekly cap, K-week cap)
    eff_cap_by_skill: Dict[str, List[int]] = {skill: [] for skill in SKILLS}
    weekly_soft_cap_nom_by_skill: Dict[str, List[int]] = {skill: [] for skill in SKILLS}
    rolling_cap_nom_by_skill: Dict[str, List[int]] = {skill: [] for skill in SKILLS}
    for skill in SKILLS:
        for cap in workers_by_skill[skill]:
            eff_cap_by_skill[skill].append(effective_weekly_cap(cap))
            weekly_soft_cap_nom_by_skill[skill].append(int(cap + weekly_soft_overage))
            rolling_cap_nom_by_skill[skill].append(int(cap * rolling_weeks_for_soft))

    # Model
    model = cp_model.CpModel()

    # Decision variables
    y = {}         # (skill, i, s) -> BoolVar: assign worker i (skill) to shift s
    hours_var = {} # (skill, i) -> IntVar: total hours across horizon
    c = []         # per-shift "fully covered" indicator

    # Create variables and constraints
    for skill in SKILLS:
        n = workforce_count_by_skill[skill]
        for i in range(n):
            # Tight Big-M for horizon hours: weeks * effective cap of this worker
            M_i = weeks * eff_cap_by_skill[skill][i] if weeks > 0 else 0
            hours_var[(skill, i)] = model.NewIntVar(0, M_i, f"h_{skill}_{i}")
            for shift in range(shifts):
                y[(skill, i, shift)] = model.NewBoolVar(f"y_{skill}_{i}_{shift}")

            # Total hours accumulation
            model.Add(hours_var[(skill, i)] == sum(y[(skill, i, shift)] * shift_hours[shift] for shift in range(shifts)))

            # No 3 consecutive shifts (rest 6h after 18h)
            for shift in range(shifts - 2):
                model.Add(y[(skill, i, shift)] + y[(skill, i, shift + 1)] + y[(skill, i, shift + 2)] <= 2)

            # Weekly soft cap per aligned week window
            for (start, end) in windows:
                model.Add(
                    sum(y[(skill, i, shift)] * shift_hours[shift] for shift in range(start, end))
                    <= weekly_soft_cap_nom_by_skill[skill][i]
                )

            # Rolling K-week cap (aligned)
            if len(windows) >= rolling_weeks_for_soft:
                for w in range(len(windows) - rolling_weeks_for_soft + 1):
                    start = windows[w][0]
                    end = windows[w + rolling_weeks_for_soft - 1][1]
                    model.Add(
                        sum(y[(skill, i, shift)] * shift_hours[shift] for shift in range(start, end))
                        <= rolling_cap_nom_by_skill[skill][i]
                    )

    # Coverage constraints per shift and skill: do not exceed demand
    for shift in range(shifts):
        for skill in SKILLS:
            demand = demand_by_shift[shift].get(skill, 0)
            if demand == 0:
                for i in range(workforce_count_by_skill[skill]):
                    model.Add(y[(skill, i, shift)] == 0)
            else:
                model.Add(sum(y[(skill, i, shift)] for i in range(workforce_count_by_skill[skill])) <= demand)

    # Shift fully covered indicator via skill demands:
    # c[s] = 1 -> for all skills with positive demand: sum y == demand (via >= and <= together)
    for shift in range(shifts):
        c_s = model.NewBoolVar(f"c_{shift}")
        c.append(c_s)
        for skill in SKILLS:
            demand = demand_by_shift[shift].get(skill, 0)
            if demand > 0:
                model.Add(
                    sum(y[(skill, i, shift)] for i in range(workforce_count_by_skill[skill]))
                    >= demand * c_s
                )

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
                    y[(skill, i, shift)]
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
    total_positions_demand = sum(sum(demand_by_shift[shift].get(skill, 0) for skill in SKILLS) for shift in range(shifts))
    if use_tiebreak_fill_positions:
        big_W = total_positions_demand + 1  # ensures 1 more full shift dominates any change in filled slots
        model.Maximize(
            big_W * sum(c[shift] for shift in range(shifts)) +
            sum(y[(skill, i, s)] for skill in SKILLS
                for i in range(workforce_count_by_skill[skill])
                for s in range(shifts))
        )
    else:
        model.Maximize(sum(c[shift] for shift in range(shifts)))

    # Solve
    solver = cp_model.CpSolver()
    if time_limit is not None:
        solver.parameters.max_time_in_seconds = float(time_limit)
    if num_search_workers is not None:
        solver.parameters.num_search_workers = int(num_search_workers)
    # solver.parameters.log_search_progress = True  # Uncomment for debug
    solver.Solve(model)

    return solver, workforce_count_by_skill, total_positions_demand, demand_per_shift_summary, c, y

    


if __name__ == "__main__":
    # Build a small workforce (weekly caps in hours)
    days = 30
    teams_per_day_shift = {"ADV": 7, "BAS": 30, "MOTO": 11}
    teams_per_night_shift = {"ADV": 7, "BAS": 30}
    weekly_soft_overage = 2
    rolling_weeks_for_soft = 4
    demo_workers: List[Tuple[str, int]] = []
    demo_workers += [("N", 10)] * 5
    demo_workers += [("N", 20)] * 14
    demo_workers += [("N", 24)] * 0
    demo_workers += [("N", 30)] * 3
    demo_workers += [("N", 40)] * 33

    demo_workers += [("NA", 10)] * 24
    demo_workers += [("NA", 20)] * 52
    demo_workers += [("NA", 24)] * 8
    demo_workers += [("NA", 30)] * 12
    demo_workers += [("NA", 40)] * 106

    demo_workers += [("MD", 10)] * 5
    demo_workers += [("MD", 20)] * 14
    demo_workers += [("MD", 24)] * 0
    demo_workers += [("MD", 30)] * 0
    demo_workers += [("MD", 40)] * 1

    demo_workers += [("D", 10)] * 7
    demo_workers += [("D", 20)] * 15
    demo_workers += [("D", 24)] * 1
    demo_workers += [("D", 30)] * 3
    demo_workers += [("D", 40)] * 85

    solver, workforce_count_by_skill, total_positions_demand, demand_per_shift_summary, c, y = solve_max_covered_shifts(
        teams_per_day_shift=teams_per_day_shift,
        teams_per_night_shift=teams_per_night_shift,
        days=days,
        weekly_soft_overage=weekly_soft_overage,
        rolling_weeks_for_soft=rolling_weeks_for_soft,
        worker_list=demo_workers,
        max_shift_imbalance=10,  # Max difference of 10 workers between any two shifts
        time_limit=300.0,
        num_search_workers=12,
    )
    summary = generate_summary(solver,
                                workforce_count_by_skill,
                                total_positions_demand,
                                teams_per_night_shift,
                                teams_per_day_shift,
                                days,
                                weekly_soft_overage,
                                rolling_weeks_for_soft,
                                demand_per_shift_summary,
                                c,
                                y)
    print_summary(summary)
    # pprint.pprint(result)
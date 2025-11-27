#!/usr/bin/env python3
"""
EMS staffing optimization (maximize team slots covered) using OR-Tools CP-SAT.

Scenario:
- Ephemeral teams per shift with explicit team formation modeling.
- Three shifts per day with hours [6, 6, 12] repeating (Morning, Afternoon, Night).
- You supply a fixed workforce as a list of tuples: ('skill', weekly_hour_cap).
  Example: ('MD', 10) is a medical doctor with a 10h/week cap.
- Objective:
    Maximize the total weighted number of team slots covered across all shifts.
    Teams can be weighted differently (e.g., ADV teams worth 2x, others 1x) to prioritize valuable teams.
    Tie-breaker: maximize the number of fully covered shifts (all teams present in a shift).
- Team formation is explicitly modeled in the optimization:
    Teams are decision variables with constraints ensuring proper skill allocation.
    ADV teams: 1 MD, 1 N, 1 D each
    BAS teams: 1 N, 1 D each
    MOTO teams: 2 N each

Rules/constraints (adapted for fixed workforce) AND MOTO weekend rule:
- MOTO teams only operate on weekday day shifts (no nights, no weekends).
- Team formation respects skill availability and team supply limits per shift.
- Rest rule: no worker may work three consecutive shifts anywhere in the sequence.
- Weekly hours flexibility:
    * In each aligned week (21-shift window aligned to Monday 06:00),
      a worker may go up to weekly_soft_cap = (overage + personal_weekly_cap).
      Given 6/12h shifts, realizable totals are multiples of 6;
    * Over any rolling K-week window (aligned, default K=4), the total must be ≤ K * personal_weekly_cap.
- Optional per-shift-type balance constraint: limit max difference in team counts within each
    shift type (morning, afternoon, night balanced separately).
    (e.g., {"ADV": 2, "BAS": 8} ensures ADV teams vary by at most 2 among morning shifts,
    at most 2 among afternoon shifts, and at most 2 among night shifts).
- Optional team weights: assign different values to team types in the objective
    (e.g., {"ADV": 2} makes ADV teams worth 2x, prioritizing their formation).

Notes:
- Teams are explicit decision variables in the CP-SAT model.
- Weeks are aligned to Monday 06:00 with shift index 0 corresponding to Monday morning.

Run:
  pip install ortools pandas
  python main.py
"""

from typing import List, Tuple
from summary import (print_summary, generate_summary, generate_team_csv_files, 
                     validate_worker_constraints, print_validation_results)
from solver import solve_max_covered_shifts

if __name__ == "__main__":
    # Build a small workforce (weekly caps in hours)
    days = 30
    teams_per_day_shift = {"ADV": 7, "BAS": 30, "MOTO": 11}
    teams_per_night_shift = {"ADV": 7, "BAS": 30}
    weekly_soft_overage = 2
    rolling_weeks_for_soft = 4
    
    # Per-shift-type imbalance constraint (optional):
    # None or {} = no constraint
    # {"ADV": 2, "BAS": 8} = ADV teams vary by max 2, BAS by max 8 within each shift type
    # Balancing is done separately for morning, afternoon, and night shifts
    # This prevents the optimizer from leaving night shifts empty while filling day shifts
    # Note: MOTO typically not constrained due to weekend/night rules (always 0 at night/weekend)
    shift_type_imbalance = {"ADV": 2, "BAS": 8}  # Balance teams within each shift type
    
    # Team targets (soft floor with penalty):
    # None or {} = no target floor
    # {"ADV": 3} = try to get at least 3 ADV teams per shift
    # The optimizer will prioritize reaching targets before maximizing total coverage
    # If targets are impossible (e.g., not enough MDs), it falls back gracefully
    # rather than failing with INFEASIBLE
    team_targets = {"ADV": 2, "BAS": 15, "MOTO": 5}  # Target at least 3 ADV teams per shift
    
    # Team weights for objective function (optional):
    # None or {} = all teams have weight 1 (equal priority)
    # {"ADV": 2} = ADV teams worth 2x, others worth 1x
    # {"ADV": 2, "BAS": 1, "MOTO": 1} = explicit weights for all types
    # Higher weight = higher priority in optimization
    team_weights = {"ADV": 8, "BAS": 2, "MOTO": 1}  # ADV teams are more valuable
    
    demo_workers: List[Tuple[str, int]] = []
    demo_workers += [("N", 10)] * 5
    demo_workers += [("N", 20)] * 14
    demo_workers += [("N", 24)] * 0
    demo_workers += [("N", 30)] * 3
    demo_workers += [("N", 40)] * 33

    demo_workers += [("N", 10)] * 24
    demo_workers += [("N", 20)] * 52
    demo_workers += [("N", 24)] * 8
    demo_workers += [("N", 30)] * 12
    demo_workers += [("N", 40)] * 106

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

    solver, workforce_count_by_skill, total_positions_demand, full_coverage, workers_assigned, teams_formed = solve_max_covered_shifts(
        teams_per_day_shift=teams_per_day_shift,
        teams_per_night_shift=teams_per_night_shift,
        days=days,
        weekly_soft_overage=weekly_soft_overage,
        rolling_weeks_for_soft=rolling_weeks_for_soft,
        worker_list=demo_workers,
        shift_type_imbalance=None,
        team_targets=team_targets,
        team_weights=team_weights,
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
                                full_coverage,
                                workers_assigned,
                                teams_formed,
                                True)
    print_summary(summary)
    generate_team_csv_files(summary, days)
    
    # Validate the solution
    validation = validate_worker_constraints(
        summary=summary,
        worker_list=demo_workers,
        weekly_soft_overage=weekly_soft_overage,
        rolling_weeks_for_soft=rolling_weeks_for_soft
    )
    print_validation_results(validation)
    # pprint.pprint(result)
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
- Weeks are aligned to Monday 06:00 with shift index 0 corresponding to Monday morning.

Run:
  pip install ortools
  python ems_cp_sat_max_coverage_with_moto_weekend.py
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
        max_shift_imbalance=None,  # Max difference of 10 workers between any two shifts
        time_limit=600.0,
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
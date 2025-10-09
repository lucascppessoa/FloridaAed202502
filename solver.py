from typing import Dict, List, Tuple, Optional, Any
from ortools.sat.python import cp_model
import math
import pprint

from helpers import (
    compute_shift_hours,
    aligned_week_windows,
    effective_weekly_cap,
    build_shift_skill_demands,
    is_night_shift,
    is_weekend_day,
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
        max_shift_imbalance: Optional[Dict[str, int]] = None,  # max imbalance per team type (e.g., {"ADV": 2, "BAS": 5})
        # Solve options:
        time_limit: float = 60.0,
        num_search_workers: int = 8,
        use_tiebreak_fill_positions: bool = True,
) -> Dict[str, Any]:
    """
    Maximize the number of team slots covered over a horizon with a fixed workforce,
    with explicit team formation modeling and enforcement of the MOTO weekend rule.

    Args:
      worker_list: list of tuples (skill, weekly_hour_cap).
      days: planning horizon (default 30).
      teams_per_day_shift: team counts for morning/afternoon shifts (default fixed EMS mix).
      teams_per_night_shift: team counts for night shifts (default fixed EMS mix).
      weekly_soft_overage: per-week overage allowed (e.g., 2 allows 42 when cap=40).
      rolling_weeks_for_soft: rolling aligned K-week window cap K * personal_weekly_cap.
      max_shift_imbalance: dict mapping team type to max allowed difference in team count between shifts
                           (e.g., {"ADV": 2, "BAS": 5}). Missing keys or None values disable constraint for that type.
      time_limit: CP-SAT time limit in seconds.
      num_search_workers: CP-SAT parallel workers.
      use_tiebreak_fill_positions: tie-break objective prefers more fully covered shifts after maximizing team slots.

    Returns:
      Tuple containing: (solver, workforce_count_by_skill, total_positions_demand, 
                        full_coverage, workers_assigned, teams_formed)
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

    # Team formation variables and constraints
    teams_formed = {}  # (shift, team_type) -> IntVar: number of teams formed
    
    for shift in range(shifts):
        # Determine team supply for this shift (accounting for MOTO weekend rule)
        if is_night_shift(shift):
            team_supply = {
                "ADV": teams_per_night_shift.get("ADV", 0),
                "BAS": teams_per_night_shift.get("BAS", 0),
                "MOTO": 0,
            }
        elif is_weekend_day(shift):
            team_supply = {
                "ADV": teams_per_day_shift.get("ADV", 0),
                "BAS": teams_per_day_shift.get("BAS", 0),
                "MOTO": 0,  # weekend rule
            }
        else:
            team_supply = {
                "ADV": teams_per_day_shift.get("ADV", 0),
                "BAS": teams_per_day_shift.get("BAS", 0),
                "MOTO": teams_per_day_shift.get("MOTO", 0),
            }
        
        # Create team formation variables
        for team_type in ["ADV", "BAS", "MOTO"]:
            max_teams = team_supply[team_type]
            teams_formed[(shift, team_type)] = model.NewIntVar(0, max_teams, f"teams_{shift}_{team_type}")
        
        # Skill assignments for this shift
        md_assigned = sum(workers_assigned[("MD", i, shift)] for i in range(workforce_count_by_skill["MD"]))
        n_assigned = sum(workers_assigned[("N", i, shift)] for i in range(workforce_count_by_skill["N"]))
        d_assigned = sum(workers_assigned[("D", i, shift)] for i in range(workforce_count_by_skill["D"]))
        
        # Team formation constraints
        # ADV teams need: 1 MD, 1 N, 1 D each
        model.Add(teams_formed[(shift, "ADV")] <= md_assigned)
        model.Add(teams_formed[(shift, "ADV")] <= d_assigned)
        
        # BAS teams need: 1 N, 1 D each
        # MOTO teams need: 2 N each
        # Combined constraint for N usage: ADV uses 1N, BAS uses 1N, MOTO uses 2N
        model.Add(
            teams_formed[(shift, "ADV")] + 
            teams_formed[(shift, "BAS")] + 
            2 * teams_formed[(shift, "MOTO")] 
            <= n_assigned
        )
        
        # Combined constraint for D usage: ADV uses 1D, BAS uses 1D
        model.Add(
            teams_formed[(shift, "ADV")] + 
            teams_formed[(shift, "BAS")] 
            <= d_assigned
        )
        
        # Full coverage indicator (kept for compatibility)
        c_s = model.NewBoolVar(f"full_coverage_{shift}")
        full_coverage.append(c_s)
        
        # Full coverage means all team slots are filled
        total_teams_required = team_supply["ADV"] + team_supply["BAS"] + team_supply["MOTO"]
        if total_teams_required > 0:
            model.Add(
                teams_formed[(shift, "ADV")] + 
                teams_formed[(shift, "BAS")] + 
                teams_formed[(shift, "MOTO")] 
                >= total_teams_required * c_s
            )

    # Balance constraint: limit difference in team counts between shifts (per team type)
    if max_shift_imbalance is not None:
        for team_type in ["ADV", "BAS", "MOTO"]:
            # Only enforce if this team type is in the dict and has a non-None value
            if team_type in max_shift_imbalance and max_shift_imbalance[team_type] is not None:
                imbalance_limit = max_shift_imbalance[team_type]
                
                # Calculate supply limit (max possible teams of this type in any shift)
                team_supply_limit = max(
                    teams_per_day_shift.get(team_type, 0),
                    teams_per_night_shift.get(team_type, 0)
                )
                
                # Collect teams formed of this type across all shifts
                teams_by_shift = [teams_formed[(shift, team_type)] for shift in range(shifts)]
                
                # Create min and max variables for this team type (domain is [0, supply_limit])
                min_teams = model.NewIntVar(0, team_supply_limit, f"min_teams_{team_type}")
                max_teams = model.NewIntVar(0, team_supply_limit, f"max_teams_{team_type}")
                
                # Constrain min and max to actual values across shifts
                model.AddMinEquality(min_teams, teams_by_shift)
                model.AddMaxEquality(max_teams, teams_by_shift)
                
                # Enforce balance constraint for this team type
                model.Add(max_teams - min_teams <= imbalance_limit)

    # Objective: maximize total number of team slots covered across all shifts
    total_positions_demand = sum(
        sum(demand_by_shift[shift].get(skill, 0) for skill in SKILLS) for shift in range(shifts))
    
    # Primary objective: maximize teams formed
    total_teams_covered = sum(
        teams_formed[(shift, team_type)] 
        for shift in range(shifts) 
        for team_type in ["ADV", "BAS", "MOTO"]
    )
    
    if use_tiebreak_fill_positions:
        # Tie-break: prefer more balanced coverage (maximize fully covered shifts as secondary)
        big_W = shifts * (teams_per_day_shift.get("ADV", 0) + teams_per_day_shift.get("BAS", 0) + teams_per_day_shift.get("MOTO", 0)) + 1
        model.Maximize(
            total_teams_covered * big_W +
            sum(full_coverage[shift] for shift in range(shifts))
        )
    else:
        model.Maximize(total_teams_covered)

    # Solve
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = float(time_limit)
    solver.parameters.num_search_workers = int(num_search_workers)
    # solver.parameters.log_search_progress = True  # Uncomment for debug
    solver.Solve(model)

    return solver, workforce_count_by_skill, total_positions_demand, full_coverage, workers_assigned, teams_formed

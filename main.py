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

SKILLS = ["MD", "N", "NA", "D"]

TEAM_REQUIREMENTS: Dict[str, Dict[str, int]] = {
    "ADV": {"MD": 1, "N": 1, "D": 1},
    "BAS": {"NA": 1, "D": 1},
    "MOTO": {"N": 1, "NA": 1},
}


def _compute_shift_hours(days: int) -> List[int]:
    """List of shift durations in hours for the horizon. Pattern per day: [6, 6, 12]."""
    return [6, 6, 12] * days


def _aligned_week_windows(shifts: int) -> List[Tuple[int, int]]:
    """Aligned weekly windows (21 shifts), last one may be partial.
    Alignment: week boundary at shift index multiples of 21 starting at 0.
    Returns list of (start, end) indices, end exclusive."""
    windows = []
    start = 0
    while start < shifts:
        end = min(shifts, start + 21)
        windows.append((start, end))
        start += 21
    return windows


def _effective_weekly_cap(weekly_hours_cap: int) -> int:
    """Realizable weekly cap constrained by 6h/12h shifts: floor to nearest multiple of 6."""
    return (weekly_hours_cap // 6) * 6


def _is_weekend_day(shift_index: int) -> bool:
    dow = (shift_index // 3) % 7
    return dow in (5, 6)

def _is_night_shift(shift_index: int) -> bool:
    return shift_index % 3 == 2


def _dow_name(day_index: int) -> str:
    names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    return names[day_index % 7]


def _build_shift_skill_demands(
    days: int,
    teams_per_day_shift: Dict[str, int],
    teams_per_night_shift: Dict[str, int],
) -> List[Dict[str, int]]:
    """Build per-shift skill demands:
        No MOTO on nights and weekends
    """
    shifts = days * 3
    demand_by_shift: List[Dict[str, int]] = []
    for shift_index in range(shifts):
        # Construct team supply for this shift
        if _is_night_shift(shift_index) or _is_weekend_day(shift_index):
            team_supply = teams_per_night_shift
        else:
            team_supply = teams_per_day_shift

        # Convert team supply to skill demand
        demand = {skill: 0 for skill in SKILLS}
        for team, count in team_supply.items():
            if count <= 0 or team not in TEAM_REQUIREMENTS:
                continue
            for skill, req in TEAM_REQUIREMENTS[team].items():
                demand[skill] += count * req
        demand_by_shift.append(demand)
    return demand_by_shift


def _day_team_allocation_from_assigned(
    assigned: Dict[str, int], teams_per_day_shift: Dict[str, int]
) -> Dict[str, int]:
    """Given assigned skill counts for a day shift, compute max numbers of ADV, BAS, MOTO teams
    that can be formed (integer) without exceeding:
      - assigned skill counts
      - offered team slots from teams_per_day_shift

    Constraints (day):
      x_ADV <= MD
      x_ADV + x_MOTO <= N
      x_BAS + x_MOTO <= NA
      x_ADV + x_BAS <= D
      0 <= x_t <= team slots for t
    We enumerate on (ADV, BAS) and pick max MOTO greedily.

    Returns: dict {"ADV": int, "BAS": int, "MOTO": int}
    """
    md = assigned.get("MD", 0)
    n = assigned.get("N", 0)
    na = assigned.get("NA", 0)
    d = assigned.get("D", 0)
    adv_supply = int(teams_per_day_shift.get("ADV", 0))
    bas_supply = int(teams_per_day_shift.get("BAS", 0))
    moto_supply = int(teams_per_day_shift.get("MOTO", 0))

    best = {"ADV": 0, "BAS": 0, "MOTO": 0}
    best_total = -1

    adv_max = min(adv_supply, md, n, d)
    for adv in range(adv_max + 1):
        d_rem = d - adv
        if d_rem < 0:
            continue
        bas_max = min(bas_supply, d_rem, na)
        for bas in range(bas_max + 1):
            na_rem = na - bas
            n_rem = n - adv
            if na_rem < 0 or n_rem < 0:
                continue
            moto_max = min(moto_supply, na_rem, n_rem)
            if moto_max < 0:
                moto_max = 0
            total = adv + bas + moto_max
            if total > best_total:
                best_total = total
                best = {"ADV": adv, "BAS": bas, "MOTO": moto_max}
    return best


def _night_team_allocation_from_assigned(
    assigned: Dict[str, int], teams_per_night_shift: Dict[str, int]
) -> Dict[str, int]:
    """Given assigned skill counts for a night shift, compute max numbers of ADV, BAS teams
    that can be formed (integer) without exceeding:
      - assigned skill counts
      - offered team slots from teams_per_night_shift

    Constraints (night):
      x_ADV <= MD
      x_ADV <= N
      x_BAS <= NA
      x_ADV + x_BAS <= D
      0 <= x_t <= team slots for t
    We enumerate on ADV and pick max BAS greedily.

    Returns: dict {"ADV": int, "BAS": int, "MOTO": 0}
    """
    md = assigned.get("MD", 0)
    n = assigned.get("N", 0)
    na = assigned.get("NA", 0)
    d = assigned.get("D", 0)
    adv_supply = int(teams_per_night_shift.get("ADV", 0))
    bas_supply = int(teams_per_night_shift.get("BAS", 0))

    best_adv = 0
    best_bas = 0
    best_total = -1

    adv_max = min(adv_supply, md, n, d)
    for adv in range(adv_max + 1):
        d_rem = d - adv
        if d_rem < 0:
            continue
        bas_max = min(bas_supply, d_rem, na)
        total = adv + bas_max
        if total > best_total:
            best_total = total
            best_adv = adv
            best_bas = bas_max

    return {"ADV": best_adv, "BAS": best_bas, "MOTO": 0}


def solve_max_covered_shifts(
    worker_list: List[Tuple[str, int]],
    days: int = 30,
    teams_per_day_shift: Dict[str, int] = {"ADV": 7, "BAS": 30, "MOTO": 11},
    teams_per_night_shift: Dict[str, int] = {"ADV": 7, "BAS": 30},
    # Weekly flexibility:
    weekly_soft_overage: int = 2,  # e.g., +10% per week: 44 when cap=40
    rolling_weeks_for_soft: int = 4,    # total over any K aligned weeks <= K * cap (e.g., 160 for K=4)
    # Solve options:
    include_assignments: bool = False,
    time_limit: Optional[float] = 60.0,
    num_search_workers: Optional[int] = None,
    add_symmetry_breaking: bool = True,
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
      include_assignments: return full assignment details when True.
      time_limit: CP-SAT time limit in seconds.
      num_search_workers: CP-SAT parallel workers.
      add_symmetry_breaking: z[i] >= z[i+1] per skill (light symmetry breaking).
      use_tiebreak_fill_positions: tie-break objective prefers more filled slots after maximizing shifts.

    Returns:
      summary dict with:
        - status
        - days, shifts_total
        - demand_per_shift_summary: {'day_weekday': {...}, 'day_weekend': {...}, 'night': {...}}
        - total_positions_demand
        - workforce_count_by_skill
        - weekly_soft_overage, rolling_weeks_for_soft
        - shifts_fully_covered, coverage_fraction
        - teams_covered_total, team_slots_total, teams_covered_fraction
        - teams_covered_by_type, team_slots_by_type_total
        - filled_positions_total, filled_positions_fraction
        - workers_used_by_skill, hours_stats_by_skill
        - c_by_shift (list)
        - team_coverage_by_shift (list of dicts per shift with covered and slots)
        - assignments (optional detailed fields)
    """

    # Shifts and demands (with MOTO weekend rule)
    shifts = days * 3
    shift_hours = _compute_shift_hours(days)
    demand_by_shift = _build_shift_skill_demands(
        days, teams_per_day_shift, teams_per_night_shift
    )
    windows = _aligned_week_windows(shifts)
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
            eff_cap_by_skill[skill].append(_effective_weekly_cap(cap))
            weekly_soft_cap_nom_by_skill[skill].append(int(cap + weekly_soft_overage))
            rolling_cap_nom_by_skill[skill].append(int(cap * rolling_weeks_for_soft))

    # Model
    model = cp_model.CpModel()

    # Decision variables
    y = {}         # (skill, i, s) -> BoolVar: assign worker i (skill) to shift s
    z = {}         # (skill, i) -> BoolVar: worker used at least once (OR of y's)
    hours_var = {} # (skill, i) -> IntVar: total hours across horizon
    c = []         # per-shift "fully covered" indicator

    # Create variables and constraints
    for skill in SKILLS:
        n = workforce_count_by_skill[skill]
        for i in range(n):
            z[(skill, i)] = model.NewBoolVar(f"z_{skill}_{i}")
            # Tight Big-M for horizon hours: weeks * effective cap of this worker
            M_i = weeks * eff_cap_by_skill[skill][i] if weeks > 0 else 0
            hours_var[(skill, i)] = model.NewIntVar(0, M_i, f"h_{skill}_{i}")
            for shift in range(shifts):
                y[(skill, i, shift)] = model.NewBoolVar(f"y_{skill}_{i}_{shift}")
                # Link y <= z
                model.Add(y[(skill, i, shift)] <= z[(skill, i)])

            # Total hours accumulation
            model.Add(hours_var[(skill, i)] == sum(y[(skill, i, shift)] * shift_hours[shift] for shift in range(shifts)))

            # z <= sum_s y (plus y <= z already added)
            model.Add(z[(skill, i)] <= sum(y[(skill, i, shift)] for shift in range(shifts)))

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

    # Optional symmetry breaking: prefer lower-index workers within each skill
    if add_symmetry_breaking:
        for skill in SKILLS:
            n = workforce_count_by_skill[skill]
            for i in range(n - 1):
                model.Add(z[(skill, i)] >= z[(skill, i + 1)])

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

    status_code = solver.Solve(model)
    status_map = {
        cp_model.OPTIMAL: "OPTIMAL",
        cp_model.FEASIBLE: "FEASIBLE",
        cp_model.INFEASIBLE: "INFEASIBLE",
        cp_model.MODEL_INVALID: "MODEL_INVALID",
        cp_model.UNKNOWN: "UNKNOWN",
    }
    status = status_map.get(status_code, str(status_code))

    # Extract results: fully covered shifts
    shifts_fully_covered = 0
    c_by_shift = []
    for shift in range(shifts):
        val = int(round(solver.Value(c[shift])))
        c_by_shift.append(val)
        if val == 1:
            shifts_fully_covered += 1
    shifts_total = shifts
    coverage_fraction = (shifts_fully_covered / shifts_total) if shifts_total > 0 else 0.0

    # Filled positions (for info/tie-break reporting)
    filled_positions_total = 0
    if status in ("OPTIMAL", "FEASIBLE"):
        for shift in range(shifts):
            for skill in SKILLS:
                filled_positions_total += sum(
                    int(solver.Value(y[(skill, i, shift)]))
                    for i in range(workforce_count_by_skill[skill])
                )
    filled_positions_fraction = (filled_positions_total / total_positions_demand) if total_positions_demand > 0 else 0.0

    # Used workers and hours stats
    workers_used_by_skill: Dict[str, int] = {}
    hours_stats_by_skill: Dict[str, Dict[str, int]] = {}
    for skill in SKILLS:
        n = workforce_count_by_skill[skill]
        used_ids = [i for i in range(n) if solver.Value(z[(skill, i)]) > 0.5]
        workers_used_by_skill[skill] = len(used_ids)
        if used_ids:
            hours_used = [int(solver.Value(hours_var[(skill, i)])) for i in used_ids]
            h_min = min(hours_used)
            h_max = max(hours_used)
            hours_stats_by_skill[skill] = {"min": h_min, "max": h_max, "spread": h_max - h_min}
        else:
            hours_stats_by_skill[skill] = {"min": 0, "max": 0, "spread": 0}

    # Granular team coverage accounting (post-solve from assigned skills)
    team_coverage_by_shift: List[Dict[str, Any]] = []
    teams_covered_by_type = {"ADV": 0, "BAS": 0, "MOTO": 0}
    team_slots_by_type_total = {"ADV": 0, "BAS": 0, "MOTO": 0}
    team_slots_total = 0
    teams_covered_total = 0

    for shift in range(shifts):
        is_night = (shift % 3 == 2)
        day_idx = shift // 3
        # Assigned skill counts in this shift
        assigned_counts = {skill: sum(int(solver.Value(y[(skill, i, shift)])) for i in range(workforce_count_by_skill[skill])) for skill in SKILLS}


        # for skill in SKILLS:
        #     assigned_counts[skill] = sum(int(solver.Value(y[(skill, i, shift)])) for i in range(workforce_count_by_skill[skill]))

        # Team supply for this shift (apply MOTO weekend rule)
        if is_night:
            team_supply = {
                "ADV": teams_per_night_shift.get("ADV", 0),
                "BAS": teams_per_night_shift.get("BAS", 0),
                "MOTO": 0,
            }
            alloc = _night_team_allocation_from_assigned(assigned_counts, team_supply)
        else:
            if _is_weekend_day(day_idx):
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
            alloc = _day_team_allocation_from_assigned(assigned_counts, team_supply)

        # Totals
        team_slots_total += sum(team_supply.values())
        for t in ["ADV", "BAS", "MOTO"]:
            team_slots_by_type_total[t] += team_supply.get(t, 0)

        covered = {"ADV": alloc.get("ADV", 0), "BAS": alloc.get("BAS", 0), "MOTO": alloc.get("MOTO", 0)}
        teams_covered_total += covered["ADV"] + covered["BAS"] + covered["MOTO"]
        for t in ["ADV", "BAS", "MOTO"]:
            teams_covered_by_type[t] += covered[t]

        team_coverage_by_shift.append(
            {
                "shift": shift,
                "shift_type": "night" if is_night else "day",
                "day_index": day_idx,
                "day_of_week": _dow_name(day_idx),
                "covered": covered,
                "slots": {"ADV": team_supply["ADV"], "BAS": team_supply["BAS"], "MOTO": team_supply["MOTO"]},
            }
        )

    teams_covered_fraction = (teams_covered_total / team_slots_total) if team_slots_total > 0 else 0.0

    # Summary
    summary = {
        "status": status,
        "days": days,
        "shifts_total": shifts_total,
        "demand_per_shift_summary": demand_per_shift_summary,
        "total_positions_demand": total_positions_demand,
        "workforce_count_by_skill": workforce_count_by_skill,
        "weekly_soft_overage": weekly_soft_overage,
        "rolling_weeks_for_soft": rolling_weeks_for_soft,
        "shifts_fully_covered": shifts_fully_covered,
        "coverage_fraction": coverage_fraction,
        "teams_covered_total": teams_covered_total,
        "team_slots_total": team_slots_total,
        "teams_covered_fraction": teams_covered_fraction,
        "teams_covered_by_type": teams_covered_by_type,
        "team_slots_by_type_total": team_slots_by_type_total,
        "filled_positions_total": filled_positions_total,
        "filled_positions_fraction": filled_positions_fraction,
        "workers_used_by_skill": workers_used_by_skill,
        "hours_stats_by_skill": hours_stats_by_skill,
        "c_by_shift": c_by_shift,
        "team_coverage_by_shift": team_coverage_by_shift,
    }

    if include_assignments and status in ("OPTIMAL", "FEASIBLE"):
        # Detailed assignment dump
        by_shift = []
        for shift in range(shifts):
            skill_to_workers = {}
            for skill in SKILLS:
                assigned = []
                for i in range(workforce_count_by_skill[skill]):
                    if solver.Value(y[(skill, i, shift)]) > 0.5:
                        assigned.append(i)
                skill_to_workers[skill] = assigned
            by_shift.append({
                "shift": shift,
                "hours": shift_hours[shift],
                "fully_covered": int(solver.Value(c[shift])),
                "teams_covered": team_coverage_by_shift[shift]["covered"],
                "team_slots": team_coverage_by_shift[shift]["slots"],
                "skill_to_workers": skill_to_workers
            })
        summary["assignments"] = {
            "by_shift": by_shift,
        }

    return summary


def _print_summary(summary: Dict[str, Any]) -> None:
    """Print concise summary with granular team coverage and MOTO weekend rule."""
    print("Status:", summary["status"])
    print(f"Horizon: {summary['days']} days, shifts total: {summary['shifts_total']}")
    print("Demand per shift type (skills) with MOTO weekend rule:")
    for stype, dem in summary["demand_per_shift_summary"].items():
        print(f"  {stype}: {dem}")
    print("Workforce counts by skill:")
    for skill, n in summary["workforce_count_by_skill"].items():
        print(f"  {skill}: {n}")
    print(f"Weekly soft overage: {int(round(100*summary['weekly_soft_overage']))}%")
    print(f"Rolling weeks for soft cap: {summary['rolling_weeks_for_soft']}")
    print(f"Shifts fully covered: {summary['shifts_fully_covered']} / {summary['shifts_total']} "
          f"({summary['coverage_fraction']:.1%})")
    print(f"Team slots covered: {summary['teams_covered_total']} / {summary['team_slots_total']} "
          f"({summary['teams_covered_fraction']:.1%})")
    print("Teams covered by type (total across horizon) vs slots:")
    for t in ["ADV", "BAS", "MOTO"]:
        covered = summary["teams_covered_by_type"].get(t, 0)
        slots = summary["team_slots_by_type_total"].get(t, 0)
        pct = (covered / slots) if slots > 0 else 0.0
        print(f"  {t}: {covered} / {slots} ({pct:.1%})")
    print(f"Filled positions: {summary['filled_positions_total']} / {summary['total_positions_demand']} "
          f"({summary['filled_positions_fraction']:.1%})")
    print("Workers used by skill:")
    for skill, n in summary["workers_used_by_skill"].items():
        print(f"  {skill}: {n}")
    print("Hours stats among used workers by skill (min, max, spread):")
    for skill, st in summary["hours_stats_by_skill"].items():
        print(f"  {skill}: {st}")


if __name__ == "__main__":
    # Build a small workforce (weekly caps in hours)
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

    result = solve_max_covered_shifts(
        days=15,
        worker_list=demo_workers,
        include_assignments=False,
        time_limit=300.0,
        num_search_workers=12,
    )
    _print_summary(result)
    # pprint.pprint(result)
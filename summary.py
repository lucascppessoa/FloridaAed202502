"""Summary and reporting functions for EMS staffing optimization results."""

from typing import Dict, Any, List
import json
import csv
from helpers import (
    is_night_shift,
    is_weekend_day,
    dow_name,
    day_team_allocation_from_assigned,
    night_team_allocation_from_assigned,
    compute_shift_hours,
    aligned_week_windows,
)

SKILLS = ["MD", "N", "D"]
TEAM_REQUIREMENTS: Dict[str, Dict[str, int]] = {
    "ADV": {"MD": 1, "N": 1, "D": 1},
    "BAS": {"N": 1, "D": 1},
    "MOTO": {"N": 2},
}

def generate_summary(solver,
                     workforce_count_by_skill,
                     total_positions_demand,
                     teams_per_night_shift,
                     teams_per_day_shift,
                     days,
                     weekly_soft_overage,
                     rolling_weeks_for_soft,
                     full_coverage,
                     workers_assigned,
                     include_assignments) -> Dict[str, Any]:
    """
    Generate a summary of the optimization results.

    Args:
      solver: the solver object.
      model: the model object.

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
        - workers_used_by_skill
        - c_by_shift (list)
        - team_coverage_by_shift (list of dicts per shift with covered and slots)
        - assignments (optional detailed fields)
    """
    status = solver.StatusName()
    shifts = days * 3

        # A summary of day/night demands considering the weekend rule
    # Day (weekday): team mix as provided
    day_weekday_demand = {skill: 0 for skill in SKILLS}
    for team, cnt in teams_per_day_shift.items():
        for skill, req in TEAM_REQUIREMENTS[team].items():
            day_weekday_demand[skill] += cnt * req
    # Day (weekend): MOTO=0
    day_weekend_demand = {skill: 0 for skill in SKILLS}
    for team, cnt in {"ADV": teams_per_day_shift.get("ADV", 0),
                      "BAS": teams_per_day_shift.get("BAS", 0),
                      "MOTO": 0}.items():
        for skill, req in TEAM_REQUIREMENTS[team].items():
            day_weekend_demand[skill] += cnt * req
    # Night (no MOTO)
    night_demand = {skill: 0 for skill in SKILLS}
    for team, cnt in teams_per_night_shift.items():
        for skill, req in TEAM_REQUIREMENTS[team].items():
            night_demand[skill] += cnt * req
    demand_per_shift_summary = {
        "day_weekday": day_weekday_demand,
        "day_weekend": day_weekend_demand,
        "night": night_demand,
    }



    # Extract results: fully covered shifts
    shifts_fully_covered = 0
    c_by_shift = []
    for shift in range(shifts):
        val = int(round(solver.Value(full_coverage[shift])))
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
                    int(solver.Value(workers_assigned[(skill, i, shift)]))
                    for i in range(workforce_count_by_skill[skill])
                )
    filled_positions_fraction = (filled_positions_total / total_positions_demand) if total_positions_demand > 0 else 0.0

    # Used workers and hours stats
    workers_used_by_skill: Dict[str, int] = {}

    # Granular team coverage accounting (post-solve from assigned skills)
    team_coverage_by_shift: List[Dict[str, Any]] = []
    teams_covered_by_type = {"ADV": 0, "BAS": 0, "MOTO": 0}
    team_slots_by_type_total = {"ADV": 0, "BAS": 0, "MOTO": 0}
    team_slots_total = 0
    teams_covered_total = 0

    for shift in range(shifts):
        is_night = is_night_shift(shift)
        day_idx = shift // 3
        # Assigned skill counts in this shift
        assigned_counts = {skill: sum(int(solver.Value(workers_assigned[(skill, i, shift)])) for i in range(workforce_count_by_skill[skill])) for skill in SKILLS}

        # Team supply for this shift (apply MOTO weekend rule)
        if is_night:
            team_supply = {
                "ADV": teams_per_night_shift.get("ADV", 0),
                "BAS": teams_per_night_shift.get("BAS", 0),
                "MOTO": 0,
            }
            alloc = night_team_allocation_from_assigned(assigned_counts, team_supply)
        else:
            if is_weekend_day(shift):
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
            alloc = day_team_allocation_from_assigned(assigned_counts, team_supply)

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
                "day_of_week": dow_name(day_idx),
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
        "c_by_shift": c_by_shift,
        "team_coverage_by_shift": team_coverage_by_shift,
    }

    if include_assignments and status in ("OPTIMAL", "FEASIBLE"):
        shift_hours = compute_shift_hours(days)
        # Detailed assignment dump
        by_shift = []
        for shift in range(shifts):
            skill_to_workers = {}
            for skill in SKILLS:
                assigned = []
                for i in range(workforce_count_by_skill[skill]):
                    if solver.Value(workers_assigned[(skill, i, shift)]) > 0.5:
                        assigned.append(i)
                skill_to_workers[skill] = assigned
            by_shift.append({
                "shift": shift,
                "hours": shift_hours[shift],
                "fully_covered": int(solver.Value(full_coverage[shift])),
                "teams_covered": team_coverage_by_shift[shift]["covered"],
                "team_slots": team_coverage_by_shift[shift]["slots"],
                "skill_to_workers": skill_to_workers
            })
        summary["assignments"] = {
            "by_shift": by_shift,
        }

    return summary


def generate_team_csv_files(summary: Dict[str, Any], days: int = 30) -> None:
    """
    Generate CSV files for each team type (ADV, BAS, MOTO) showing the number
    of teams formed per shift (morning, afternoon, night) per day.
    
    Args:
        summary: The summary dictionary containing team_coverage_by_shift data.
        days: The number of days in the planning horizon (default 30).
    """
    # Initialize data structure: team_type -> shift_type -> day -> count
    team_data = {
        "ADV": {"morning": [0] * days, "afternoon": [0] * days, "night": [0] * days},
        "BAS": {"morning": [0] * days, "afternoon": [0] * days, "night": [0] * days},
        "MOTO": {"morning": [0] * days, "afternoon": [0] * days, "night": [0] * days},
    }
    
    # Parse the team coverage data
    for shift_info in summary["team_coverage_by_shift"]:
        shift_num = shift_info["shift"]
        day_idx = shift_info["day_index"]
        covered = shift_info["covered"]
        
        # Determine shift type within the day (0=morning, 1=afternoon, 2=night)
        shift_in_day = shift_num % 3
        if shift_in_day == 0:
            shift_type = "morning"
        elif shift_in_day == 1:
            shift_type = "afternoon"
        else:  # shift_in_day == 2
            shift_type = "night"
        
        # Record the number of teams covered for each team type
        for team_type in ["ADV", "BAS", "MOTO"]:
            team_data[team_type][shift_type][day_idx] = covered.get(team_type, 0)
    
    # Generate CSV files for each team type
    for team_type in ["ADV", "BAS", "MOTO"]:
        filename = f"{team_type.lower()}.csv"
        with open(filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            
            # Write header: "Shift,1,2,3,...,30"
            header = ["Shift"] + list(range(1, days + 1))
            writer.writerow(header)
            
            # Write rows for each shift type
            for shift_type in ["morning", "afternoon", "night"]:
                row = [shift_type] + team_data[team_type][shift_type]
                writer.writerow(row)
        
        print(f"Generated {filename}")


def validate_worker_constraints(
    summary: Dict[str, Any],
    worker_list: List[tuple],
    weekly_soft_overage: int = 2,
    rolling_weeks_for_soft: int = 4
) -> Dict[str, Any]:
    """
    Validate that all worker constraints are satisfied in the solution.
    
    Args:
        summary: The summary dictionary containing assignment data
        worker_list: List of tuples (skill, weekly_hour_cap) for each worker
        weekly_soft_overage: Per-week overage allowed (e.g., 2 allows 42 when cap=40)
        rolling_weeks_for_soft: Rolling aligned K-week window cap
    
    Returns:
        Dictionary with validation results:
            - valid: bool, True if all constraints satisfied
            - violations: list of violation descriptions
            - stats: dictionary with violation counts by type
    """
    violations = []
    stats = {
        "consecutive_shifts": 0,
        "weekly_overage": 0,
        "rolling_weeks_overage": 0,
    }
    
    if "assignments" not in summary or summary["status"] not in ("OPTIMAL", "FEASIBLE"):
        return {
            "valid": False,
            "violations": ["No valid assignments to validate"],
            "stats": stats,
        }
    
    days = summary["days"]
    shifts = summary["shifts_total"]
    shift_hours = compute_shift_hours(days)
    windows = [(w[0], w[1]) for w in aligned_week_windows(shifts)]
    
    # Build worker data structure: skill -> worker_id -> worker_info
    workers_by_skill: Dict[str, Dict[int, Dict[str, Any]]] = {skill: {} for skill in SKILLS}
    worker_idx_by_skill = {skill: 0 for skill in SKILLS}
    
    for skill, cap in worker_list:
        worker_id = worker_idx_by_skill[skill]
        workers_by_skill[skill][worker_id] = {
            "skill": skill,
            "weekly_cap": cap,
            "weekly_soft_cap": cap + weekly_soft_overage,
            "rolling_cap": cap * rolling_weeks_for_soft,
            "shifts": [],  # List of shift indices this worker is assigned to
        }
        worker_idx_by_skill[skill] += 1
    
    # Extract assignments from summary
    for shift_data in summary["assignments"]["by_shift"]:
        shift = shift_data["shift"]
        for skill, worker_ids in shift_data["skill_to_workers"].items():
            for worker_id in worker_ids:
                if worker_id in workers_by_skill[skill]:
                    workers_by_skill[skill][worker_id]["shifts"].append(shift)
    
    # Validate constraints for each worker
    for skill in SKILLS:
        for worker_id, worker in workers_by_skill[skill].items():
            worker_name = f"{skill}-{worker_id}"
            assigned_shifts = sorted(worker["shifts"])
            
            if not assigned_shifts:
                continue  # Worker not assigned to any shift
            
            # 1. Check no 3 consecutive shifts
            for i in range(len(assigned_shifts) - 2):
                if (assigned_shifts[i+1] == assigned_shifts[i] + 1 and 
                    assigned_shifts[i+2] == assigned_shifts[i] + 2):
                    violations.append(
                        f"Worker {worker_name} has 3 consecutive shifts: "
                        f"{assigned_shifts[i]}, {assigned_shifts[i+1]}, {assigned_shifts[i+2]}"
                    )
                    stats["consecutive_shifts"] += 1
            
            # 2. Check weekly soft cap for each aligned week window
            for week_idx, (start, end) in enumerate(windows):
                week_shifts = [s for s in assigned_shifts if start <= s < end]
                week_hours = sum(shift_hours[s] for s in week_shifts)
                
                if week_hours > worker["weekly_soft_cap"]:
                    violations.append(
                        f"Worker {worker_name} exceeds weekly soft cap in week {week_idx} "
                        f"(shifts {start}-{end-1}): {week_hours}h > {worker['weekly_soft_cap']}h"
                    )
                    stats["weekly_overage"] += 1
            
            # 3. Check rolling K-week cap
            if len(windows) >= rolling_weeks_for_soft:
                for w in range(len(windows) - rolling_weeks_for_soft + 1):
                    start = windows[w][0]
                    end = windows[w + rolling_weeks_for_soft - 1][1]
                    rolling_shifts = [s for s in assigned_shifts if start <= s < end]
                    rolling_hours = sum(shift_hours[s] for s in rolling_shifts)
                    
                    if rolling_hours > worker["rolling_cap"]:
                        violations.append(
                            f"Worker {worker_name} exceeds {rolling_weeks_for_soft}-week rolling cap "
                            f"(weeks {w}-{w+rolling_weeks_for_soft-1}, shifts {start}-{end-1}): "
                            f"{rolling_hours}h > {worker['rolling_cap']}h"
                        )
                        stats["rolling_weeks_overage"] += 1
    
    return {
        "valid": len(violations) == 0,
        "violations": violations,
        "stats": stats,
    }


def print_validation_results(validation: Dict[str, Any]) -> None:
    """
    Print validation results in a readable format.
    
    Args:
        validation: The validation results dictionary from validate_worker_constraints
    """
    print("\n" + "="*80)
    print("CONSTRAINT VALIDATION RESULTS")
    print("="*80)
    
    if validation["valid"]:
        print("✓ All constraints satisfied! The solution is valid.")
    else:
        print("✗ Constraint violations detected!")
        print(f"\nTotal violations: {len(validation['violations'])}")
        print(f"  - Consecutive shift violations: {validation['stats']['consecutive_shifts']}")
        print(f"  - Weekly overage violations: {validation['stats']['weekly_overage']}")
        print(f"  - Rolling {4}-week overage violations: {validation['stats']['rolling_weeks_overage']}")
        
        if validation["violations"]:
            print("\nDetailed violations:")
            print("-" * 80)
            for i, violation in enumerate(validation["violations"], 1):
                print(f"{i}. {violation}")
    
    print("="*80 + "\n")


def print_summary(summary) -> None:
    """Print concise summary with granular team coverage and MOTO weekend rule."""
    print("Status:", summary["status"])
    print(f"Horizon: {summary['days']} days, shifts total: {summary['shifts_total']}")
    print("Demand per shift type (skills) with MOTO weekend rule:")
    for stype, dem in summary["demand_per_shift_summary"].items():
        print(f"  {stype}: {dem}")
    print("Workforce counts by skill:")
    for skill, n in summary["workforce_count_by_skill"].items():
        print(f"  {skill}: {n}")
    print(f"Weekly soft overage: {summary['weekly_soft_overage']}")
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


    print("Assignments:")
    abcd = {}
    for shift in summary["assignments"]["by_shift"]:
        abcd[shift['shift']] = {'workers': shift['skill_to_workers'], 'teams': shift['teams_covered']}
    
    # Write abcd to a json file
    with open('out.json', 'w') as f:
        json.dump(abcd, f)


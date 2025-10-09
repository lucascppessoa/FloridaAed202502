"""Helper functions for EMS staffing optimization."""

from typing import Dict, List, Tuple


def compute_shift_hours(days: int) -> List[int]:
    """List of shift durations in hours for the horizon. Pattern per day: [6, 6, 12]."""
    return [6, 6, 12] * days


def aligned_week_windows(shifts: int) -> List[Tuple[int, int]]:
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


def effective_weekly_cap(weekly_hours_cap: int) -> int:
    """Realizable weekly cap constrained by 6h/12h shifts: floor to nearest multiple of 6."""
    return (weekly_hours_cap // 6) * 6


def is_weekend_day(shift_index: int) -> bool:
    dow = (shift_index // 3) % 7
    return dow in (5, 6)

def is_night_shift(shift_index: int) -> bool:
    return shift_index % 3 == 2


def dow_name(day_index: int) -> str:
    names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    return names[day_index % 7]


def build_shift_skill_demands(
    days: int,
    teams_per_day_shift: Dict[str, int],
    teams_per_night_shift: Dict[str, int],
    skills: List[str],
    team_requirements: Dict[str, Dict[str, int]],
) -> List[Dict[str, int]]:
    """Build per-shift skill demands:
        No MOTO on nights and weekends
    """
    shifts = days * 3
    demand_by_shift: List[Dict[str, int]] = []
    for shift_index in range(shifts):
        # Construct team supply for this shift
        if is_night_shift(shift_index) or is_weekend_day(shift_index):
            team_supply = teams_per_night_shift
        else:
            team_supply = teams_per_day_shift

        # Convert team supply to skill demand
        demand = {skill: 0 for skill in skills}
        for team, count in team_supply.items():
            if count <= 0 or team not in team_requirements:
                continue
            for skill, req in team_requirements[team].items():
                demand[skill] += count * req
        demand_by_shift.append(demand)
    return demand_by_shift


def day_team_allocation_from_assigned(
    assigned: Dict[str, int], teams_per_day_shift: Dict[str, int]
) -> Dict[str, int]:
    """Given assigned skill counts for a day shift, compute max numbers of ADV, BAS, MOTO teams
    that can be formed (integer) without exceeding:
      - assigned skill counts
      - offered team slots from teams_per_day_shift

    Constraints (day):
      x_ADV <= MD
      x_ADV + x_BAS + 2*x_MOTO <= N  (ADV uses 1N, BAS uses 1N, MOTO uses 2N)
      x_ADV + x_BAS <= D
      0 <= x_t <= team slots for t
    We enumerate on (ADV, BAS) and pick max MOTO greedily.

    Returns: dict {"ADV": int, "BAS": int, "MOTO": int}
    """
    md = assigned.get("MD", 0)
    n = assigned.get("N", 0)
    d = assigned.get("D", 0)
    adv_supply = int(teams_per_day_shift.get("ADV", 0))
    bas_supply = int(teams_per_day_shift.get("BAS", 0))
    moto_supply = int(teams_per_day_shift.get("MOTO", 0))

    best = {"ADV": 0, "BAS": 0, "MOTO": 0}
    best_total = -1

    adv_max = min(adv_supply, md, n, d)
    for adv in range(adv_max + 1):
        d_rem = d - adv
        n_rem = n - adv
        if d_rem < 0 or n_rem < 0:
            continue
        bas_max = min(bas_supply, d_rem, n_rem)
        for bas in range(bas_max + 1):
            n_rem_after_bas = n_rem - bas
            if n_rem_after_bas < 0:
                continue
            # MOTO needs 2N each
            moto_max = min(moto_supply, n_rem_after_bas // 2)
            if moto_max < 0:
                moto_max = 0
            total = adv + bas + moto_max
            if total > best_total:
                best_total = total
                best = {"ADV": adv, "BAS": bas, "MOTO": moto_max}
    return best


def night_team_allocation_from_assigned(
    assigned: Dict[str, int], teams_per_night_shift: Dict[str, int]
) -> Dict[str, int]:
    """Given assigned skill counts for a night shift, compute max numbers of ADV, BAS teams
    that can be formed (integer) without exceeding:
      - assigned skill counts
      - offered team slots from teams_per_night_shift

    Constraints (night):
      x_ADV <= MD
      x_ADV + x_BAS <= N  (both ADV and BAS use 1N each)
      x_ADV + x_BAS <= D
      0 <= x_t <= team slots for t
    We enumerate on ADV and pick max BAS greedily.

    Returns: dict {"ADV": int, "BAS": int, "MOTO": 0}
    """
    md = assigned.get("MD", 0)
    n = assigned.get("N", 0)
    d = assigned.get("D", 0)
    adv_supply = int(teams_per_night_shift.get("ADV", 0))
    bas_supply = int(teams_per_night_shift.get("BAS", 0))

    best_adv = 0
    best_bas = 0
    best_total = -1

    adv_max = min(adv_supply, md, n, d)
    for adv in range(adv_max + 1):
        d_rem = d - adv
        n_rem = n - adv
        if d_rem < 0 or n_rem < 0:
            continue
        bas_max = min(bas_supply, d_rem, n_rem)
        total = adv + bas_max
        if total > best_total:
            best_total = total
            best_adv = adv
            best_bas = bas_max

    return {"ADV": best_adv, "BAS": best_bas, "MOTO": 0}


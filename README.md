# EMS Staffing Optimization

A Constraint Programming solution for optimizing Emergency Medical Services (EMS) staffing over a multi-week planning horizon. This system maximizes team coverage while respecting worker constraints, labor rules, and operational requirements.

## Overview

The system uses Google OR-Tools CP-SAT solver to schedule workers across shifts while:
- Forming emergency response teams (Advanced, Basic, Motorcycle)
- Enforcing labor laws (weekly hours, rest periods, rolling overtime limits)
- Respecting operational rules (MOTO teams only on weekday days)
- Prioritizing high-value teams through configurable weights

## Features

- ✅ **Explicit Team Formation Modeling**: Teams are decision variables with proper skill allocation
- ✅ **Worker Constraint Validation**: No 3 consecutive shifts, weekly hour limits, rolling 4-week caps
- ✅ **Flexible Team Weights**: Prioritize more valuable teams (e.g., ADV teams worth 2-8x others)
- ✅ **Balance Constraints**: Optional limits on team count variation across shifts
- ✅ **MOTO Weekend Rule**: Motorcycle teams only operate on weekday day shifts
- ✅ **Solution Validation**: Automatic checking of all constraints after optimization

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup

1. Clone or download this repository

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install ortools pandas numpy
```

## Quick Start

Run the optimization with default settings:
```bash
python main.py
```

This will:
1. Solve the staffing problem (5 minutes with default settings)
2. Print a summary of results
3. Generate CSV files for each team type (adv.csv, bas.csv, moto.csv)
4. Validate all worker constraints
5. Save detailed assignments to out.json

## Configuration

### Input Parameters

Edit `main.py` to configure the optimization:

#### **Planning Horizon**
```python
days = 30  # Number of days to plan (default: 30)
```
- Creates 3 shifts per day (morning, afternoon, night)
- Total shifts = days × 3

#### **Team Requirements**
```python
teams_per_day_shift = {"ADV": 7, "BAS": 30, "MOTO": 11}
teams_per_night_shift = {"ADV": 7, "BAS": 30}
```
- **ADV (Advanced)**: Requires 1 MD + 1 Nurse + 1 Driver
- **BAS (Basic)**: Requires 1 Nurse + 1 Driver  
- **MOTO (Motorcycle)**: Requires 2 Nurses
- MOTO automatically set to 0 for night shifts and weekends

#### **Worker Pool**
```python
demo_workers: List[Tuple[str, int]] = []
demo_workers += [("N", 40)] * 100   # 100 nurses with 40hr/week cap
demo_workers += [("MD", 40)] * 20   # 20 doctors with 40hr/week cap
demo_workers += [("D", 40)] * 111   # 111 drivers with 40hr/week cap
```
- First element: Skill type ("N"=Nurse, "MD"=Doctor, "D"=Driver)
- Second element: Personal weekly hour cap

#### **Weekly Hour Flexibility**
```python
weekly_soft_overage = 2        # Hours over cap allowed per week
rolling_weeks_for_soft = 4     # Rolling window for overtime compensation
```
- **Example**: 40hr/week cap + 2hr overage = 42hr max per week
- Over 4 weeks: Cannot exceed 4 × 40 = 160 total hours
- Allows temporary overtime but requires compensation

#### **Team Weights** (Objective Function)
```python
team_weights = {"ADV": 8, "BAS": 2, "MOTO": 1}
```
- Controls priority in optimization
- Higher weight = higher priority
- **None** or **{}** = all teams equal weight (1)
- Optimizer will prefer forming 1 ADV team over 8 MOTO teams

#### **Balance Constraints** (Optional)
```python
team_imbalance = {"ADV": 3, "BAS": 10}
```
- Limits variation in team counts across shifts
- **Example**: ADV teams must be within 3 of each other across all shifts
- **None** = no balance constraints
- Typically don't constrain MOTO due to weekend/night restrictions

#### **Solver Settings**
```python
time_limit = 300.0        # Maximum solve time in seconds
num_search_workers = 12   # Parallel search threads
```

## Output Files

### 1. Console Output

#### Summary Statistics
```
Status: OPTIMAL
Horizon: 30 days, shifts total: 90
Demand per shift type (skills) with MOTO weekend rule:
  day_weekday: {'MD': 7, 'N': 99, 'D': 48}
  day_weekend: {'MD': 7, 'N': 37, 'D': 37}
  night: {'MD': 7, 'N': 37, 'D': 37}
Workforce counts by skill:
  MD: 20
  N: 257
  D: 111
Weekly soft overage: 2
Rolling weeks for soft cap: 4
Shifts fully covered: 87 / 90 (96.7%)
Team slots covered: 3450 / 3540 (97.5%)
Teams covered by type (total across horizon) vs slots:
  ADV: 615 / 630 (97.6%)
  BAS: 2610 / 2700 (96.7%)
  MOTO: 225 / 210 (107.1%)
```

#### Constraint Validation
```
================================================================================
CONSTRAINT VALIDATION RESULTS
================================================================================
✓ All constraints satisfied! The solution is valid.
================================================================================
```

Or if violations are found:
```
✗ Constraint violations detected!

Total violations: 3
  - Consecutive shift violations: 1
  - Weekly overage violations: 2
  - Rolling 4-week overage violations: 0

Detailed violations:
--------------------------------------------------------------------------------
1. Worker N-45 has 3 consecutive shifts: 12, 13, 14
2. Worker N-78 exceeds weekly soft cap in week 2 (shifts 42-62): 48h > 42h
...
```

### 2. Team CSV Files

Three CSV files are generated showing teams formed per shift:

**adv.csv**, **bas.csv**, **moto.csv**
```csv
Shift,1,2,3,4,5,...,30
morning,7,7,7,6,7,...,7
afternoon,7,7,6,7,7,...,7
night,7,7,7,7,7,...,6
```
- Rows: Shift types (morning, afternoon, night)
- Columns: Days (1-30)
- Values: Number of teams formed

**Use cases**:
- Visualize coverage patterns
- Identify low-coverage days
- Track team availability over time
- Import into Excel/Google Sheets for reporting

### 3. Detailed Assignments (out.json)

JSON file with complete worker assignments:

```json
{
  "0": {
    "workers": {
      "MD": [2, 5],
      "N": [1, 3, 8, 15, 22, ...],
      "D": [0, 4, 7, 12, ...]
    },
    "teams": {
      "ADV": 7,
      "BAS": 30,
      "MOTO": 11
    }
  },
  "1": { ... },
  ...
}
```

- **Key**: Shift index (0-89)
- **workers**: Lists worker IDs assigned to each skill
  - Example: `"N": [1, 3, 8]` means Nurse #1, #3, and #8 work this shift
- **teams**: Number of teams formed by type

**Use cases**:
- Generate individual worker schedules
- Track worker utilization
- Verify constraint compliance
- Integration with other systems

## Understanding the Algorithm

### Decision Variables

1. **Worker Assignments** (`workers_assigned[skill, worker_id, shift]`): Boolean
   - Whether a specific worker is assigned to a specific shift
   - ~27,000 boolean variables for 300 workers × 90 shifts

2. **Team Formation** (`teams_formed[shift, team_type]`): Integer  
   - Number of teams of each type formed per shift
   - 270 integer variables (90 shifts × 3 team types)

3. **Full Coverage** (`full_coverage[shift]`): Boolean
   - Whether a shift has all required teams present
   - 90 boolean variables

### Constraints

#### Worker Constraints
- **No 3 consecutive shifts**: At most 2 out of any 3 consecutive shifts
- **Weekly hour limits**: Personal cap + overage (e.g., 42 hours/week max)
- **Rolling 4-week cap**: 4 × personal_weekly_cap over any 4-week window

#### Team Formation Constraints
- **ADV teams**: 1 MD + 1 N + 1 D each
- **BAS teams**: 1 N + 1 D each
- **MOTO teams**: 2 N each
- **Resource sharing**: Skills allocated efficiently across team types
- **MOTO weekend rule**: MOTO = 0 on nights and weekends

#### Optional Constraints
- **Balance**: Max difference in team counts across shifts per team type

### Objective Function

**Maximize**: Weighted sum of teams formed

```
Σ (weight[team_type] × teams_formed[shift, team_type])
  for all shifts and team types
```

**With tie-breaker**: Among solutions with equal weighted team counts, prefer solutions with more fully covered shifts.

## Troubleshooting

### Common Issues

**1. No feasible solution found**
- Status: INFEASIBLE
- **Cause**: Not enough workers to meet minimum demands
- **Solution**: Increase worker pool or reduce team requirements

**2. Time limit reached**
- Status: FEASIBLE (not OPTIMAL)
- **Cause**: Problem too large for time limit
- **Solution**: Increase `time_limit` or reduce planning horizon

**3. Constraint violations reported**
- **Cause**: Bug in model or validation
- **Solution**: Review violation details and report as issue

**4. High memory usage**
- **Cause**: Large planning horizon × workforce
- **Solution**: Reduce `days` or workforce size, or increase system RAM

### Performance Tips

- Start with shorter horizons (15 days) to test configurations
- Use `num_search_workers` ≤ number of CPU cores
- Adjust `time_limit` based on problem size (60s for 15 days, 300s+ for 30 days)
- Balance constraints can slow solve time significantly

## Example Configurations

### Scenario 1: Maximum Coverage (No Balance)
```python
team_weights = None  # Equal priority
team_imbalance = None  # No balance constraints
time_limit = 300.0
```
Best for: Maximizing total coverage when distribution doesn't matter

### Scenario 2: Prioritize ADV Teams
```python
team_weights = {"ADV": 8, "BAS": 2, "MOTO": 1}
team_imbalance = None
time_limit = 300.0
```
Best for: When advanced teams are much more valuable

### Scenario 3: Balanced Coverage
```python
team_weights = {"ADV": 2}
team_imbalance = {"ADV": 2, "BAS": 5}
time_limit = 600.0
```
Best for: Ensuring consistent service levels across all shifts

### Scenario 4: Quick Testing
```python
days = 7  # One week only
time_limit = 60.0
num_search_workers = 4
```
Best for: Rapid iteration during configuration

## Project Structure

```
.
├── main.py           # Entry point and configuration
├── solver.py         # CP-SAT optimization model
├── summary.py        # Results reporting and validation
├── helpers.py        # Utility functions
├── README.md         # This file
├── adv.csv          # Generated: ADV team coverage
├── bas.csv          # Generated: BAS team coverage
├── moto.csv         # Generated: MOTO team coverage
└── out.json         # Generated: Detailed assignments
```

## License

This project is provided as-is for educational and operational use.

## Support

For questions, issues, or feature requests, please review the code comments and validation output for guidance.

---

**Version**: 1.0  
**Last Updated**: 2025


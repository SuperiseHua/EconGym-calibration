import csv
import json


CORE_METRICS = (
    "central_bank_gov_reward", "inflation_rate", "inflation_gap", "growth_gap",
    "GDP", "social_welfare", "income_gini", "wealth_gini",
    "base_interest_rate", "reserve_ratio",
)


def _value(value):
    """Turn scalar-shaped arrays/lists into one readable value."""
    if hasattr(value, "tolist"):
        value = value.tolist()
    while isinstance(value, list) and len(value) == 1:
        value = value[0]
    return value


def _format(value):
    value = _value(value)
    if isinstance(value, (int, float)):
        return f"{value:,.6g}"
    return json.dumps(value, ensure_ascii=False)


def _table(headers, rows):
    rows = [[str(cell) for cell in row] for row in rows]
    widths = [max(len(str(h)), *(len(row[i]) for row in rows)) for i, h in enumerate(headers)]
    line = lambda row: "  ".join(str(cell).ljust(widths[i]) for i, cell in enumerate(row))
    return "\n".join((line(headers), line(["-" * width for width in widths]), *(line(row) for row in rows)))


def _selected_periods(trajectory, max_rows=40):
    """Keep short runs intact and sample long runs without hiding early dynamics."""
    if len(trajectory) <= max_rows:
        return trajectory
    selected = set(range(min(10, len(trajectory))))
    selected.add(len(trajectory) - 1)
    selected.update(range(19, len(trajectory), 20))
    return [trajectory[index] for index in sorted(selected)]


def _print_trajectory(trajectory):
    print("\nEconomic trajectory (episode 1; GDP and price are base-period indices)")
    shown = _selected_periods(trajectory)
    previous_period = 0
    for row in shown:
        if row["period"] > previous_period + 1:
            print(f"... periods {previous_period + 1}-{row['period'] - 1} omitted ...")
        warning = " [WARN: I=0]" if row["investment_gdp"] <= 1e-10 else ""
        print(
            f"t={row['period']:02d} | Real GDP={row['real_gdp_index']:6.1f} | "
            f"Growth={row['gdp_growth']:+6.1%} | Price={row['price_index']:6.1f} | "
            f"Inflation={row['inflation_rate']:+6.1%} | C/Y={row['consumption_gdp']:5.1%} | "
            f"I/Y={row['investment_gdp']:5.1%}{warning} | G/Y={row['government_gdp']:5.1%}"
        )
        previous_period = row["period"]


def _print_summary(mean):
    print("\nRun summary (mean across evaluation episodes)")
    rows = [
        ["Real GDP", f"100.0 -> {mean['GDP_index_end']:.1f}",
         f"last growth {mean['GDP_growth_end']:+.1%}",
         f"volatility {mean['GDP_growth_volatility']:.1%}"],
        ["Prices", f"100.0 -> {mean['price_index_end']:.1f}",
         f"mean inflation {mean['inflation_mean']:+.1%}", ""],
        ["Expenditure", f"C/Y {mean['consumption_gdp_mean']:.1%}",
         f"I/Y {mean['investment_gdp_mean']:.1%}",
         f"G/Y {mean['government_gdp_mean']:.1%}"],
        ["Investment stability", f"I=0 {mean['zero_investment_share']:.1%}",
         f"zero/nonzero switches {mean['investment_switch_share']:.1%}",
         f"nonpositive user cost {mean.get('nonpositive_user_cost_share', 0):.1%}"],
        ["Binding constraints", f"credit {mean['credit_binding_share']:.1%}",
         f"goods {mean['goods_binding_share']:.1%}",
         f"unused output {mean['unused_output_gdp_mean']:.1%}"],
    ]
    print(_table(["block", "level", "change/stability", "structure"], rows))


def _print_investment_diagnostics(trajectory):
    if not any(row["investment_gdp"] <= 1e-10 for row in trajectory):
        return
    rows = []
    for row in trajectory[:12]:
        bindings = []
        if row.get("nonpositive_user_cost", False):
            bindings.append("nonpositive user cost")
        if row["credit_binding"]:
            bindings.append("credit")
        if row["goods_binding"]:
            bindings.append("goods")
        rows.append([
            row["period"], f"{row['expected_inflation_used']:+.1%}",
            f"{row['real_lending_rate']:+.1%}", f"{row['capital_user_cost']:.1%}",
            f"{row['capital_target_ratio']:.3g}", f"{row['desired_investment_gdp']:.1%}",
            f"{row['financed_investment_gdp']:.1%}", f"{row['investment_gdp']:.1%}",
            ", ".join(bindings) or "none",
        ])
    print("\nInvestment diagnostics (first 12 periods)")
    print(_table(
        ["t", "pi used", "real r", "user cost", "K*/K", "Id/Y", "If/Y", "I/Y", "binding"],
        rows,
    ))


def report_evaluation(results, comparison_path):
    """Print scale-free diagnostics and maintain one compact cross-run CSV."""
    episodes, mean = results["episodes"], results["aggregate"]
    trajectories = results.get("trajectories", [])
    for episode in episodes:
        if episode.get("termination_reason"):
            print(f"Simulation failed: episode {episode['episode']}, "
                  f"reason={episode['termination_reason']}")
    if trajectories:
        _print_trajectory(trajectories[0])
        _print_summary(mean)
        _print_investment_diagnostics(trajectories[0])

    metadata = results["experiment"]
    row = {**metadata, **{key: _value(value) for key, value in mean.items()}}
    exists = comparison_path.exists()
    old_rows = []
    if exists:
        with open(comparison_path, newline="", encoding="utf-8") as file:
            old_rows = list(csv.DictReader(file))
        exists = list(old_rows[0]) == list(row) if old_rows else False
    with open(comparison_path, "a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=row)
        if not exists:
            file.seek(0)
            file.truncate()
            writer.writeheader()
            writer.writerows({key: old.get(key, "") for key in row} for old in old_rows)
        writer.writerow(row)

    with open(comparison_path, newline="", encoding="utf-8") as file:
        runs = list(csv.DictReader(file))
    labels = [f"{run['run']}:{run['central_bank_alg']}" for run in runs]
    comparable_metrics = [
        metric for metric in CORE_METRICS
        if all(run.get(metric, "") not in {"", None} for run in runs)
    ]
    if comparable_metrics:
        comparison = [
            [metric, *(_format(float(run[metric])) for run in runs)]
            for metric in comparable_metrics
        ]
        print("\nBaseline comparison (one column per run)")
        print(_table(["metric", *labels], comparison))

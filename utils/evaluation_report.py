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


def report_evaluation(results, comparison_path):
    """Print every metric and maintain one compact cross-run CSV."""
    episodes, mean = results["episodes"], results["aggregate"]
    columns = [(f"episode_{ep['episode']}", ep) for ep in episodes] + [("mean", mean)]
    rows = [[metric, *(_format(values.get(metric, "-")) for _, values in columns)] for metric in mean]
    print("\nEvaluation by episode (all economic variables)")
    print(_table(["metric", *(name for name, _ in columns)], rows))

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
    comparison = [[metric, *(_format(float(run[metric])) for run in runs)] for metric in CORE_METRICS]
    print("\nBaseline comparison (one column per run)")
    print(_table(["metric", *labels], comparison))

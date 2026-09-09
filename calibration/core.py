"""Backend-independent bounded search. No economics or simulator imports.

New implementation of the existing block search idea; no historical executor
is changed. Caller-supplied targets never auto-read evaluation outcomes.
"""
from __future__ import annotations
from dataclasses import asdict, dataclass
import math
import random
from typing import Protocol


class CalibrationError(ValueError):
    pass


@dataclass(frozen=True)
class Parameter:
    name: str
    lower: float
    upper: float
    initial: float
    moment: str
    role: str = "calibrated"

    def validate(self):
        if self.role != "calibrated":
            raise CalibrationError(f"{self.name}: {self.role} must not be optimized")
        if not self.name or not self.moment:
            raise CalibrationError("parameter name and identifying moment required")
        if not all(math.isfinite(x) for x in (self.lower, self.upper, self.initial)):
            raise CalibrationError("non-finite parameter domain")
        if not self.lower < self.upper or not self.lower <= self.initial <= self.upper:
            raise CalibrationError(f"invalid domain for {self.name}")


@dataclass(frozen=True)
class Target:
    value: float
    scale: float
    tolerance: float

    def validate(self):
        if not all(math.isfinite(x) for x in (self.value, self.scale, self.tolerance)):
            raise CalibrationError("non-finite target")
        if self.scale <= 0 or self.tolerance < 0:
            raise CalibrationError("positive scale and nonnegative tolerance required")


@dataclass(frozen=True)
class Budget:
    max_configurations: int
    search_seeds: tuple[int, ...]
    validation_seeds: tuple[int, ...]

    def validate(self):
        if type(self.max_configurations) is not int or self.max_configurations < 1:
            raise CalibrationError("positive configuration budget required")
        for group in (self.search_seeds, self.validation_seeds):
            if not group or len(set(group)) != len(group):
                raise CalibrationError("seed groups must be nonempty and unique")
            if any(not isinstance(s, int) or isinstance(s, bool) or not 0 <= s < 2**32 for s in group):
                raise CalibrationError("seeds must be uint32 integers")
        if set(self.search_seeds) & set(self.validation_seeds):
            raise CalibrationError("search and validation seeds must be disjoint")


class Backend(Protocol):
    def describe(self) -> dict: ...
    def evaluate(self, parameters: dict[str, float], seeds: tuple[int, ...]) -> dict:
        """Return {moments, valid, diagnostics}."""
        ...


def score(result: dict, targets: dict[str, Target]) -> dict:
    moments = result.get("moments", {})
    if result.get("valid") is not True:
        return {"valid": False, "loss": None, "passed": False, "reason": "backend_invalid"}
    if any(name not in moments or not math.isfinite(float(moments[name])) for name in targets):
        return {"valid": False, "loss": None, "passed": False, "reason": "missing_or_nonfinite_moment"}
    errors = {name: float(moments[name]) - spec.value for name, spec in targets.items()}
    loss = math.sqrt(sum((errors[n] / t.scale)**2 for n, t in targets.items()) / len(targets))
    return {"valid": True, "loss": loss, "errors": errors,
            "passed": all(abs(errors[n]) <= t.tolerance for n, t in targets.items())}


def calibrate(backend: Backend, parameters: tuple[Parameter, ...],
              targets: dict[str, Target], budget: Budget, *, method: str = "blockwise",
              optimizer_seed: int = 902070, max_sweeps: int = 3,
              max_bisections: int = 12) -> dict:
    """Coordinate moment matching or equal-budget random search.

    Validation occurs once after selection; no tuning follows it. All supplied
    target tolerances and backend validity must pass. 'accepted' is only a
    numerical contract result, not economic validity or full identification.
    """
    budget.validate()
    if method not in ("blockwise", "random"):
        raise CalibrationError("supported methods: blockwise, random")
    if not parameters or len({p.name for p in parameters}) != len(parameters):
        raise CalibrationError("nonempty unique parameter names required")
    if not targets or max_sweeps < 1 or max_bisections < 1:
        raise CalibrationError("targets and positive search iterations required")
    for target in targets.values():
        target.validate()
    for parameter in parameters:
        parameter.validate()
        if parameter.moment not in targets:
            raise CalibrationError(f"missing target for {parameter.name}")
    theta = {p.name: p.initial for p in parameters}
    trace: list[dict] = []
    cache: dict[tuple, dict] = {}
    reasons: list[str] = []
    calls = 0

    def evaluate(candidate):
        nonlocal calls
        key = tuple(sorted(candidate.items()))
        if key in cache:
            return cache[key]
        if calls >= budget.max_configurations:
            return None
        calls += 1
        result = backend.evaluate(dict(candidate), budget.search_seeds)
        row = {"parameters": dict(candidate), "result": result, "score": score(result, targets)}
        trace.append(row)
        cache[key] = row
        return row

    evaluate(theta)
    if method == "random":
        rng = random.Random(optimizer_seed)
        while calls < budget.max_configurations:
            evaluate({p.name: rng.uniform(p.lower, p.upper) for p in parameters})
    else:
        for _ in range(max_sweeps):
            current = evaluate(theta)
            if current is not None and current["score"]["passed"]:
                break
            moved = False
            for parameter in parameters:
                lower, upper = parameter.lower, parameter.upper
                lo = evaluate({**theta, parameter.name: lower})
                hi = evaluate({**theta, parameter.name: upper})
                if lo is None or hi is None:
                    reasons.append("configuration_budget_exhausted")
                    break
                if not lo["score"]["valid"] or not hi["score"]["valid"]:
                    reasons.append(f"{parameter.name}:invalid_bracket_endpoint")
                    continue
                f_lo = lo["score"]["errors"][parameter.moment]
                f_hi = hi["score"]["errors"][parameter.moment]
                scale = targets[parameter.moment].scale
                if abs(f_hi - f_lo) <= 1e-10 * scale:
                    reasons.append(f"{parameter.name}:not_identified_flat_response")
                    continue
                if f_lo * f_hi > 0:
                    reasons.append(f"{parameter.name}:target_not_bracketed")
                    continue
                best = min((lo, hi), key=lambda r: abs(r["score"]["errors"][parameter.moment]))
                for _ in range(max_bisections):
                    if abs(best["score"]["errors"][parameter.moment]) <= targets[parameter.moment].tolerance:
                        break
                    middle = (lower + upper) / 2
                    row = evaluate({**theta, parameter.name: middle})
                    if row is None:
                        reasons.append("configuration_budget_exhausted")
                        break
                    if not row["score"]["valid"]:
                        reasons.append(f"{parameter.name}:invalid_midpoint")
                        break
                    f_mid = row["score"]["errors"][parameter.moment]
                    if not min(f_lo, f_hi) - 1e-10 * scale <= f_mid <= max(f_lo, f_hi) + 1e-10 * scale:
                        reasons.append(f"{parameter.name}:nonmonotone_or_noisy_bracket")
                        break
                    if abs(f_mid) < abs(best["score"]["errors"][parameter.moment]):
                        best = row
                    if f_lo * f_mid <= 0:
                        upper, f_hi = middle, f_mid
                    else:
                        lower, f_lo = middle, f_mid
                new_theta = dict(best["parameters"])
                moved |= new_theta != theta
                theta = new_theta
            if calls >= budget.max_configurations or not moved:
                break

    valid_rows = [r for r in trace if r["score"]["valid"]]
    feasible_rows = [r for r in valid_rows if r["score"]["passed"]]
    pool = feasible_rows or valid_rows
    candidate = min(pool, key=lambda r: r["score"]["loss"]) if pool else None
    validation = None
    if candidate is not None:
        validation_result = backend.evaluate(dict(candidate["parameters"]), budget.validation_seeds)
        validation = {"result": validation_result, "score": score(validation_result, targets)}
    accepted = bool(candidate and candidate["score"]["passed"] and validation["score"]["passed"])
    return {
        "schema_version": 1, "method": method,
        "evidence_level": "development_integration_not_confirmatory",
        "backend": backend.describe(), "parameter_specs": [asdict(p) for p in parameters],
        "targets": {n: asdict(t) for n, t in targets.items()}, "budget": asdict(budget),
        "search_configurations": calls,
        "search_seed_paths_requested": calls * len(budget.search_seeds),
        "validation_seed_paths_requested": len(budget.validation_seeds) if validation else 0,
        "diagnostic_reasons": sorted(set(reasons)), "search_trace": trace,
        "selected_before_validation": candidate["parameters"] if candidate else None,
        "search_score": candidate["score"] if candidate else None, "validation": validation,
        "status": "accepted_within_contract" if accepted else "validation_failed" if candidate else "no_valid_candidate",
        "accepted_within_contract": accepted, "economic_validity_certified": False,
        "post_validation_tuning": False,
        "identification_claim": "none; bracket diagnostics are not full identification tests",
    }

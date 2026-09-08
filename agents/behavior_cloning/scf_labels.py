"""SCF 2022 partial-consumption labels, in annual survey-year dollars.

SCF does not measure full PCE. Income is prior-calendar-year income expressed
in survey-year dollars; applying the model's 2022 tax schedule is an explicit
reference-tax approximation, not observed disposable income.
"""
import numpy as np

LABEL_VERSION = 'scf2022_annual_partial_consumption_net_income_v2'


def annual_consumption_proxy(frame):
    """Annual food plus annual rent; debt service is not consumption."""
    return (frame['FOODHOME'].to_numpy(dtype=float)
            + frame['FOODAWAY'].to_numpy(dtype=float)
            + frame['FOODDELV'].to_numpy(dtype=float)
            + 12 * frame['RENT'].to_numpy(dtype=float))


def saving_share(disposable_income, consumption, lower=-0.5, upper=1.0):
    """Unspent income share, not a bank-deposit allocation or balance.

    With nonpositive disposable income, this multiplicative action cannot
    express positive consumption from assets. Keep the existing zero action;
    report such records separately instead of dividing by an arbitrary epsilon.
    """
    income = np.asarray(disposable_income, dtype=float)
    consumption = np.asarray(consumption, dtype=float)
    if income.shape != consumption.shape:
        raise ValueError('Income and consumption must have the same shape.')
    if not np.isfinite(income).all() or not np.isfinite(consumption).all():
        raise ValueError('Income and consumption labels must be finite.')
    if np.any(consumption < 0):
        raise ValueError('Consumption labels must be non-negative.')
    result = np.zeros_like(income)
    positive = income > 0
    result[positive] = 1 - consumption[positive] / income[positive]
    return np.clip(result, lower, upper)


def build_scf_consumption_labels(frame, income_tax_function):
    consumption = annual_consumption_proxy(frame)
    gross_income = frame['INCOME'].to_numpy(dtype=float)
    disposable_income = gross_income - income_tax_function(gross_income)
    action = saving_share(disposable_income, consumption)
    positive = disposable_income > 0
    metadata = {
        'version': LABEL_VERSION,
        'coverage': 'annual food plus annual rent only; not total consumption',
        'income_basis': 'SCF prior-year gross income less model reference federal tax',
        'nonpositive_income_records': int((~positive).sum()),
        'consumption_above_action_capacity_records': int(
            (positive & (consumption > 1.5 * disposable_income)).sum()),
    }
    return action, metadata

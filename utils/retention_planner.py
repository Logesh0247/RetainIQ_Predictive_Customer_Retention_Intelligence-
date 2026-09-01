"""
Retention Planner — portfolio-level what-if campaign simulation.

The bulk run answers "who is going to churn?". This module answers the next
question a retention team actually has to sign off on: *if we run this
campaign, on this segment, what does it cost and what does it save?*

Every number here is expected value produced by the same Logistic Regression
model used for scoring: the selected retention levers are applied to the
cleaned customer records, the cohort is re-scored, and the drop in churn
probability is converted into retained revenue.
"""
import logging

import numpy as np
import pandas as pd

from src.preprocessing import build_feature_frame
from utils.prediction import load_model, calculate_risk_level

logger = logging.getLogger("retainiq")


class RetentionPlannerError(RuntimeError):
    pass


# ---------------------------------------------------------------------------
# Levers: each one edits the cleaned customer frame in place (vectorised) and
# carries a monthly cost per treated customer.
# ---------------------------------------------------------------------------

def _mask_true(df):
    return pd.Series(True, index=df.index)


def _has_internet(df):
    return df["InternetService"].astype(str).str.strip().str.lower().ne("no")


LEVERS = [
    {
        "id": "contract_1y",
        "label": "Upgrade to a one-year contract",
        "detail": "Offer a modest discount in exchange for a 12-month commitment.",
        "eligible": lambda df: df["Contract"].eq("Month-to-month"),
        "changes": {"Contract": "One year"},
        "cost_pct": 0.05,
        "cost_flat": 0.0,
        "cost_note": "5% of monthly charges (incentive)",
    },
    {
        "id": "contract_2y",
        "label": "Upgrade to a two-year contract",
        "detail": "The strongest lever in the model — needs a bigger incentive.",
        "eligible": lambda df: df["Contract"].ne("Two year"),
        "changes": {"Contract": "Two year"},
        "cost_pct": 0.10,
        "cost_flat": 0.0,
        "cost_note": "10% of monthly charges (incentive)",
    },
    {
        "id": "auto_pay",
        "label": "Move manual payers to auto-pay",
        "detail": "Electronic and mailed cheque customers churn more; auto-pay removes monthly friction.",
        "eligible": lambda df: df["PaymentMethod"].isin(["Electronic check", "Mailed check"]),
        "changes": {"PaymentMethod": "Bank transfer (automatic)"},
        "cost_pct": 0.0,
        "cost_flat": 2.0,
        "cost_note": "$2.00 per customer (sign-up credit, amortised monthly)",
    },
    {
        "id": "tech_support",
        "label": "Bundle Tech Support",
        "detail": "Give internet customers without support a bundled plan.",
        "eligible": lambda df: _has_internet(df) & df["TechSupport"].ne("Yes"),
        "changes": {"TechSupport": "Yes"},
        "cost_pct": 0.0,
        "cost_flat": 6.0,
        "cost_note": "$6.00 per customer per month (service cost)",
    },
    {
        "id": "online_security",
        "label": "Bundle Online Security",
        "detail": "Low-cost add-on that measurably increases stickiness.",
        "eligible": lambda df: _has_internet(df) & df["OnlineSecurity"].ne("Yes"),
        "changes": {"OnlineSecurity": "Yes"},
        "cost_pct": 0.0,
        "cost_flat": 4.0,
        "cost_note": "$4.00 per customer per month (service cost)",
    },
    {
        "id": "loyalty_discount",
        "label": "Apply a 10% loyalty discount",
        "detail": "Straight price reduction on the monthly bill.",
        "eligible": _mask_true,
        "changes": {"__discount__": 0.10},
        "cost_pct": 0.10,
        "cost_flat": 0.0,
        "cost_note": "10% of monthly charges (revenue given up)",
    },
]

LEVERS_BY_ID = {lever["id"]: lever for lever in LEVERS}
DEFAULT_LEVERS = ["contract_1y", "auto_pay", "tech_support"]

RISK_CHOICES = ["All", "High Risk", "Medium Risk", "Low Risk"]
CONTRACT_CHOICES = ["All", "Month-to-month", "One year", "Two year"]
TENURE_CHOICES = [
    ("all", "Any tenure"),
    ("0-12", "0–12 months"),
    ("13-24", "13–24 months"),
    ("25+", "25+ months"),
]


# ---------------------------------------------------------------------------
# Segment selection
# ---------------------------------------------------------------------------

def segment_mask(bundle, risk="All", contract="All", tenure="all", min_spend=0.0):
    results = bundle["results_df"]
    cleaned = bundle["cleaned_df"]
    mask = pd.Series(True, index=cleaned.index)

    if risk and risk != "All":
        mask &= results["Risk Level"].reset_index(drop=True).eq(risk).values
    if contract and contract != "All":
        mask &= cleaned["Contract"].eq(contract)

    tenure_values = pd.to_numeric(cleaned["tenure"], errors="coerce").fillna(0)
    if tenure == "0-12":
        mask &= tenure_values.le(12)
    elif tenure == "13-24":
        mask &= tenure_values.between(13, 24)
    elif tenure == "25+":
        mask &= tenure_values.ge(25)

    if min_spend:
        charges = pd.to_numeric(cleaned["MonthlyCharges"], errors="coerce").fillna(0)
        mask &= charges.ge(float(min_spend))

    return mask


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------

def _score(frame):
    bundle = load_model()
    if bundle is None:
        raise RetentionPlannerError(
            "The churn model is unavailable, so campaigns cannot be simulated."
        )
    features = build_feature_frame(frame)
    return np.clip(bundle["model"].predict_proba(features)[:, 1], 0, 1)


def _apply_levers(frame, lever_ids):
    """Return (treated_frame, per-customer monthly cost, treated mask)."""
    treated = frame.copy()
    charges = pd.to_numeric(frame["MonthlyCharges"], errors="coerce").fillna(0.0)
    cost = pd.Series(0.0, index=frame.index)
    touched = pd.Series(False, index=frame.index)

    for lever_id in lever_ids:
        lever = LEVERS_BY_ID.get(lever_id)
        if lever is None:
            continue
        eligible = lever["eligible"](treated).fillna(False)
        if not eligible.any():
            continue
        for column, value in lever["changes"].items():
            if column == "__discount__":
                treated.loc[eligible, "MonthlyCharges"] = (
                    pd.to_numeric(treated.loc[eligible, "MonthlyCharges"], errors="coerce")
                    .fillna(0.0) * (1 - value)
                ).round(2)
            else:
                treated.loc[eligible, column] = value
        cost.loc[eligible] += charges.loc[eligible] * lever["cost_pct"] + lever["cost_flat"]
        touched |= eligible

    return treated, cost.round(2), touched


def _round(value, digits=2):
    return round(float(value), digits)


# ---------------------------------------------------------------------------
# Campaign simulation
# ---------------------------------------------------------------------------

def simulate_campaign(bundle, lever_ids, risk="All", contract="All",
                      tenure="all", min_spend=0.0, top_n=15):
    """Simulate a retention campaign over a segment of a scored run."""
    if bundle is None:
        raise RetentionPlannerError("Score a customer portfolio first.")
    if bundle.get("is_universal"):
        raise RetentionPlannerError(
            "The planner's retention levers are defined for the Telco schema. "
            "Re-run a Telco-format dataset (or the bundled sample portfolio) to plan a campaign."
        )
    cleaned = bundle.get("cleaned_df")
    if cleaned is None or cleaned.empty:
        raise RetentionPlannerError("This run does not contain customer-level data.")

    lever_ids = [lever_id for lever_id in lever_ids if lever_id in LEVERS_BY_ID]
    mask = segment_mask(bundle, risk=risk, contract=contract, tenure=tenure, min_spend=min_spend)
    segment = cleaned.loc[mask].copy()

    summary = {
        "segment_size": int(len(segment)),
        "portfolio_size": int(len(cleaned)),
        "levers": [LEVERS_BY_ID[lever_id] for lever_id in lever_ids],
        "filters": {
            "risk": risk, "contract": contract,
            "tenure": tenure, "min_spend": float(min_spend or 0),
        },
    }
    if segment.empty or not lever_ids:
        summary.update({"empty": True, "customers": [], "lever_breakdown": []})
        return summary

    charges = pd.to_numeric(segment["MonthlyCharges"], errors="coerce").fillna(0.0)
    probs_before = _score(segment)
    treated, cost, touched = _apply_levers(segment, lever_ids)
    probs_after = _score(treated)

    # Untreated customers must not drift: no lever applied means no change.
    probs_after = np.where(touched.values, probs_after, probs_before)
    delta = probs_before - probs_after

    revenue_at_risk_before = float((probs_before * charges.values).sum())
    revenue_at_risk_after = float((probs_after * charges.values).sum())
    revenue_protected = revenue_at_risk_before - revenue_at_risk_after
    campaign_cost = float(cost[touched].sum())
    net_monthly = revenue_protected - campaign_cost

    risk_before = pd.Series([calculate_risk_level(p) for p in probs_before]).value_counts()
    risk_after = pd.Series([calculate_risk_level(p) for p in probs_after]).value_counts()

    detail = pd.DataFrame({
        "Customer ID": segment["customerID"].astype(str).values,
        "Contract": segment["Contract"].values,
        "Tenure": pd.to_numeric(segment["tenure"], errors="coerce").fillna(0).astype(int).values,
        "Monthly Charges": charges.round(2).values,
        "Churn Probability Before (%)": np.round(probs_before * 100, 1),
        "Churn Probability After (%)": np.round(probs_after * 100, 1),
        "Reduction (pts)": np.round(delta * 100, 1),
        "Revenue Protected / mo": np.round(delta * charges.values, 2),
        "Campaign Cost / mo": cost.where(touched, 0.0).round(2).values,
        "Treated": np.where(touched.values, "Yes", "No"),
    })
    detail["Net Value / mo"] = (
        detail["Revenue Protected / mo"] - detail["Campaign Cost / mo"]
    ).round(2)

    top = detail.sort_values("Revenue Protected / mo", ascending=False).head(top_n)

    summary.update({
        "empty": False,
        "treated_count": int(touched.sum()),
        "treated_pct": _round(touched.sum() / len(segment) * 100, 1),
        "avg_probability_before": _round(probs_before.mean() * 100, 1),
        "avg_probability_after": _round(probs_after.mean() * 100, 1),
        "avg_reduction_points": _round(delta.mean() * 100, 1),
        "expected_churners_before": _round(probs_before.sum(), 1),
        "expected_churners_after": _round(probs_after.sum(), 1),
        "customers_saved": _round(delta.sum(), 1),
        "revenue_at_risk_before": _round(revenue_at_risk_before),
        "revenue_at_risk_after": _round(revenue_at_risk_after),
        "revenue_protected_monthly": _round(revenue_protected),
        "revenue_protected_annual": _round(revenue_protected * 12),
        "campaign_cost_monthly": _round(campaign_cost),
        "campaign_cost_annual": _round(campaign_cost * 12),
        "net_monthly": _round(net_monthly),
        "net_annual": _round(net_monthly * 12),
        "roi_pct": _round(net_monthly / campaign_cost * 100, 1) if campaign_cost > 0 else None,
        "cost_per_customer_saved": _round(campaign_cost / delta.sum()) if delta.sum() > 0.01 else None,
        "risk_mix_before": {level: int(risk_before.get(level, 0)) for level in ("High Risk", "Medium Risk", "Low Risk")},
        "risk_mix_after": {level: int(risk_after.get(level, 0)) for level in ("High Risk", "Medium Risk", "Low Risk")},
        "lever_breakdown": _lever_breakdown(segment, lever_ids, probs_before, charges),
        "customers": top.to_dict("records"),
        "detail_df": detail,
    })
    return summary


def _lever_breakdown(segment, lever_ids, probs_before, charges):
    """Impact of each selected lever applied on its own — attribution, not sum."""
    rows = []
    for lever_id in lever_ids:
        lever = LEVERS_BY_ID[lever_id]
        treated, cost, touched = _apply_levers(segment, [lever_id])
        if not touched.any():
            rows.append({
                "id": lever_id, "label": lever["label"], "cost_note": lever["cost_note"],
                "eligible": 0, "avg_reduction_points": 0.0,
                "revenue_protected": 0.0, "cost": 0.0, "net": 0.0,
            })
            continue
        probs_after = _score(treated)
        probs_after = np.where(touched.values, probs_after, probs_before)
        delta = probs_before - probs_after
        revenue = float((delta * charges.values).sum())
        lever_cost = float(cost[touched].sum())
        rows.append({
            "id": lever_id,
            "label": lever["label"],
            "cost_note": lever["cost_note"],
            "eligible": int(touched.sum()),
            "avg_reduction_points": _round(delta[touched.values].mean() * 100, 1),
            "revenue_protected": _round(revenue),
            "cost": _round(lever_cost),
            "net": _round(revenue - lever_cost),
        })
    rows.sort(key=lambda row: row["net"], reverse=True)
    return rows

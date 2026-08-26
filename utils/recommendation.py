"""
utils/recommendation.py
------------------------
Shared retention recommendation engine.

Used by the Flask UI, bulk CSV export, API, and notebooks so the same
customer always gets the same advice.
"""


def _get(record, *keys, default=""):
    for key in keys:
        if record is None:
            break
        if hasattr(record, "get"):
            value = record.get(key)
        else:
            try:
                value = record[key]
            except Exception:
                value = None
        if value is not None and not (isinstance(value, float) and value != value):
            return value
    return default


def _as_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_int(value, default=0):
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _normalize_profile(record, probability, risk_level):
    contract = str(_get(record, "Contract", default="Month-to-month"))
    one_year = _as_float(_get(record, "Contract_One year", default=0))
    two_year = _as_float(_get(record, "Contract_Two year", default=0))
    if contract in ("", "nan") or contract == "Month-to-month":
        if one_year == 1:
            contract = "One year"
        elif two_year == 1:
            contract = "Two year"
        elif "Contract" not in getattr(record, "index", []) and one_year == 0 and two_year == 0:
            # Encoded row with both dummies off is month-to-month.
            if "Contract_One year" in getattr(record, "index", []) or "Contract_One year" in getattr(record, "keys", lambda: [])():
                contract = "Month-to-month"

    monthly = _as_float(_get(record, "MonthlyCharges", "Monthly Charges", default=0))
    tenure = _as_int(_get(record, "tenure", "Tenure Months", "Tenure", default=0))

    internet = str(_get(record, "InternetService", "Internet Service", default=""))
    if not internet or internet == "nan":
        if _as_float(_get(record, "Internet Service_Fiber optic", default=0)) == 1:
            internet = "Fiber optic"
        elif _as_float(_get(record, "Internet Service_No", default=0)) == 1:
            internet = "No"
        else:
            internet = "DSL"

    tech = str(_get(record, "TechSupport", "Tech Support", default="No"))
    if tech in ("", "nan"):
        tech = "Yes" if _as_float(_get(record, "Tech Support_Yes", default=0)) == 1 else "No"

    security = str(_get(record, "OnlineSecurity", "Online Security", default="No"))
    if security in ("", "nan"):
        security = "Yes" if _as_float(_get(record, "Online Security_Yes", default=0)) == 1 else "No"

    payment = str(_get(record, "PaymentMethod", "Payment Method", default=""))
    if not payment or payment == "nan":
        if _as_float(_get(record, "Payment Method_Electronic check", default=0)) == 1:
            payment = "Electronic check"

    paperless = str(_get(record, "PaperlessBilling", "Paperless Billing", default="No"))
    if paperless in ("", "nan"):
        paperless = "Yes" if _as_float(_get(record, "Paperless Billing_Yes", default=0)) == 1 else "No"

    segment = str(_get(record, "Risk_Segment", "Risk Level", default=risk_level or ""))
    if not segment or segment == "nan":
        segment = risk_level or "Medium Risk"

    return {
        "contract": contract,
        "monthly": monthly,
        "tenure": tenure,
        "internet": internet,
        "tech": tech,
        "security": security,
        "payment": payment,
        "paperless": paperless,
        "risk": segment,
        "probability": float(probability or 0),
    }


def generate_recommendation(cleaned_record, probability: float, risk_level: str):
    """
    cleaned_record: pandas Series / mapping with raw or encoded fields.
    Returns: list[str] of recommendation strings, most important first.
    """
    p = _normalize_profile(cleaned_record, probability, risk_level)
    reasons = []
    issue_count = 0

    if p["contract"] == "Month-to-month":
        reasons.append(
            "Offer an incentive to switch from a month-to-month plan to a 1- or 2-year contract "
            "(e.g. a discounted rate for committing longer-term)."
        )
        issue_count += 1

    if p["monthly"] >= 80:
        reasons.append(
            "Review pricing -- this customer's monthly bill is high; consider a loyalty discount "
            "or a right-sized plan to reduce bill shock."
        )
        issue_count += 1

    if p["tech"] == "No" and p["internet"] != "No":
        reasons.append("Proactively offer a free trial of Tech Support -- customers without it churn more often.")
        issue_count += 1

    if p["security"] == "No" and p["internet"] != "No":
        reasons.append("Bundle in an Online Security add-on at a promotional rate to increase stickiness.")
        issue_count += 1

    if p["tenure"] >= 24 and p["probability"] >= 0.5:
        reasons.append(
            "This is a long-tenured customer showing high churn risk -- escalate to a loyalty "
            "retention specialist with a personalized win-back offer."
        )
        issue_count += 1

    if p["internet"] == "Fiber optic" and p["monthly"] >= 85 and p["risk"] == "High Risk":
        reasons.append(
            "Fiber customer paying a premium and at high risk -- schedule a personalized "
            "service/pricing review call before the next billing cycle."
        )
        issue_count += 1

    if p["payment"] == "Electronic check":
        reasons.append(
            "Encourage a switch to automatic payment (credit card / bank transfer) -- "
            "electronic-check customers show a higher churn tendency in this segment."
        )
        issue_count += 1

    if p["paperless"] == "Yes" and p["risk"] in ("Medium Risk", "High Risk"):
        reasons.append(
            "Send a personalized retention email/SMS (this customer is opted into paperless "
            "communication, making digital outreach the most effective channel)."
        )

    if issue_count >= 3:
        reasons.insert(
            0,
            "Multiple service and pricing risk factors detected -- route this account to "
            "priority customer support for a full account review.",
        )

    if not reasons:
        if p["risk"] == "Low Risk":
            reasons.append(
                "No urgent action needed -- continue standard engagement and periodic "
                "satisfaction check-ins to maintain loyalty."
            )
        else:
            reasons.append(
                "Monitor this account closely and consider a general satisfaction survey "
                "to catch emerging dissatisfaction early."
            )

    return reasons[:4]


def generate_recommendation_summary(cleaned_record, probability: float, risk_level: str) -> str:
    """A single-sentence version, used in bulk result tables."""
    full = generate_recommendation(cleaned_record, probability, risk_level)
    return full[0] if full else "Monitor account."


def retention_action_label(cleaned_record, probability: float = None, risk_level: str = None) -> str:
    """Short campaign label used by notebooks, Power BI, and CSV exports."""
    p = _normalize_profile(cleaned_record, probability or 0, risk_level or "")
    if p["risk"] == "High Risk":
        if p["monthly"] > 80:
            return "Offer 15% Discount"
        if p["tenure"] < 12:
            return "Welcome Retention Package"
        if p["security"] == "No" and p["internet"] != "No":
            return "Free Online Security"
        if p["tech"] == "No" and p["internet"] != "No":
            return "Free Premium Support"
        if p["contract"] == "Month-to-month":
            return "Promote Long-Term Contract"
        return "Personal Retention Call"
    if p["risk"] == "Medium Risk":
        if p["monthly"] > 80:
            return "5% Discount Offer"
        if p["contract"] == "Month-to-month":
            return "Promote Long-Term Contract"
        if p["payment"] == "Electronic check":
            return "Switch to Autopay"
        return "Satisfaction Check-in"
    return "Standard Engagement"

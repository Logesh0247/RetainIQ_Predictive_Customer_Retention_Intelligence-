"""
utils/recommendation.py
------------------------
Retention recommendation engine.

Generates one or more explainable recommendations based on a customer's
actual characteristics + risk profile. Recommendations are never identical
for every customer -- they are derived from the specific record.
"""


def generate_recommendation(cleaned_record, probability: float, risk_level: str):
    """
    cleaned_record: a pandas Series (row) with the cleaned raw fields
                     (Contract, MonthlyCharges, InternetService, etc.)
    Returns: list[str] of recommendation strings, most important first.
    """
    reasons = []

    contract = cleaned_record.get("Contract", "Month-to-month")
    monthly_charges = float(cleaned_record.get("MonthlyCharges", 0) or 0)
    tenure = int(cleaned_record.get("tenure", 0) or 0)
    tech_support = cleaned_record.get("TechSupport", "No")
    online_security = cleaned_record.get("OnlineSecurity", "No")
    internet_service = cleaned_record.get("InternetService", "No")
    payment_method = cleaned_record.get("PaymentMethod", "")
    paperless = cleaned_record.get("PaperlessBilling", "No")

    issue_count = 0

    if contract == "Month-to-month":
        reasons.append("Offer an incentive to switch from a month-to-month plan to a 1- or 2-year contract "
                        "(e.g. a discounted rate for committing longer-term).")
        issue_count += 1

    if monthly_charges >= 80:
        reasons.append("Review pricing -- this customer's monthly bill is high; consider a loyalty discount "
                        "or a right-sized plan to reduce bill shock.")
        issue_count += 1

    if tech_support == "No" and internet_service != "No":
        reasons.append("Proactively offer a free trial of Tech Support -- customers without it churn more often.")
        issue_count += 1

    if online_security == "No" and internet_service != "No":
        reasons.append("Bundle in an Online Security add-on at a promotional rate to increase stickiness.")
        issue_count += 1

    if tenure >= 24 and probability >= 0.5:
        reasons.append("This is a long-tenured customer showing high churn risk -- escalate to a loyalty "
                        "retention specialist with a personalized win-back offer.")
        issue_count += 1

    if internet_service == "Fiber optic" and monthly_charges >= 85 and risk_level == "High Risk":
        reasons.append("Fiber customer paying a premium and at high risk -- schedule a personalized "
                        "service/pricing review call before the next billing cycle.")
        issue_count += 1

    if payment_method == "Electronic check":
        reasons.append("Encourage a switch to automatic payment (credit card / bank transfer) -- "
                        "electronic-check customers show a higher churn tendency in this segment.")
        issue_count += 1

    if paperless == "Yes" and risk_level in ("Medium Risk", "High Risk"):
        reasons.append("Send a personalized retention email/SMS (this customer is opted into paperless "
                        "communication, making digital outreach the most effective channel).")

    if issue_count >= 3:
        reasons.insert(0, "Multiple service and pricing risk factors detected -- route this account to "
                           "priority customer support for a full account review.")

    if not reasons:
        if risk_level == "Low Risk":
            reasons.append("No urgent action needed -- continue standard engagement and periodic "
                            "satisfaction check-ins to maintain loyalty.")
        else:
            reasons.append("Monitor this account closely and consider a general satisfaction survey "
                            "to catch emerging dissatisfaction early.")

    return reasons[:4]


def generate_recommendation_summary(cleaned_record, probability: float, risk_level: str) -> str:
    """A single-sentence version, used in bulk result tables."""
    full = generate_recommendation(cleaned_record, probability, risk_level)
    return full[0] if full else "Monitor account."

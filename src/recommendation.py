"""
==========================================================
Customer Retention Intelligence Recommendation Engine
==========================================================

This module generates personalized retention strategies
based on:

• Churn Risk
• Customer Contract
• Monthly Charges
• Tenure
• Internet Services
• Security Services
• Payment Behaviour
• Customer Profile

==========================================================
"""


# ==========================================================
# Risk Classification
# ==========================================================

def get_risk(probability):
    """
    Convert probability into risk category.
    """

    probability = probability * 100

    if probability >= 70:

        return "High"

    elif probability >= 40:

        return "Medium"

    else:

        return "Low"


# ==========================================================
# Recommendation Engine
# ==========================================================

def get_recommendation(customer, risk):

    recommendations = []

    # ======================================================
    # Base Recommendations
    # ======================================================

    if risk == "High":

        recommendations.extend([

            "Contact the customer within the next 24 hours.",

            "Assign a dedicated Relationship Manager."

        ])

    elif risk == "Medium":

        recommendations.extend([

            "Schedule a proactive engagement call.",

            "Send personalized loyalty offers."

        ])

    else:

        recommendations.extend([

            "Send a customer appreciation message.",

            "Reward customer loyalty with exclusive benefits."

        ])


    # ======================================================
    # Contract Analysis
    # ======================================================

    contract = customer.get("Contract", "")

    if contract == "Month-to-month":

        recommendations.append(

            "Offer a discounted 12-month or 24-month contract."

        )

    elif contract == "One year":

        recommendations.append(

            "Recommend upgrading to a two-year contract."

        )

    elif contract == "Two year":

        recommendations.append(

            "Reward the customer with exclusive loyalty benefits."

        )


    # ======================================================
    # Monthly Charges Analysis
    # ======================================================

    monthly = float(customer.get("Monthly Charges", 0))

    if monthly >= 90:

        recommendations.append(

            "Provide a 20% loyalty discount on monthly charges."

        )

    elif monthly >= 70:

        recommendations.append(

            "Recommend a personalized pricing plan."

        )

    elif monthly <= 30:

        recommendations.append(

            "Promote premium value-added services."

        )


    # ======================================================
    # Tenure Analysis
    # ======================================================

    tenure = float(customer.get("Tenure Months", 0))

    if tenure < 6:

        recommendations.append(

            "Assign a Customer Success Manager during onboarding."

        )

    elif tenure < 12:

        recommendations.append(

            "Provide a welcome retention package."

        )

    elif tenure >= 48:

        recommendations.append(

            "Reward long-term loyalty with premium membership."

        )


    # ======================================================
    # Internet Service
    # ======================================================

    internet = customer.get("Internet Service", "")

    if internet == "Fiber optic":

        recommendations.append(

            "Offer an exclusive Fiber Optic loyalty package."

        )

    elif internet == "DSL":

        recommendations.append(

            "Offer a discounted high-speed internet upgrade."

        )
        
            # ======================================================
    # Online Security
    # ======================================================

    if customer.get("Online Security", "") == "No":

        recommendations.append(

            "Offer a complimentary Online Security package for 3 months."

        )

    else:

        recommendations.append(

            "Thank the customer for using Online Security services."

        )


    # ======================================================
    # Online Backup
    # ======================================================

    if customer.get("Online Backup", "") == "No":

        recommendations.append(

            "Provide a free Online Backup trial."

        )

    else:

        recommendations.append(

            "Recommend increasing cloud backup storage."

        )


    # ======================================================
    # Device Protection
    # ======================================================

    if customer.get("Device Protection", "") == "No":

        recommendations.append(

            "Offer Device Protection at a discounted price."

        )


    # ======================================================
    # Technical Support
    # ======================================================

    if customer.get("Tech Support", "") == "No":

        recommendations.append(

            "Provide complimentary Technical Support for 90 days."

        )

    else:

        recommendations.append(

            "Invite the customer to premium technical assistance."

        )


    # ======================================================
    # Streaming TV
    # ======================================================

    if customer.get("Streaming TV", "") == "No":

        recommendations.append(

            "Offer a discounted Streaming TV bundle."

        )


    # ======================================================
    # Streaming Movies
    # ======================================================

    if customer.get("Streaming Movies", "") == "No":

        recommendations.append(

            "Provide an Entertainment Pack with Streaming Movies."

        )


    # ======================================================
    # Multiple Lines
    # ======================================================

    if customer.get("Multiple Lines", "") == "No":

        recommendations.append(

            "Recommend a Family Multi-Line Plan."

        )


    # ======================================================
    # Payment Method
    # ======================================================

    payment = customer.get("Payment Method", "")

    if payment == "Electronic check":

        recommendations.append(

            "Encourage AutoPay enrollment with cashback rewards."

        )

    elif payment == "Mailed check":

        recommendations.append(

            "Recommend secure online payment methods."

        )

    elif payment == "Bank transfer (automatic)":

        recommendations.append(

            "Reward AutoPay customers with loyalty points."

        )

    elif payment == "Credit card (automatic)":

        recommendations.append(

            "Offer exclusive credit card cashback rewards."

        )


    # ======================================================
    # Paperless Billing
    # ======================================================

    if customer.get("Paperless Billing", "") == "Yes":

        recommendations.append(

            "Send exclusive digital loyalty coupons."

        )


    # ======================================================
    # Partner
    # ======================================================

    if customer.get("Partner", "") == "No":

        recommendations.append(

            "Introduce referral rewards for inviting family members."

        )


    # ======================================================
    # Dependents
    # ======================================================

    if customer.get("Dependents", "") == "Yes":

        recommendations.append(

            "Recommend family protection and bundled service plans."

        )


    # ======================================================
    # Senior Citizen
    # ======================================================

    senior = customer.get("Senior Citizen", 0)

    if senior in [1, "1", "Yes", True]:

        recommendations.append(

            "Provide an exclusive Senior Citizen loyalty discount."

        )


    # ======================================================
    # Phone Service
    # ======================================================

    if customer.get("Phone Service", "") == "No":

        recommendations.append(

            "Offer an affordable phone service bundle."

        )
        
            # ======================================================
    # Risk-Specific Smart Recommendations
    # ======================================================

    if risk == "High":

        recommendations.extend([

            "Prioritize this customer for immediate retention campaigns.",

            "Monitor customer activity weekly.",

            "Provide a personalized loyalty package."

        ])

    elif risk == "Medium":

        recommendations.extend([

            "Monitor customer satisfaction monthly.",

            "Send personalized promotional emails.",

            "Recommend suitable service upgrades."

        ])

    else:

        recommendations.extend([

            "Maintain regular customer engagement.",

            "Invite the customer to the referral program.",

            "Promote premium value-added services."

        ])


    # ======================================================
    # Remove Duplicate Recommendations
    # ======================================================

    unique_recommendations = []

    for recommendation in recommendations:

        recommendation = recommendation.strip()

        if recommendation not in unique_recommendations:

            unique_recommendations.append(recommendation)


    # ======================================================
    # Intelligent Prioritization
    # ======================================================

    priority_keywords = [

        "Contact",

        "Assign",

        "Offer",

        "Provide",

        "Recommend",

        "Reward",

        "Monitor"

    ]

    priority_recommendations = []

    normal_recommendations = []

    for recommendation in unique_recommendations:

        if any(

            recommendation.startswith(keyword)

            for keyword in priority_keywords

        ):

            priority_recommendations.append(recommendation)

        else:

            normal_recommendations.append(recommendation)

    final_recommendations = (

        priority_recommendations +

        normal_recommendations

    )


    # ======================================================
    # Limit Recommendations Based on Risk
    # ======================================================

    if risk == "High":

        final_recommendations = final_recommendations[:8]

    elif risk == "Medium":

        final_recommendations = final_recommendations[:6]

    else:

        final_recommendations = final_recommendations[:5]


    # ======================================================
    # Fallback Recommendation
    # ======================================================

    if len(final_recommendations) == 0:

        final_recommendations.append(

            "Continue providing excellent customer service."

        )


    # ======================================================
    # Return Final Recommendation List
    # ======================================================

    return final_recommendations
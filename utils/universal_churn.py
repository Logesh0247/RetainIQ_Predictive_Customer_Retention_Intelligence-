"""
Universal Churn Prediction Engine — accepts any CSV with a churn target column.
Auto-detects schema, trains a GradientBoosting model on-the-fly.
"""
import logging
import re
import warnings
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder

logger = logging.getLogger("retainiq")

CHURN_TARGET_PATTERNS = [
    r'churn', r'churned', r'cancelled', r'canceled', r'left', r'exited', r'exit',
    r'attrited', r'attrition', r'lost', r'inactive', r'dropped', r'turnover',
    r'retained', r'retention', r'stay', r'stayed', r'subscribed', r'renewed',
]

CHURN_POSITIVE_VALUES = {
    'yes', 'y', '1', 'true', 't', 'churned', 'churn', 'left', 'exited',
    'cancelled', 'canceled', 'lost', 'inactive', 'dropped', 'attrited',
}

CHURN_NEGATIVE_VALUES = {
    'no', 'n', '0', 'false', 'f', 'retained', 'stayed', 'active', 'kept',
    'renewed', 'continued',
}

NEGATIVE_DIRECTION_PATTERNS = [
    r'support|ticket|complaint|issue|problem',
    r'response_time|resolution_time|wait_time',
    r'overdue|delinquent|late_payment|arrear',
    r'bug|error|crash|downtime|outage',
    r'overtime|workload|hours_worked',
    r'distance|commute',
    r'cart_abandon|abandonment',
    r'last_login|days_since|since_last|recency|inactive_days',
    r'churn_risk|risk_score',
]

RECOMMENDATION_RULES = [
    (r'login_count|log_in_count|sign_in_count', {"low": "Customer shows low login activity — send a re-engagement email with a personalized reason to come back", "high": "Customer is highly active — leverage engagement with loyalty rewards or referral incentives"}),
    (r'last_login|days_since_login|inactive_days', {"low": "Customer logged in recently — engagement is healthy", "high": "Customer hasn't logged in for a while — trigger an immediate re-engagement campaign"}),
    (r'visit|page_view|session_count|sessions', {"low": "Low visit frequency — trigger a win-back campaign with compelling content", "high": "Frequent visitor — consider offering premium features or an upgrade path"}),
    (r'engagement|interaction|activity_score', {"low": "Engagement is below average — schedule a personalized check-in", "high": "Strong engagement — candidate for advocacy and referral programs"}),
    (r'email_open|email_click|open_rate|click_rate', {"low": "Low email engagement — try alternative channels (SMS, in-app, phone)", "high": "Responsive to emails — use email as the primary retention channel"}),
    (r'support|ticket|complaint|issue|problem', {"low": "Few support interactions — maintain proactive communication", "high": "High support ticket volume — escalate to a senior support agent and review unresolved issues"}),
    (r'response_time|resolution|wait_time|sla', {"low": "Service response times are good — maintain current SLA", "high": "Long resolution times — prioritize this customer's open tickets"}),
    (r'satisfaction|csat|nps|rating_score', {"low": "Low satisfaction score — conduct a personal outreach call", "high": "High satisfaction — invite to a referral or case study program"}),
    (r'charge|billing|fee|price|cost|revenue|spend|amount|invoice|monthly_spend', {"low": "Low billing amount — consider upselling to a higher-value plan", "high": "High billing amount at risk — offer a loyalty discount or payment plan"}),
    (r'payment_method|payment_type', {"low": "Standard payment method — no specific payment-related churn risk", "high": "Review payment method — consider incentivizing automatic payment"}),
    (r'discount|coupon|promo', {"low": "Rarely uses discounts — highlight value-add services", "high": "Heavy discount usage — review whether full-price renewal is sustainable"}),
    (r'overdue|delinquent|late_payment|arrear', {"low": "Payment history is clean — maintain standard billing", "high": "Overdue payments — reach out with flexible payment options"}),
    (r'tenure|duration|days_since_signup|months_active|years_active|length|loyalty', {"low": "New customer — activate onboarding retention program", "high": "Long-tenured customer — recognize loyalty with VIP benefits"}),
    (r'usage|consumption|feature_use|adoption', {"low": "Low product usage — send guided tutorials", "high": "Heavy usage — ensure reliability and consider upselling"}),
    (r'order|purchase|transaction|buy', {"low": "Few recent purchases — trigger product recommendation campaign", "high": "Active buyer — offer VIP status or early access"}),
    (r'cart|abandon|wishlist', {"low": "Low cart activity — send personalized suggestions", "high": "Frequent cart abandonment — offer a time-limited discount"}),
    (r'plan|subscription|tier|level|grade', {"low": "On a basic plan — demonstrate value of upgrading", "high": "On a premium plan — ensure premium support quality"}),
    (r'contract|commitment|agreement', {"low": "Short-term contract — offer incentives for longer commitment", "high": "Long-term contract — track renewal before expiration"}),
    (r'auto_renew|renewal', {"low": "Auto-renewal is off — send a reminder with incentive", "high": "Auto-renewal active — verify satisfaction before renewal"}),
    (r'upgrade|downgrade', {"low": "No upgrade history — suggest higher tier with clear ROI", "high": "Recent downgrade — investigate and offer win-back plan"}),
    (r'age|senior', {"low": "Younger customer — emphasize digital-first engagement", "high": "Older customer — ensure accessible support channels"}),
    (r'product|service|quality|performance|uptime|speed|latency', {"low": "Product experience is healthy — maintain service levels", "high": "Service quality issues — proactively compensate"}),
    (r'bug|error|crash|downtime|outage', {"low": "No significant technical issues — continue monitoring", "high": "Multiple technical issues — assign dedicated support and offer credits"}),
    (r'call_count|phone|contact_count', {"low": "No recent contact — schedule a proactive wellness call", "high": "Frequent contacts — review for unresolved issues"}),
    (r'salary|income|compensation|pay', {"low": "Below-median compensation — consider retention bonus", "high": "Above-median compensation — focus on non-monetary retention"}),
    (r'overtime|workload|hours_worked', {"low": "Reasonable workload — maintain work-life balance", "high": "Excessive overtime — address workload to prevent burnout"}),
    (r'promotion|career|growth|development', {"low": "Limited career progression — discuss development opportunities", "high": "Active career growth — maintain with stretch assignments"}),
    (r'distance|commute|remote', {"low": "Short commute or remote — not a retention risk", "high": "Long commute — offer flexible/remote work options"}),
    (r'training|learning|course', {"low": "Low training participation — encourage enrollment", "high": "Active learner — invest in advanced certifications"}),
    (r'balance|deposit|saving', {"low": "Low account balance — offer financial wellness tools", "high": "High balance at risk — assign relationship manager"}),
    (r'credit_score|credit_rating', {"low": "Lower credit profile — offer credit-building products", "high": "Strong credit profile — cross-sell premium products"}),
    (r'num_product|products|service_count|num_', {"low": "Uses few products — cross-sell complementary services", "high": "Multi-product customer — ensure seamless experience"}),
    (r'days_since|since_last|recency', {"low": "Recent activity is healthy — maintain engagement", "high": "Long time since last activity — trigger re-engagement campaign"}),
    (r'frequency|rate|count|number', {"low": "Below-average frequency — investigate barriers", "high": "Above-average frequency — recognize and reward"}),
]


class UniversalChurnError(ValueError):
    pass


def detect_target_column(df):
    columns_lower = {col.lower(): col for col in df.columns}
    for pattern in CHURN_TARGET_PATTERNS:
        if pattern in columns_lower:
            col = columns_lower[pattern]
            if _is_binary_target(df[col]):
                return col, True
    for col in df.columns:
        col_lower = col.lower()
        for pattern in CHURN_TARGET_PATTERNS:
            if pattern in col_lower:
                if _is_binary_target(df[col]):
                    return col, True
    # No target column found — return None instead of raising
    return None, False


def _is_binary_target(series):
    unique_vals = series.dropna().astype(str).str.lower().str.strip().unique()
    if len(unique_vals) != 2:
        return False
    has_positive = any(val in CHURN_POSITIVE_VALUES for val in unique_vals)
    has_negative = any(val in CHURN_NEGATIVE_VALUES for val in unique_vals)
    if set(unique_vals) in [{'0', '1'}, {'yes', 'no'}, {'true', 'false'}, {'y', 'n'}]:
        return True
    return has_positive and has_negative


def detect_schema(df, target_col):
    schema = {'target': target_col, 'numeric': [], 'categorical': [], 'datetime': [], 'skip': []}
    for col in df.columns:
        if col == target_col:
            continue
        if _is_id_column(col, df[col]):
            schema['skip'].append(col)
            continue
        if _is_datetime_column(df[col]):
            schema['datetime'].append(col)
            continue
        if _is_numeric_column(df[col]):
            schema['numeric'].append(col)
            continue
        if _is_categorical_column(df[col]):
            schema['categorical'].append(col)
        else:
            schema['skip'].append(col)
    return schema


def _is_id_column(col_name, series):
    col_lower = col_name.lower()
    if any(kw in col_lower for kw in ['id', 'customer_id', 'user_id', 'account_id', 'uid', 'cid']):
        return True
    if series.dtype == 'object':
        try:
            pd.to_numeric(series, errors='raise')
        except:
            if series.nunique() == len(series):
                return True
    return False


def _is_datetime_column(series):
    try:
        pd.to_numeric(series, errors='raise')
        return False
    except:
        pass
    try:
        sample = series.dropna().head(10)
        if len(sample) == 0:
            return False
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pd.to_datetime(sample, errors='raise')
        return True
    except:
        return False


def _is_numeric_column(series):
    try:
        pd.to_numeric(series, errors='raise')
        return True
    except:
        return False


def _is_categorical_column(series, max_cardinality=50):
    nunique = series.nunique()
    return nunique <= max_cardinality and nunique > 1


def encode_target(series):
    series_str = series.astype(str).str.lower().str.strip()
    encoded = series_str.map(lambda x: 1 if x in CHURN_POSITIVE_VALUES else 0)
    if encoded.isna().any():
        le = LabelEncoder()
        encoded = pd.Series(le.fit_transform(series_str), index=series.index)
        positive_class = le.classes_[1] if len(le.classes_) > 1 else le.classes_[0]
    else:
        positive_class = series[encoded == 1].iloc[0] if (encoded == 1).any() else 'Churned'
    return encoded, str(positive_class)


def auto_feature_engineer(df, schema):
    features = pd.DataFrame(index=df.index)
    for col in schema['numeric']:
        numeric_col = pd.to_numeric(df[col], errors='coerce')
        features[col] = numeric_col.fillna(numeric_col.median())
    for col in schema['categorical']:
        dummies = pd.get_dummies(df[col], prefix=col, drop_first=False)
        if dummies.shape[1] > 20:
            top_cats = df[col].value_counts().head(19).index
            dummies = pd.get_dummies(df[col].where(df[col].isin(top_cats), 'Other'), prefix=col, drop_first=False)
        features = pd.concat([features, dummies], axis=1)
    for col in schema['datetime']:
        try:
            dt_col = pd.to_datetime(df[col], errors='coerce')
            features[f'{col}_year'] = dt_col.dt.year.fillna(0)
            features[f'{col}_month'] = dt_col.dt.month.fillna(0)
            features[f'{col}_day'] = dt_col.dt.day.fillna(0)
            features[f'{col}_dayofweek'] = dt_col.dt.dayofweek.fillna(0)
        except:
            pass
    features = features.apply(pd.to_numeric, errors='coerce').fillna(0)
    return features, list(features.columns)


def compute_feature_stats(X, schema):
    stats = {}
    for col in X.columns:
        series = X[col]
        if series.dtype == bool:
            series = series.astype(float)
        parent_col = None
        category_value = None
        if "_" in col:
            parts = col.rsplit("_", 1)
            if len(parts) == 2 and parts[0] in schema.get("categorical", []):
                parent_col = parts[0]
                category_value = parts[1]
        try:
            stats[col] = {
                "median": float(series.median()), "mean": float(series.mean()),
                "q25": float(series.quantile(0.25)), "q75": float(series.quantile(0.75)),
                "std": float(series.std()) if len(series) > 1 else 0.0,
                "is_onehot": parent_col is not None, "parent_col": parent_col, "category_value": category_value,
            }
        except (TypeError, ValueError):
            stats[col] = {"median": 0.0, "mean": 0.0, "q25": 0.0, "q75": 0.0, "std": 0.0,
                          "is_onehot": parent_col is not None, "parent_col": parent_col, "category_value": category_value}
    return stats


def train_universal_model(X, y, test_size=0.2, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
    model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=random_state)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    metrics = {
        'accuracy': round(accuracy_score(y_test, y_pred), 4),
        'precision': round(precision_score(y_test, y_pred, zero_division=0), 4),
        'recall': round(recall_score(y_test, y_pred, zero_division=0), 4),
        'f1': round(f1_score(y_test, y_pred, zero_division=0), 4),
        'roc_auc': round(roc_auc_score(y_test, y_proba), 4),
    }
    return model, metrics


def _is_negative_direction(feature_name):
    name_lower = feature_name.lower().replace(" ", "_")
    for pattern in NEGATIVE_DIRECTION_PATTERNS:
        if re.search(pattern, name_lower):
            return True
    return False


def _humanize_feature_name(name):
    if "_" in name:
        parts = name.split("_")
        if len(parts) >= 2 and parts[-1][0].islower():
            return f"{' '.join(parts[:-1])} ({parts[-1]})"
    return name.replace("_", " ").title()


def _match_recommendation(feature_name, value, median, is_onehot):
    name_lower = feature_name.lower().replace(" ", "_")
    negative_direction = _is_negative_direction(feature_name)
    for pattern, recs in RECOMMENDATION_RULES:
        if re.search(pattern, name_lower):
            if is_onehot:
                if value > 0.5:
                    return recs.get("high"), 1.0
                return None, 0.0
            else:
                if negative_direction:
                    if value > median:
                        return recs.get("high"), 1.0
                    elif value < median:
                        return recs.get("low"), 1.0
                else:
                    if value < median:
                        return recs.get("low"), 1.0
                    elif value > median:
                        return recs.get("high"), 1.0
                return None, 0.0
    if is_onehot:
        if value > 0.5:
            return f"Customer has {_humanize_feature_name(feature_name)} — review if this correlates with retention risk", 0.5
    else:
        human = _humanize_feature_name(feature_name)
        if value < median:
            return f"{human} is below average — investigate if improving this could reduce churn", 0.5
        elif value > median:
            return f"{human} is above average — assess if this is a risk factor or opportunity", 0.5
    return None, 0.0


def generate_universal_recommendations(row, probability, risk_level,
                                        feature_values=None, feature_names=None,
                                        feature_importances=None, feature_stats=None):
    if feature_values is None or feature_names is None or feature_importances is None:
        return _generic_recommendations(risk_level)
    scored_recs = []
    for name, value, importance in zip(feature_names, feature_values, feature_importances):
        if importance <= 0:
            continue
        stats = feature_stats.get(name, {}) if feature_stats else {}
        median = stats.get("median", 0)
        is_onehot = stats.get("is_onehot", False)
        rec_text, rec_score = _match_recommendation(name, value, median, is_onehot)
        if rec_text:
            scored_recs.append((importance * rec_score, rec_text))
    scored_recs.sort(key=lambda x: x[0], reverse=True)
    recommendations = []
    seen = set()
    for score, rec in scored_recs:
        if rec not in seen:
            seen.add(rec)
            recommendations.append(rec)
        if len(recommendations) >= 4:
            break
    if risk_level == "High Risk" and recommendations:
        recommendations.insert(0, f"Priority retention case — {probability*100:.0f}% churn probability driven by {len(scored_recs)} risk factor{'s' if len(scored_recs) != 1 else ''}. Immediate outreach recommended.")
    elif risk_level == "Medium Risk" and recommendations:
        recommendations.insert(0, f"Proactive intervention suggested — {probability*100:.0f}% churn probability. Address the factors below within the next week.")
    if not recommendations:
        return _generic_recommendations(risk_level)
    return recommendations[:5]


def _generic_recommendations(risk_level):
    if risk_level == 'High Risk':
        return ["Immediate intervention — reach out within 24 hours", "Offer personalized retention incentives", "Conduct satisfaction survey", "Assign dedicated account manager"]
    elif risk_level == 'Medium Risk':
        return ["Schedule proactive check-in within 1 week", "Review recent interactions for warning signs", "Offer loyalty rewards or upgrade options", "Monitor engagement metrics closely"]
    return ["Continue standard engagement strategy", "Consider for referral program", "Periodic satisfaction check", "Explore upsell/cross-sell opportunities"]


def _heuristic_score_features(X, feature_names, feature_stats):
    """
    Score each row's churn risk heuristically using domain knowledge patterns.
    Returns an array of probabilities in [0, 1].
    """
    n = len(X)
    scores = np.zeros(n)
    weights_total = 0.0
    for feat_idx, name in enumerate(feature_names):
        name_lower = name.lower().replace(" ", "_")
        # Determine if this feature matters for churn
        is_negative = _is_negative_direction(name)
        # Check recommendation rules for relevance
        relevance = 0.0
        for pattern, _ in RECOMMENDATION_RULES:
            if re.search(pattern, name_lower):
                relevance = 1.0
                break
        if relevance == 0.0:
            continue
        stats = feature_stats.get(name, {})
        median = stats.get("median", 0)
        q25 = stats.get("q25", median)
        q75 = stats.get("q75", median)
        std = stats.get("std", 1.0) or 1.0
        is_onehot = stats.get("is_onehot", False)
        values = X.iloc[:, feat_idx].values.astype(float)
        if is_onehot:
            # One-hot: presence = contributes to risk score
            contrib = values * relevance
        elif is_negative:
            # Higher = more risk (e.g., support tickets, payment delays)
            contrib = np.clip((values - median) / std, -2, 2) * relevance
        else:
            # Lower = more risk (e.g., tenure, logins, engagement)
            contrib = np.clip((median - values) / std, -2, 2) * relevance
        scores += contrib
        weights_total += relevance
    if weights_total > 0:
        scores /= weights_total
    # Convert to probability via sigmoid centered at 0
    probabilities = 1.0 / (1.0 + np.exp(-scores * 1.5))
    return probabilities


def run_universal_churn(file_storage):
    try:
        df = pd.read_csv(file_storage, keep_default_na=False, na_values=[])
        if df.empty:
            raise UniversalChurnError("The uploaded CSV is empty.")
        if len(df) > 100000:
            raise UniversalChurnError(f"The uploaded CSV has too many rows ({len(df):,}). Maximum is 100,000.")

        target_col, is_binary = detect_target_column(df)

        # Heuristic mode: no target column found
        if target_col is None:
            logger.info("No target column found — using heuristic risk scoring")
            schema = detect_schema(df, None)
            X, feature_names = auto_feature_engineer(df, schema)
            if X.empty or X.shape[1] == 0:
                raise UniversalChurnError("No valid features could be extracted from the dataset.")
            feature_stats = compute_feature_stats(X, schema)
            # Heuristic scoring
            probabilities = _heuristic_score_features(X, feature_names, feature_stats)
            # No model — use uniform pseudo-importances weighted by feature relevance
            feature_importances = np.zeros(len(feature_names))
            for i, name in enumerate(feature_names):
                name_lower = name.lower().replace(" ", "_")
                for pattern, _ in RECOMMENDATION_RULES:
                    if re.search(pattern, name_lower):
                        feature_importances[i] = 1.0
                        break
            total_imp = feature_importances.sum()
            if total_imp > 0:
                feature_importances /= total_imp

            predictions = (probabilities >= 0.5).astype(int)
            risk_levels = ['High Risk' if p >= 0.7 else 'Medium Risk' if p >= 0.4 else 'Low Risk' for p in probabilities]
            metrics = {'accuracy': None, 'precision': None, 'recall': None, 'f1': None, 'roc_auc': None,
                       'note': 'Heuristic scoring (no labeled target column found)'}

            results_df = df.copy()
            results_df['Churn Probability'] = np.round(probabilities * 100, 1)
            results_df['Churn_Probability'] = np.round(probabilities, 4)
            results_df['Predicted_Churn'] = predictions
            results_df['Prediction'] = ['Likely to Churn' if p == 1 else 'Likely to Stay' for p in predictions]
            results_df['Risk Level'] = risk_levels
            results_df['Risk_Segment'] = risk_levels
            if 'customerID' not in results_df.columns:
                id_col = None
                for col in df.columns:
                    if _is_id_column(col, df[col]):
                        id_col = col
                        break
                if id_col:
                    results_df['customerID'] = df[id_col]
                else:
                    results_df['customerID'] = [f"CUST-{i+1}" for i in range(len(results_df))]
            results_df['Customer ID'] = results_df['customerID']
            # Telco-compatible columns
            if 'Monthly Charges' not in results_df.columns:
                for col in schema['numeric']:
                    if any(kw in col.lower() for kw in ['charge', 'billing', 'fee', 'price', 'amount', 'revenue', 'cost', 'spend']):
                        results_df['Monthly Charges'] = pd.to_numeric(results_df[col], errors='coerce').fillna(0).round(2)
                        break
                else:
                    results_df['Monthly Charges'] = 0.0
            if 'Tenure' not in results_df.columns:
                for col in schema['numeric']:
                    if any(kw in col.lower() for kw in ['tenure', 'duration', 'months', 'years', 'since']):
                        results_df['Tenure'] = pd.to_numeric(results_df[col], errors='coerce').fillna(0).astype(int)
                        break
                else:
                    results_df['Tenure'] = 0
            if 'Contract' not in results_df.columns:
                for col in schema['categorical']:
                    if 'contract' in col.lower():
                        results_df['Contract'] = results_df[col]
                        break
                else:
                    results_df['Contract'] = 'Month-to-month'
            if 'Internet Service' not in results_df.columns:
                results_df['Internet Service'] = 'N/A'
            if 'Payment Method' not in results_df.columns:
                for col in schema['categorical']:
                    if 'payment' in col.lower():
                        results_df['Payment Method'] = results_df[col]
                        break
                else:
                    results_df['Payment Method'] = 'N/A'
            if 'Gender' not in results_df.columns:
                for col in schema['categorical']:
                    if 'gender' in col.lower():
                        results_df['Gender'] = results_df[col]
                        break
                else:
                    results_df['Gender'] = 'N/A'
            if 'Senior Citizen' not in results_df.columns:
                results_df['Senior Citizen'] = 0

            # Recommendations
            all_recommendations = []
            for i in range(len(results_df)):
                recs = generate_universal_recommendations(
                    results_df.iloc[i], float(probabilities[i]), risk_levels[i],
                    feature_values=X.iloc[i].values, feature_names=feature_names,
                    feature_importances=feature_importances, feature_stats=feature_stats,
                )
                all_recommendations.append(recs)
            summaries = [recs[0] for recs in all_recommendations]
            results_df['Recommendation'] = summaries
            results_df['Retention_Recommendation'] = summaries
            at_risk = [level in ("High Risk", "Medium Risk") for level in risk_levels]
            results_df['Potential_Revenue_Saved'] = np.where(at_risk, results_df['Monthly Charges'].astype(float).round(2), 0.0)

            return {
                'results_df': results_df, 'cleaned_df': df, 'feature_frame': X,
                'probabilities': probabilities, 'model': None, 'metrics': metrics,
                'schema': schema, 'target_column': None, 'positive_class': 'Churned',
                'feature_names': feature_names, 'feature_stats': feature_stats,
                'feature_importances': feature_importances, 'all_recommendations': all_recommendations,
                'is_universal': True, 'heuristic_mode': True,
            }

        # Normal mode: target column found, train a model
        y, positive_class = encode_target(df[target_col])
        schema = detect_schema(df, target_col)
        X, feature_names = auto_feature_engineer(df, schema)
        if X.empty or X.shape[1] == 0:
            raise UniversalChurnError("No valid features could be extracted from the dataset.")
        model, metrics = train_universal_model(X, y)
        feature_stats = compute_feature_stats(X, schema)
        feature_importances = model.feature_importances_
        probabilities = model.predict_proba(X)[:, 1]
        predictions = model.predict(X)
        risk_levels = []
        for prob in probabilities:
            if prob >= 0.7:
                risk_levels.append('High Risk')
            elif prob >= 0.4:
                risk_levels.append('Medium Risk')
            else:
                risk_levels.append('Low Risk')
        results_df = df.copy()
        results_df['Churn Probability'] = np.round(probabilities * 100, 1)
        results_df['Churn_Probability'] = np.round(probabilities, 4)
        results_df['Predicted_Churn'] = predictions
        results_df['Prediction'] = ['Likely to Churn' if p == 1 else 'Likely to Stay' for p in predictions]
        results_df['Risk Level'] = risk_levels
        results_df['Risk_Segment'] = risk_levels
        if 'customerID' not in results_df.columns:
            results_df['customerID'] = [f"CUST-{i+1}" for i in range(len(results_df))]
        results_df['Customer ID'] = results_df['customerID']
        # Telco-compatible columns
        if 'Monthly Charges' not in results_df.columns:
            for col in schema['numeric']:
                if any(kw in col.lower() for kw in ['charge', 'billing', 'fee', 'price', 'amount', 'revenue', 'cost', 'spend']):
                    results_df['Monthly Charges'] = pd.to_numeric(results_df[col], errors='coerce').fillna(0).round(2)
                    break
            else:
                results_df['Monthly Charges'] = 0.0
        if 'Tenure' not in results_df.columns:
            for col in schema['numeric']:
                if any(kw in col.lower() for kw in ['tenure', 'duration', 'age', 'months', 'years', 'since']):
                    results_df['Tenure'] = pd.to_numeric(results_df[col], errors='coerce').fillna(0).astype(int)
                    break
            else:
                results_df['Tenure'] = 0
        if 'Contract' not in results_df.columns:
            for col in schema['categorical']:
                if 'contract' in col.lower():
                    results_df['Contract'] = results_df[col]
                    break
            else:
                results_df['Contract'] = 'Month-to-month'
        if 'Internet Service' not in results_df.columns:
            results_df['Internet Service'] = 'N/A'
        if 'Payment Method' not in results_df.columns:
            for col in schema['categorical']:
                if 'payment' in col.lower():
                    results_df['Payment Method'] = results_df[col]
                    break
            else:
                results_df['Payment Method'] = 'N/A'
        if 'Gender' not in results_df.columns:
            results_df['Gender'] = 'N/A'
        if 'Senior Citizen' not in results_df.columns:
            results_df['Senior Citizen'] = 0
        # Recommendations
        all_recommendations = []
        for i in range(len(results_df)):
            recs = generate_universal_recommendations(
                results_df.iloc[i], float(probabilities[i]), risk_levels[i],
                feature_values=X.iloc[i].values, feature_names=feature_names,
                feature_importances=feature_importances, feature_stats=feature_stats,
            )
            all_recommendations.append(recs)
        summaries = [recs[0] for recs in all_recommendations]
        results_df['Recommendation'] = summaries
        results_df['Retention_Recommendation'] = summaries
        at_risk = [level in ("High Risk", "Medium Risk") for level in risk_levels]
        results_df['Potential_Revenue_Saved'] = np.where(at_risk, results_df['Monthly Charges'].astype(float).round(2), 0.0)
        return {
            'results_df': results_df, 'cleaned_df': df, 'feature_frame': X,
            'probabilities': probabilities, 'model': model, 'metrics': metrics,
            'schema': schema, 'target_column': target_col, 'positive_class': positive_class,
            'feature_names': feature_names, 'feature_stats': feature_stats,
            'feature_importances': feature_importances, 'all_recommendations': all_recommendations,
            'is_universal': True, 'heuristic_mode': False,
        }
    except UniversalChurnError:
        raise
    except Exception as exc:
        logger.exception("Universal churn prediction failed")
        raise UniversalChurnError(f"Universal churn prediction failed: {exc}") from exc

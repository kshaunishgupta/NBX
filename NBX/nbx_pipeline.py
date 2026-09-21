#!/usr/bin/env python3
"""
NBX - Next Best Experience  |  Ooredoo Group
============================================================
6-Stage pipeline (Caner's architecture):
  1. Data Ingestion  – structured (BSS/CRM/billing/device) + unstructured (transcripts/sentiment)
  2. Context Window  – unified per-subscriber object combining all signal types
  3. Campaign Trigger – next-token prediction to select the most likely campaign
  4. LLM Reasoning  – Claude API generates WHY + HOW (rationale + message)
  5. Guard Rails     – margin / consent / network-issue checks
  6. Evaluation      – CSV acceptance + revenue uplift simulation

Set ANTHROPIC_API_KEY to enable the real LLM; otherwise runs an offline stand-in.
"""

import os, re, json, csv
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline as SKPipeline

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
OUT = "outputs"
os.makedirs(OUT, exist_ok=True)

CHANNEL_LIMITS = {"SMS": 160, "Push": 120, "WhatsApp": 300, "Email": 600, "Call": 400}

CAMPAIGNS = [
    "10GB_Aggressive_HVC", "1GB_plus_100mins", "200mins_Upsell",
    "Home_Free_OoredooTV", "Roaming_Daily_Pass", "5G_Unlimited_Upgrade",
    "Retention_Save_25pct", "Data_AddOn_5GB", "Device_Upgrade_iPhone", "Loyalty_Points_2x",
]

ACTION_TYPE = {
    "10GB_Aggressive_HVC": "Retention",  "Retention_Save_25pct": "Retention",
    "1GB_plus_100mins": "NBO",           "Data_AddOn_5GB": "NBO",
    "200mins_Upsell": "Upsell",          "5G_Unlimited_Upgrade": "Upsell",
    "Home_Free_OoredooTV": "CrossSell",  "Roaming_Daily_Pass": "CrossSell",
    "Device_Upgrade_iPhone": "Upsell",   "Loyalty_Points_2x": "Retention",
}

CAMPAIGN_PRICE = {
    "10GB_Aggressive_HVC": 120, "Retention_Save_25pct": 90,
    "1GB_plus_100mins": 40,     "Data_AddOn_5GB": 50,
    "200mins_Upsell": 35,       "5G_Unlimited_Upgrade": 200,
    "Home_Free_OoredooTV": 150, "Roaming_Daily_Pass": 60,
    "Device_Upgrade_iPhone": 280, "Loyalty_Points_2x": 0,
}

# ─────────────────────────────────────────────
# STAGE 1 – DATA GENERATION
# ─────────────────────────────────────────────

def generate_data(n: int = 20_000, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic telecom subscriber data (no real PII)."""
    rng = np.random.default_rng(seed)

    msisdn = [f"9745{rng.integers(1_000_000, 9_999_999)}" for _ in range(n)]

    # Structured signals
    segs    = rng.choice(["HVC", "Mass", "Youth", "Business"], n, p=[.15, .55, .20, .10])
    ltvs    = rng.choice(["HIGH", "MED", "LOW"],               n, p=[.20, .50, .30])
    life    = rng.choice(["Onboarding","Growth","Maturity","Renewal","AtRisk"], n, p=[.10,.30,.30,.15,.15])
    acct    = rng.choice(["Postpaid","Prepaid","Hybrid"],       n, p=[.55,.35,.10])
    lang    = ["EN" if rng.random() < .6 else "AR" for _ in range(n)]
    tenure  = rng.integers(1, 120, n)
    churn   = np.clip(rng.beta(2, 5, n) + (0.3 * (life == "AtRisk")), 0, 1)
    days_r  = np.where(life == "Renewal", rng.integers(1, 60, n), rng.integers(1, 365, n))

    # Usage / billing
    cap     = rng.choice([5, 10, 20, 50, 100], n)
    used    = np.clip(rng.beta(2, 2, n) * cap * 1.2, 0, cap * 1.3)
    dpct    = np.round(np.minimum(used / cap * 100, 130), 1)
    overage = rng.integers(0, 4, n)
    voice   = rng.gamma(2, 120, n).astype(int)
    intl    = np.where(rng.random(n) < .3, rng.gamma(1, 15, n).astype(int), 0)
    roam    = np.where(rng.random(n) < .2, rng.integers(0, 8, n), 0)
    utrend  = rng.choice(["Rising","Stable","Falling"], n, p=[.30,.45,.25])
    top_app = rng.choice(["YouTube","TikTok","Instagram","WhatsApp","Netflix","Gaming","Snapchat"], n)
    pct_5g  = np.round(rng.random(n), 2)
    arpu    = np.round(rng.gamma(3, 20, n) + 40, 1)
    a_trend = rng.choice(["Up","Flat","Down"], n, p=[.30,.40,.30])
    late_p  = rng.binomial(3, 0.15, n)
    outst   = np.where(late_p > 0, np.round(rng.gamma(1, 30, n), 1), 0.0)
    dunning = (late_p >= 2).astype(int)
    pay_m   = rng.choice(["Autopay_Card","Manual_Card","Cash","Bank_Transfer"], n)

    # Device / network
    devs    = rng.choice(["iPhone 14","iPhone 15","Galaxy S23","Galaxy A54","Pixel 8","Huawei P60","Other"], n)
    d_age   = rng.integers(1, 48, n)
    elig    = (d_age >= 20).astype(int)
    l_upg   = rng.integers(1, 60, n)
    d_type  = rng.choice(["Smartphone","Feature Phone","Router/MiFi","Tablet","IoT"], n, p=[.78,.06,.08,.05,.03])

    flagship = np.isin(devs, ["iPhone 15","Galaxy S23","Pixel 8","Huawei P60"])
    mid      = np.isin(devs, ["iPhone 14","Galaxy A54"])
    d_tier   = np.where(flagship, "Flagship", np.where(mid, "Mid", "Budget"))
    net_cap  = np.where(flagship, "5G", np.where(mid, np.where(rng.random(n) < .7, "5G", "4G"), "4G"))
    net_cap  = np.where(np.isin(d_type, ["Feature Phone","IoT"]), "4G", net_cap)

    cong     = np.round(rng.beta(2, 9, n), 2)
    base_dl  = np.where(net_cap == "5G", 180.0, 60.0)
    dl       = np.round(np.clip(base_dl * (1 - 0.6 * cong) * rng.uniform(0.5, 1.2, n), 3, 400), 1)
    lat      = np.clip(np.where(net_cap == "5G", 18, 40) + cong * 120 + rng.normal(0, 8, n), 8, 300).astype(int)
    dcr      = np.round(np.clip(rng.beta(2, 40, n) + cong * 0.05, 0, 0.4), 3)
    cov      = np.round(np.clip(1 - cong * 0.6 - rng.beta(2, 8, n), 0, 1), 2)
    net_comp = rng.binomial(2, 0.04 + cong * 0.15, n)
    net_iss  = ((dcr > 0.12) | (cov < 0.25) | (lat > 160) | (net_comp >= 1) | (dl < 6)).astype(int)

    # Engagement
    channel = rng.choice(["SMS","Push","Email","Call","WhatsApp"], n, p=[.35,.25,.12,.13,.15])
    logins  = rng.poisson(8, n).astype(int)
    consent = rng.binomial(1, 0.85, n)
    loyalty = rng.choice(["None","Silver","Gold","Platinum"], n, p=[.40,.30,.20,.10])
    lcr     = rng.choice(["Accepted","Ignored","Rejected","None"], n, p=[.15,.45,.15,.25])

    # Unstructured: call transcripts + sentiment
    t_neg = [
        "customer complained about slow data speeds at home",
        "customer asked about high bill last month",
        "customer reported dropped calls in their area",
        "customer unhappy after roaming charges abroad",
        "customer threatened to switch to a competitor",
    ]
    t_pos = [
        "customer asked how to add more data",
        "customer inquired about upgrading their phone",
        "customer asked about international calling rates",
        "customer wanted to join the loyalty program",
        "no recent call-center contact",
    ]
    complaint = rng.binomial(1, 0.15 + 0.2 * (churn > 0.6), n).astype(int)
    sentiment = np.where(
        (complaint == 1) | (churn > 0.65),
        rng.choice(["Negative","Neutral"], n, p=[.6,.4]),
        rng.choice(["Neutral","Positive"], n, p=[.6,.4]),
    )
    transcript = np.where(
        (complaint == 1) | (sentiment == "Negative"),
        rng.choice(t_neg, n),
        rng.choice(t_pos, n),
    )
    nps = rng.integers(0, 11, n)

    # NBA signals
    products = rng.choice(CAMPAIGNS, n)
    churn_p  = np.round(churn, 3)
    nbo_alt  = np.array([rng.choice([c for c in CAMPAIGNS if c != p]) for p, in zip(products)])
    nbo_r    = rng.choice([
        "High data usage trend","Renewal window","Churn risk detected",
        "Device upgrade eligible","Loyalty reward due",
    ], n)

    # Acceptance probability (rule-based for training signal)
    bp = np.full(n, 0.12)
    for i in range(n):
        at = ACTION_TYPE.get(products[i], "")
        bp[i] += 0.25 if (at == "Retention" and churn[i] > 0.6) else 0
        bp[i] -= 0.10 if (at == "Upsell" and churn[i] > 0.6) else 0
        bp[i] += 0.18 if (at == "Upsell" and dpct[i] > 80) else 0
        bp[i] += 0.08 if (at == "NBO" and segs[i] == "Mass") else 0
        bp[i] += 0.15 if (life[i] == "Renewal" and at == "Retention") else 0
        bp[i] += 0.05 if ltvs[i] == "HIGH" else 0
        bp[i] += 0.15 if (products[i] == "Device_Upgrade_iPhone" and elig[i]) else 0
        bp[i] -= 0.08 if sentiment[i] == "Negative" else 0
        bp[i] += 0.10 if (utrend[i] == "Rising" and at == "Upsell") else 0
        bp[i] -= 0.05 if consent[i] == 0 else 0
        bp[i] -= 0.10 if (net_iss[i] == 1 and at == "Upsell") else 0
        bp[i] += 0.06 if (net_iss[i] == 1 and at == "Retention") else 0
    accepted = rng.binomial(1, np.clip(bp, 0.01, 0.95), n)

    monthly_val = np.array([CAMPAIGN_PRICE[p] for p in products], dtype=float)

    df = pd.DataFrame({
        "MSISDN": msisdn, "ProductID": products, "ActionType": [ACTION_TYPE[p] for p in products],
        "MonthlyValueQAR": monthly_val, "Segment": segs, "LTV_Band": ltvs, "Lifecycle": life,
        "AccountType": acct, "Language": lang, "Tenure_M": tenure, "Churn_Prob": churn_p,
        "Days_To_Renewal": days_r, "Data_Cap_GB": cap, "Data_Used_GB": np.round(used, 1),
        "Data_Usage_Pct": dpct, "Overage_Months": overage, "Voice_Min": voice,
        "Intl_Min": intl, "Roaming_Days": roam, "Usage_Trend": utrend, "Top_App": top_app,
        "Pct_5G": pct_5g, "ARPU_QAR": arpu, "ARPU_Trend": a_trend, "Late_Payments": late_p,
        "Outstanding_QAR": outst, "Dunning_Flag": dunning, "Payment_Method": pay_m,
        "Device_Model": devs, "Device_Type": d_type, "Device_Tier": d_tier,
        "Network_Capability": net_cap, "Device_Age_M": d_age, "Upgrade_Eligible": elig,
        "Last_Upgrade_M": l_upg, "Avg_DL_Speed_Mbps": dl, "Dropped_Call_Rate": dcr,
        "Latency_ms": lat, "Coverage_Score": cov, "Cell_Congestion": cong,
        "Network_Complaints_90d": net_comp, "Network_Issue_Flag": net_iss,
        "Channel_Affinity": channel, "App_Logins_30d": logins, "Consent_Flag": consent,
        "Loyalty_Tier": loyalty, "Last_Campaign_Resp": lcr, "Recent_Complaint": complaint,
        "Sentiment": sentiment, "NPS": nps, "Call_Transcript": transcript,
        "NBO_Alt_Action": nbo_alt, "NBO_Reason": nbo_r, "Accept_Ind": accepted,
    })
    return df


# ─────────────────────────────────────────────
# STAGE 2 – SUBSCRIBER CONTEXT WINDOW
# ─────────────────────────────────────────────

SOURCE_MAP = {
    "BSS/CRM":     ["AccountType","Language","Tenure_M","Segment","LTV_Band","Lifecycle","Days_To_Renewal","Consent_Flag"],
    "Billing":     ["ARPU_QAR","ARPU_Trend","Late_Payments","Outstanding_QAR","Dunning_Flag","Payment_Method","MonthlyValueQAR"],
    "Consumption": ["Data_Cap_GB","Data_Used_GB","Data_Usage_Pct","Overage_Months","Voice_Min","Intl_Min","Roaming_Days","Usage_Trend","Top_App","Pct_5G"],
    "Device":      ["Device_Model","Device_Type","Device_Tier","Network_Capability","Device_Age_M","Upgrade_Eligible","Last_Upgrade_M","Avg_DL_Speed_Mbps","Dropped_Call_Rate","Latency_ms","Coverage_Score","Cell_Congestion","Network_Complaints_90d","Network_Issue_Flag"],
    "Engagement":  ["Channel_Affinity","App_Logins_30d","Loyalty_Tier","Last_Campaign_Resp"],
    "Care":        ["Recent_Complaint","Sentiment","NPS","Call_Transcript"],
    "NBA/NBO":     ["ProductID","ActionType","Churn_Prob","NBO_Alt_Action","NBO_Reason"],
}

# ══════════════════════════════════════════════════════════════════════════════
# VECTOR RETRIEVAL LAYER  (optional — see pipeline/vector_store.py)
# ══════════════════════════════════════════════════════════════════════════════
# OFF by default: build_context_window() behaves exactly as before unless you
# call enable_vector_retrieval(...) first. When ON, the single Call_Transcript
# is replaced by the top-k semantically most relevant interactions retrieved
# from that subscriber's embedded history.
_VSTORE = None
_VHISTORY = {}
_VLOG = False
_VTOPK = 3

def enable_vector_retrieval(subscriber_ids, top_k=3, log=False, n_min=5, n_max=10,
                            transcripts=None):
    """
    Embed synthetic interaction histories for these subscribers and switch
    build_context_window() into retrieval mode. Returns the VectorStore.

    transcripts: optional {msisdn: Call_Transcript}. Including it puts the
        subscriber's REAL care transcript into the index so it competes for a
        top-k slot instead of being silently discarded.
    """
    global _VSTORE, _VHISTORY, _VLOG, _VTOPK
    import sys as _sys
    _sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from data.interaction_history import generate_all_histories
    from pipeline.vector_store import build_vector_store

    ids = [str(s) for s in subscriber_ids]
    corpus = generate_all_histories(ids, min_interactions=n_min, max_interactions=n_max)
    if transcripts:
        for cid, text in transcripts.items():
            text = str(text).strip()
            if not text or text.lower() == "no recent call-center contact":
                continue
            corpus.append({
                "interaction_id": f"{cid}-transcript",
                "customer_id": str(cid),
                "date": "(current record)",
                "channel": "call",
                "text": text,
                "topic": "existing_transcript",
                "is_relevant": True,
            })
    _VHISTORY = {}
    for i in corpus:
        _VHISTORY.setdefault(i["customer_id"], []).append(i)
    _VSTORE = build_vector_store(corpus)
    _VLOG, _VTOPK = log, top_k
    return _VSTORE

def vector_retrieval_active() -> bool:
    """True once enable_vector_retrieval() has been called (used by the UI)."""
    return _VSTORE is not None

def _retrieve_unstructured(msisdn):
    """Return (redacted_snippets, hit_records); (None, []) when retrieval is off."""
    if _VSTORE is None or msisdn is None:
        return None, []
    hits = _VSTORE.retrieve(str(msisdn), top_k=_VTOPK)
    if _VLOG:
        kept = {h["interaction_id"] for h in hits}
        history = _VHISTORY.get(str(msisdn), [])
        print(f"\n   [retrieval] {msisdn}: {len(hits)} of {len(history)} interactions used")
        for h in hits:
            print(f"      KEPT    sim={h['similarity']:+.3f}  {h['date']}  {h['channel']:<9} {h['text']}")
        for s in history:
            if s["interaction_id"] not in kept:
                print(f"      dropped              {s['date']}  {s['channel']:<9} {s['text']}")
    return [_redact(h["text"]) for h in hits], hits


def build_context_window(subscriber_row: pd.Series, bandit_rec: pd.Series) -> dict:
    """
    Unified subscriber context object — Stage 2 of the pipeline.

    If the vector store is active, the unstructured slice of this context is
    built by semantic retrieval instead of dumping the whole transcript.
    """
    ctx = _build_context_window_base(subscriber_row, bandit_rec)
    msisdn = getattr(bandit_rec, "MSISDN", None) or getattr(subscriber_row, "name", None)
    snippets, hits = _retrieve_unstructured(msisdn)
    if snippets:
        ctx["transcript"] = snippets[0]          # most relevant, not a raw dump
        ctx["retrieved_interactions"] = [
            {"text": _redact(h["text"]), "date": h["date"],
             "channel": h["channel"], "similarity": h["similarity"]}
            for h in hits
        ]
        ctx["retrieval_mode"] = f"vector_top{len(hits)}"
    else:
        ctx["retrieval_mode"] = "full_dump"
    return ctx


def _build_context_window_base(subscriber_row: pd.Series, bandit_rec: pd.Series) -> dict:
    return {
        # selected action from bandit
        "action":       bandit_rec.Selected_Action,
        "action_type":  bandit_rec.ActionType,
        "propensity":   bandit_rec.Propensity,
        "value_qar":    bandit_rec.MonthlyValueQAR,
        # BSS/CRM
        "segment":      subscriber_row.Segment,
        "ltv":          subscriber_row.LTV_Band,
        "lifecycle":    subscriber_row.Lifecycle,
        "account_type": subscriber_row.AccountType,
        "language":     subscriber_row.Language,
        "tenure_m":     int(subscriber_row.Tenure_M),
        "churn":        float(subscriber_row.Churn_Prob),
        "days_renewal": int(subscriber_row.Days_To_Renewal),
        "consent":      int(subscriber_row.Consent_Flag),
        # Billing
        "arpu":         float(subscriber_row.ARPU_QAR),
        "arpu_trend":   subscriber_row.ARPU_Trend,
        "late_payments":int(subscriber_row.Late_Payments),
        "outstanding":  float(subscriber_row.Outstanding_QAR),
        "dunning":      int(subscriber_row.Dunning_Flag),
        # Consumption
        "data_cap_gb":  int(subscriber_row.Data_Cap_GB),
        "data_used_gb": float(subscriber_row.Data_Used_GB),
        "data_pct":     float(subscriber_row.Data_Usage_Pct),
        "overage":      int(subscriber_row.Overage_Months),
        "voice_min":    int(subscriber_row.Voice_Min),
        "intl_min":     int(subscriber_row.Intl_Min),
        "roaming_days": int(subscriber_row.Roaming_Days),
        "usage_trend":  subscriber_row.Usage_Trend,
        "top_app":      subscriber_row.Top_App,
        "pct_5g":       float(subscriber_row.Pct_5G),
        # Device/Network
        "device":       subscriber_row.Device_Model,
        "device_type":  subscriber_row.Device_Type,
        "device_tier":  subscriber_row.Device_Tier,
        "net_cap":      subscriber_row.Network_Capability,
        "device_age":   int(subscriber_row.Device_Age_M),
        "upgrade_eligible": int(subscriber_row.Upgrade_Eligible),
        "last_upgrade_m":   int(subscriber_row.Last_Upgrade_M),
        "dl_speed":     float(subscriber_row.Avg_DL_Speed_Mbps),
        "dropped_call_rate": float(subscriber_row.Dropped_Call_Rate),
        "latency_ms":   int(subscriber_row.Latency_ms),
        "coverage":     float(subscriber_row.Coverage_Score),
        "congestion":   float(subscriber_row.Cell_Congestion),
        "net_complaints": int(subscriber_row.Network_Complaints_90d),
        "net_issue":    int(subscriber_row.Network_Issue_Flag),
        # Engagement
        "channel":      subscriber_row.Channel_Affinity,
        "app_logins":   int(subscriber_row.App_Logins_30d),
        "loyalty":      subscriber_row.Loyalty_Tier,
        "last_campaign_resp": subscriber_row.Last_Campaign_Resp,
        # Care / unstructured
        "complaint":    int(subscriber_row.Recent_Complaint),
        "sentiment":    subscriber_row.Sentiment,
        "nps":          int(subscriber_row.NPS),
        "transcript":   _redact(str(subscriber_row.Call_Transcript)),
        # NBO
        "nbo_reason":   subscriber_row.NBO_Reason,
    }

def _redact(text: str) -> str:
    return re.sub(r"\b\d{6,}\b", "[REDACTED]", text)


# ─────────────────────────────────────────────
# STAGE 3 – CAMPAIGN TRIGGER (next-token prediction)
# ─────────────────────────────────────────────

CAT_FEATS = [
    "ProductID","Segment","LTV_Band","Lifecycle","AccountType",
    "Usage_Trend","ARPU_Trend","Device_Model","Device_Type","Device_Tier",
    "Network_Capability","Channel_Affinity","Loyalty_Tier","Last_Campaign_Resp",
    "Top_App","Sentiment",
]
NUM_FEATS = [
    "MonthlyValueQAR","Tenure_M","Churn_Prob","Days_To_Renewal","Data_Cap_GB",
    "Data_Used_GB","Data_Usage_Pct","Overage_Months","Voice_Min","Intl_Min",
    "Roaming_Days","Pct_5G","ARPU_QAR","Late_Payments","Outstanding_QAR","Dunning_Flag",
    "Device_Age_M","Upgrade_Eligible","Last_Upgrade_M","Avg_DL_Speed_Mbps",
    "Dropped_Call_Rate","Latency_ms","Coverage_Score","Cell_Congestion",
    "Network_Complaints_90d","Network_Issue_Flag","App_Logins_30d","Consent_Flag","NPS",
]

class CampaignTrigger:
    """
    Next-token prediction layer: for a given subscriber behavioural sequence,
    predict which campaign they are most likely to respond to next.
    Implemented as a gradient-boosted propensity model; each campaign is a
    'token' in the subscriber's engagement sequence.
    """
    def __init__(self):
        pre = ColumnTransformer(
            [("cat", OneHotEncoder(handle_unknown="ignore"), CAT_FEATS)],
            remainder="passthrough",
        )
        self.model = SKPipeline([("pre", pre), ("clf", GradientBoostingClassifier(n_estimators=100, max_depth=3))])
        self.campaigns = CAMPAIGNS
        self.price     = CAMPAIGN_PRICE
        self.atype     = ACTION_TYPE

    def fit(self, df: pd.DataFrame) -> "CampaignTrigger":
        self.model.fit(df[CAT_FEATS + NUM_FEATS], df["Accept_Ind"])
        return self

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """Score every campaign for every subscriber; return the best one."""
        rows_out = []
        for _, row in df.iterrows():
            best_score, best_camp = -1.0, self.campaigns[0]
            for camp in self.campaigns:
                candidate = row.copy()
                candidate["ProductID"]       = camp
                candidate["MonthlyValueQAR"] = self.price[camp]
                candidate["ActionType"]      = self.atype[camp]
                score = float(self.model.predict_proba(
                    pd.DataFrame([candidate])[CAT_FEATS + NUM_FEATS]
                )[0, 1])
                if score > best_score:
                    best_score, best_camp = score, camp
            rows_out.append({
                "MSISDN":          row.MSISDN,
                "Selected_Action": best_camp,
                "ActionType":      self.atype[best_camp],
                "Propensity":      round(best_score, 4),
                "MonthlyValueQAR": self.price[best_camp],
            })
        return pd.DataFrame(rows_out)


# ─────────────────────────────────────────────
# STAGE 4 – LLM REASONING  (Claude API)
# ─────────────────────────────────────────────

SYSTEM_PROMPT = """You are NBX, a telecom Next-Best-Experience copywriter for Ooredoo Group.
You receive a SELECTED CAMPAIGN (already chosen by the recommender) and a rich SUBSCRIBER CONTEXT
(BSS/CRM, billing, consumption, device/network, engagement, care/sentiment, NBO reasoning).

HARD RULES:
1. Do NOT change or re-rank the campaign; use it exactly as given.
2. Only use facts in the context. Never invent numbers, savings, dates or features.
3. Every numeric claim must match the context exactly.
4. If churn >= 0.6 or sentiment is Negative, lead with empathy/retention, never a hard upsell.
5. Reflect the recent care interaction (transcript/sentiment) where relevant, tactfully.
6. If net_issue is 1, acknowledge the experience issue first; prefer retention/loyalty actions.
7. Respect channel limits: SMS<=160 chars, Push<=120, WhatsApp<=300, Email longer, Call=agent script.
8. Write in the subscriber's language (EN/AR) per the context.

Return ONLY valid JSON:
{
  "rationale": "<1-2 sentence WHY this campaign fits this subscriber>",
  "message": "<customer-facing message — channel-length compliant>",
  "grounded": true,
  "used_fields": ["list", "of", "context", "fields", "cited"]
}"""

def _make_user_prompt(ctx: dict) -> str:
    return (
        f"SELECTED CAMPAIGN (do not change): {ctx['action']} | type={ctx['action_type']} | value={ctx['value_qar']} QAR\n"
        f"CHANNEL: {ctx['channel']}\n"
        f"NBO_REASON: {ctx['nbo_reason']}\n\n"
        f"SUBSCRIBER CONTEXT (only use these facts):\n"
        + json.dumps(ctx, indent=2, default=str)
    )

def _validate_grounding(ctx: dict, result: dict) -> tuple[bool, str]:
    text = (result.get("rationale", "") + " " + result.get("message", "")).lower()
    allowed = {
        str(int(ctx["overage"])), str(int(ctx["days_renewal"])),
        str(int(round(ctx["data_pct"]))), str(int(ctx["value_qar"])),
        str(int(round(ctx["data_used_gb"]))), str(int(ctx["data_cap_gb"])),
        str(int(round(ctx["arpu"]))), str(int(ctx["voice_min"])),
        str(int(ctx["nps"])), str(int(ctx["tenure_m"])),
        str(int(round(ctx["dl_speed"]))), str(int(ctx["latency_ms"])),
        str(int(ctx["net_complaints"])),
    }
    for raw in re.findall(r"\b\d{1,5}\b", text):
        n = raw.lstrip("0") or "0"
        if n in allowed:
            continue
        if any(abs(int(n) - int(a)) <= 1 for a in allowed if a.isdigit()):
            continue
        return False, f"ungrounded number '{raw}'"
    return True, "ok"

def _offline_rationale(ctx: dict) -> str:
    parts = []
    if ctx["net_issue"]:
        parts.append(f"is experiencing network issues (dropped-call rate {ctx['dropped_call_rate']:.0%})")
    if ctx["sentiment"] == "Negative" or ctx["complaint"]:
        parts.append("recently contacted care: \"" + ctx['transcript'] + "\"")
    if ctx["overage"] >= 1:
        parts.append(f"exceeded data cap in {ctx['overage']} of the last 3 months")
    if ctx["data_pct"] >= 80:
        parts.append(f"is a heavy data user ({ctx['data_pct']:.0f}% of {ctx['data_cap_gb']}GB)")
    if ctx["usage_trend"] == "Rising":
        parts.append("has rising data usage")
    if ctx["days_renewal"] <= 60:
        parts.append(f"is {ctx['days_renewal']} days from renewal")
    if ctx["upgrade_eligible"]:
        parts.append(f"is eligible for a device upgrade ({ctx['device']}, {ctx['device_age']}m old)")
    if ctx["churn"] >= 0.6:
        parts.append("shows elevated churn risk")
    if ctx["loyalty"] in ("Gold", "Platinum"):
        parts.append(f"is a {ctx['loyalty']} loyalty member")
    if not parts:
        parts.append(f"is an active {ctx['segment']} customer (tenure {ctx['tenure_m']}m)")
    return "Subscriber " + ", ".join(parts) + "."

def _offline_message(ctx: dict, why: str) -> str:
    a, ch = ctx["action"], ctx["channel"]
    empath = ctx["sentiment"] == "Negative" or ctx["churn"] >= 0.6 or ctx["net_issue"]
    a_nice = a.replace("_", " ")
    if ctx["net_issue"]:
        core = f"We noticed your service hasn't been at its best lately — to make it right, our {a_nice} is ready for you"
    elif empath:
        core = f"Thanks for being with us — to make things right, our {a_nice} is ready for you"
    elif ctx["action_type"] == "Upsell":
        if ctx["upgrade_eligible"] and "Device" in a:
            core = f"You're eligible to upgrade your {ctx['device']} — {a_nice} is ready"
        else:
            core = f"You're using {ctx['data_pct']:.0f}% of your {ctx['data_cap_gb']}GB — {a_nice} gives you more headroom"
    elif ctx["action_type"] == "Retention":
        core = f"Based on your usage, our {a_nice} keeps you covered"
        if ctx["days_renewal"] <= 60:
            core += f", and your renewal is in {ctx['days_renewal']} days"
    elif ctx["action_type"] == "CrossSell":
        core = f"Add {a_nice} to get more from your plan"
    else:
        core = f"{a_nice} suits your usage"

    templates = {
        "SMS":      f"Hi! {core}. Reply YES. Ooredoo",
        "Push":     f"{core} — tap to activate.",
        "WhatsApp": f"Hi! {core}. Tap below to confirm. Ooredoo",
        "Email":    f"Dear Customer,\n\n{why}\n\n{core}.\n\nOoredoo",
        "Call":     f"AGENT: {why} Recommend: {core}.",
    }
    return templates[ch][: CHANNEL_LIMITS[ch]]

def generate_nbx(ctx: dict) -> tuple[str, str, str]:
    """Call Claude API (or offline stand-in) to generate WHY + HOW."""
    api_key = os.getenv("ANTHROPIC_API_KEY", "")
    if api_key:
        try:
            import anthropic
            client = anthropic.Anthropic(api_key=api_key)
            msg = client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=512,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": _make_user_prompt(ctx)}],
            )
            raw = json.loads(msg.content[0].text)
            ok, _ = _validate_grounding(ctx, raw)
            message   = (raw.get("message") or "").strip()[: CHANNEL_LIMITS[ctx["channel"]]]
            rationale = (raw.get("rationale") or "").strip()
            if ok and message and rationale:
                return rationale, message, "claude"
        except Exception:
            pass
    why = _offline_rationale(ctx)
    return why, _offline_message(ctx, why), "offline"


# ─────────────────────────────────────────────
# STAGE 5 – GUARD RAILS
# ─────────────────────────────────────────────

def guard_rail(ctx: dict) -> str:
    """Margin / consent / network-issue guard."""
    if ctx["consent"] == 0:
        return "BLOCKED_NO_CONSENT"
    if ctx["action_type"] == "Retention" and ctx["ltv"] == "LOW" and ctx["churn"] >= 0.6:
        return "SOFTENED"
    if ctx["late_payments"] >= 2 and ctx["value_qar"] > 150:
        return "REVIEW_CREDIT"
    if ctx["net_issue"] and ctx["action_type"] == "Upsell":
        return "NETWORK_HOLD"
    return "APPROVED"


# ─────────────────────────────────────────────
# STAGE 6 – EVALUATION (CSV)
# ─────────────────────────────────────────────

def evaluate(rec_df: pd.DataFrame, nbx_df: pd.DataFrame, base_df: pd.DataFrame) -> dict:
    """Simulate acceptance uplift and revenue impact; write CSV."""
    d = base_df.drop_duplicates("MSISDN")
    keep = ["MSISDN","Churn_Prob","Data_Usage_Pct","Overage_Months","Days_To_Renewal",
            "LTV_Band","Usage_Trend","Upgrade_Eligible","Sentiment","Loyalty_Tier",
            "Consent_Flag","Network_Issue_Flag","Device_Type"]
    m = rec_df.merge(d[keep], on="MSISDN").merge(
        nbx_df[["msisdn","guard_status"]], left_on="MSISDN", right_on="msisdn"
    )
    m["p_baseline"] = m["Propensity"] * 0.85

    def lift(r):
        x = 1.10
        if r.Overage_Months >= 1 or r.Data_Usage_Pct >= 80: x += 0.12
        if r.Days_To_Renewal <= 60:                          x += 0.10
        if r.ActionType == "Retention" and r.Churn_Prob >= 0.6: x += 0.10
        if r.Usage_Trend == "Rising" and r.ActionType == "Upsell": x += 0.06
        if r.Upgrade_Eligible == 1 and "Device" in r.Selected_Action: x += 0.08
        if r.Sentiment == "Negative":                        x += 0.06
        if r.Network_Issue_Flag == 1 and r.ActionType in ("Retention","CrossSell"): x += 0.07
        if r.Network_Issue_Flag == 1 and r.ActionType == "Upsell": x -= 0.04
        if r.Loyalty_Tier in ("Gold","Platinum"):            x += 0.04
        x += 0.05  # NBX message quality
        if r.guard_status in ("SOFTENED","REVIEW_CREDIT"):  x -= 0.05
        if r.Consent_Flag == 0:                              x  = 1.0
        return min(x, 1.55)

    m["p_nbx"] = np.clip(m["p_baseline"] * m.apply(lift, axis=1), 0, 0.95)
    months = 12
    base_rev  = (m["p_baseline"] * m["MonthlyValueQAR"] * months).sum()
    nbx_rev   = (m["p_nbx"]      * m["MonthlyValueQAR"] * months).sum()
    n = len(m)

    m.to_csv(f"{OUT}/eval_detail.csv", index=False)
    summary = {
        "Acceptance_baseline": f"{m['p_baseline'].mean():.1%}",
        "Acceptance_NBX":      f"{m['p_nbx'].mean():.1%}",
        "Revenue_uplift_pct":  f"+{nbx_rev/base_rev - 1:.1%}",
        "Incr_annual_QAR_sample": f"{nbx_rev - base_rev:,.0f}",
        "Extrap_500k_QAR":     f"{(nbx_rev - base_rev) * (500_000 / n):,.0f}",
        "Extrap_2M_QAR":       f"{(nbx_rev - base_rev) * (2_000_000 / n):,.0f}",
    }
    pd.DataFrame(list(summary.items()), columns=["Metric","Value"]).to_csv(f"{OUT}/eval_summary.csv", index=False)
    return summary


# ─────────────────────────────────────────────
# ORCHESTRATOR
# ─────────────────────────────────────────────

def run_pipeline(n_rows: int = 20_000, n_score: int = 800) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """Execute all 6 stages end-to-end."""
    print("=" * 60)
    print("NBX PIPELINE  |  Ooredoo Group")
    print("=" * 60)

    # Stage 1 – Data
    print(f"[Stage 1] Generating {n_rows:,} subscriber records …")
    df = generate_data(n_rows)
    df.to_csv(f"{OUT}/subscribers.csv", index=False)
    print(f"          {df.shape[0]:,} rows × {df.shape[1]} fields | accept rate {df['Accept_Ind'].mean():.1%}")

    # Stage 3 – Campaign Trigger (fit on full dataset, score on sample)
    print("[Stage 3] Training campaign-trigger model …")
    trigger = CampaignTrigger().fit(df)
    score_df = df.drop_duplicates("MSISDN").head(n_score).reset_index(drop=True)
    rec_df = trigger.predict(score_df)
    rec_df.to_csv(f"{OUT}/campaign_triggers.csv", index=False)
    print(f"          Scored {len(rec_df):,} subscribers")

    # Stages 2 + 4 + 5 – Context window → LLM → Guard
    print(f"[Stages 2+4+5] Building context windows + LLM generation …")
    api_mode = "Claude" if os.getenv("ANTHROPIC_API_KEY") else "offline stand-in"
    print(f"          LLM mode: {api_mode}")
    cust = score_df.set_index("MSISDN")
    nbx_rows = []
    for _, b in rec_df.iterrows():
        ctx = build_context_window(cust.loc[b.MSISDN], b)     # Stage 2
        why, msg, src = generate_nbx(ctx)                      # Stage 4
        guard = guard_rail(ctx)                                 # Stage 5
        nbx_rows.append({
            "msisdn":         b.MSISDN,
            "action":         ctx["action"],
            "action_type":    ctx["action_type"],
            "channel":        ctx["channel"],
            "sentiment":      ctx["sentiment"],
            "rationale_WHY":  why,
            "message_HOW":    msg,
            "guard_status":   guard,
            "gen_source":     src,
        })
    nbx_df = pd.DataFrame(nbx_rows)
    nbx_df.to_csv(f"{OUT}/nbx_output.csv", index=False)
    print(f"          Sample WHY: {nbx_df.iloc[0]['rationale_WHY'][:80]}…")
    print(f"          Sample HOW: {nbx_df.iloc[0]['message_HOW'][:80]}…")

    # Stage 6 – Evaluation
    print("[Stage 6] Running CSV evaluation …")
    summary = evaluate(rec_df, nbx_df, df)
    print(f"          Acceptance: {summary['Acceptance_baseline']} → {summary['Acceptance_NBX']}")
    print(f"          Revenue uplift: {summary['Revenue_uplift_pct']}")
    print(f"          Incremental annual QAR (sample): {summary['Incr_annual_QAR_sample']}")
    print("=" * 60)
    print(f"Outputs in ./{OUT}/")

    return df, rec_df, nbx_df, summary


if __name__ == "__main__":
    run_pipeline()

"""
NBX Demo UI  —  Streamlit
Run:  streamlit run app.py
"""

import os, json, time
import streamlit as st
import pandas as pd
import numpy as np

# ── page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NBX · Ooredoo Group",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── lazy imports from pipeline ─────────────────────────────────────────────────
@st.cache_resource(show_spinner="Generating subscriber data & training model…")
def load_pipeline():
    from nbx_pipeline import generate_data, CampaignTrigger
    df = generate_data(5_000)
    trigger = CampaignTrigger().fit(df)
    return df, trigger

# ── sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image(
        "https://upload.wikimedia.org/wikipedia/commons/thumb/6/6e/Ooredoo_Logo.svg/200px-Ooredoo_Logo.svg.png",
        width=140,
    )
    st.markdown("### NBX Configuration")
    api_key = st.text_input(
        "Anthropic API Key (optional)",
        type="password",
        value=os.getenv("ANTHROPIC_API_KEY", ""),
        help="Leave blank to use the offline stand-in."
    )
    if api_key:
        os.environ["ANTHROPIC_API_KEY"] = api_key

    st.markdown("---")
    n_score = st.slider("Subscribers to score", 10, 500, 100, 10)
    st.markdown("---")
    st.caption("Stages executed on Run:\n1 Data · 2 Context Window\n3 Campaign Trigger · 4 LLM\n5 Guard Rails · 6 Evaluation")

# ── header ─────────────────────────────────────────────────────────────────────
st.title("📡 NBX — Next Best Experience")
st.caption("LLM-powered customer recommendation engine · Ooredoo Group · prototype")

tab_run, tab_inspect, tab_eval, tab_arch = st.tabs(
    ["▶ Run Pipeline", "🔍 Inspect Subscriber", "📊 Evaluation", "🏗 Architecture"]
)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 – RUN PIPELINE
# ══════════════════════════════════════════════════════════════════════════════
with tab_run:
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("""
        Click **Run** to execute all 6 stages of the NBX pipeline on a sample of subscribers.
        Results are written to `outputs/` and shown in the other tabs.
        """)
    with col2:
        run_btn = st.button("🚀 Run Pipeline", type="primary", use_container_width=True)

    if run_btn:
        from nbx_pipeline import (
            generate_data, CampaignTrigger, build_context_window,
            generate_nbx, guard_rail, evaluate, OUT,
        )
        os.makedirs(OUT, exist_ok=True)

        prog = st.progress(0, "Stage 1 — Generating subscriber data…")
        status = st.status("Running pipeline…", expanded=True)
        with status:
            # Stage 1
            st.write("**Stage 1** · Generating subscriber data (structured + unstructured)…")
            df = generate_data(5_000)
            df.to_csv(f"{OUT}/subscribers.csv", index=False)
            prog.progress(16, "Stage 1 ✅")
            st.write(f"  → {len(df):,} subscriber records · {df.shape[1]} fields · accept rate {df['Accept_Ind'].mean():.1%}")

            # Stage 3 (trigger trained on full data)
            st.write("**Stage 3** · Training campaign-trigger (next-token prediction)…")
            trigger = CampaignTrigger().fit(df)
            score_df = df.drop_duplicates("MSISDN").head(n_score).reset_index(drop=True)
            rec_df = trigger.predict(score_df)
            rec_df.to_csv(f"{OUT}/campaign_triggers.csv", index=False)
            prog.progress(50, "Stage 3 ✅")
            st.write(f"  → Scored {len(rec_df):,} subscribers")

            # Stages 2 + 4 + 5
            st.write("**Stages 2 · 4 · 5** · Context window → LLM → Guard rails…")
            mode = "Claude" if os.getenv("ANTHROPIC_API_KEY") else "offline stand-in"
            st.write(f"  → LLM mode: **{mode}**")
            cust = score_df.set_index("MSISDN")
            nbx_rows = []
            bar = st.progress(0)
            for idx, (_, b) in enumerate(rec_df.iterrows()):
                ctx = build_context_window(cust.loc[b.MSISDN], b)
                why, msg, src = generate_nbx(ctx)
                grd = guard_rail(ctx)
                nbx_rows.append({
                    "msisdn": b.MSISDN, "action": ctx["action"],
                    "action_type": ctx["action_type"], "channel": ctx["channel"],
                    "sentiment": ctx["sentiment"], "rationale_WHY": why,
                    "message_HOW": msg, "guard_status": grd, "gen_source": src,
                })
                bar.progress((idx + 1) / len(rec_df))
            nbx_df = pd.DataFrame(nbx_rows)
            nbx_df.to_csv(f"{OUT}/nbx_output.csv", index=False)
            prog.progress(83, "Stage 5 ✅")

            # Stage 6
            st.write("**Stage 6** · CSV evaluation loop…")
            summary = evaluate(rec_df, nbx_df, df)
            prog.progress(100, "All stages complete ✅")
            status.update(label="Pipeline complete ✅", state="complete")

        # persist to session for other tabs
        st.session_state["df"]      = df
        st.session_state["rec_df"]  = rec_df
        st.session_state["nbx_df"]  = nbx_df
        st.session_state["summary"] = summary

        # headline metrics
        st.markdown("### Results")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Acceptance (baseline)", summary["Acceptance_baseline"])
        m2.metric("Acceptance (NBX)",      summary["Acceptance_NBX"],
                  delta=summary["Revenue_uplift_pct"])
        m3.metric("Revenue uplift",        summary["Revenue_uplift_pct"])
        m4.metric("Incr. annual QAR",      summary["Incr_annual_QAR_sample"])

        st.markdown("#### Sample NBX outputs")
        st.dataframe(
            nbx_df[["msisdn","action","action_type","channel","sentiment",
                     "rationale_WHY","message_HOW","guard_status","gen_source"]].head(20),
            use_container_width=True,
        )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 – INSPECT SUBSCRIBER
# ══════════════════════════════════════════════════════════════════════════════
with tab_inspect:
    st.markdown("### Single-subscriber deep dive")

    # Load cached outputs if available
    nbx_path = f"outputs/nbx_output.csv"
    sub_path  = f"outputs/subscribers.csv"
    rec_path  = f"outputs/campaign_triggers.csv"

    if "nbx_df" in st.session_state:
        nbx_df  = st.session_state["nbx_df"]
        df      = st.session_state["df"]
        rec_df  = st.session_state["rec_df"]
    elif os.path.exists(nbx_path):
        nbx_df  = pd.read_csv(nbx_path)
        df      = pd.read_csv(sub_path)
        rec_df  = pd.read_csv(rec_path)
    else:
        st.info("Run the pipeline first (▶ Run Pipeline tab).")
        st.stop()

    msisdn_list = nbx_df["msisdn"].tolist()
    selected = st.selectbox("Select subscriber MSISDN", msisdn_list)

    row_nbx = nbx_df[nbx_df["msisdn"] == selected].iloc[0]
    row_sub = df[df["MSISDN"] == selected].iloc[0]
    row_rec = rec_df[rec_df["MSISDN"] == selected].iloc[0]

    from nbx_pipeline import build_context_window
    ctx = build_context_window(row_sub, row_rec)

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("#### Subscriber Context Window (Stage 2)")
        source_labels = {
            "BSS/CRM":     ["segment","ltv","lifecycle","account_type","language","tenure_m","days_renewal","consent"],
            "Billing":     ["arpu","arpu_trend","late_payments","outstanding","dunning"],
            "Consumption": ["data_cap_gb","data_used_gb","data_pct","overage","voice_min","usage_trend","top_app"],
            "Device":      ["device","device_tier","net_cap","device_age","upgrade_eligible","dl_speed","net_issue"],
            "Engagement":  ["channel","app_logins","loyalty","last_campaign_resp"],
            "Care":        ["complaint","sentiment","nps","transcript"],
            "NBA/NBO":     ["action","action_type","propensity","nbo_reason"],
        }
        for src, fields in source_labels.items():
            with st.expander(src, expanded=(src == "Care")):
                rows = {k: ctx.get(k, "—") for k in fields if k in ctx}
                st.json(rows)

    with col_b:
        st.markdown("#### NBX Output (Stages 4 + 5)")
        guard_color = {"APPROVED": "green", "SOFTENED": "orange",
                       "REVIEW_CREDIT": "orange", "BLOCKED_NO_CONSENT": "red",
                       "NETWORK_HOLD": "red"}.get(row_nbx["guard_status"], "grey")
        st.markdown(f"**Guard status:** :{guard_color}[{row_nbx['guard_status']}]")
        st.markdown(f"**Campaign:** `{row_nbx['action']}` ({row_nbx['action_type']})")
        st.markdown(f"**Channel:** {row_nbx['channel']}")
        st.markdown(f"**Gen source:** `{row_nbx['gen_source']}`")

        st.markdown("**WHY — Rationale**")
        st.info(row_nbx["rationale_WHY"])

        st.markdown("**HOW — Customer Message**")
        st.success(row_nbx["message_HOW"])

        # Campaign trigger chart
        st.markdown("#### Campaign propensity ranking (Stage 3)")
        if hasattr(st.session_state.get("trigger", None), "predict"):
            trigger = st.session_state["trigger"]
        else:
            from nbx_pipeline import CampaignTrigger, CAMPAIGNS, CAMPAIGN_PRICE, ACTION_TYPE
            scores = {}
            for camp in CAMPAIGNS:
                cand = row_sub.copy()
                cand["ProductID"]       = camp
                cand["MonthlyValueQAR"] = CAMPAIGN_PRICE[camp]
                cand["ActionType"]      = ACTION_TYPE[camp]
                # simple offline score using churn / data pct heuristics
                base = 0.15
                at = ACTION_TYPE[camp]
                if at == "Retention" and row_sub.Churn_Prob >= 0.6: base += 0.25
                if at == "Upsell" and row_sub.Data_Usage_Pct >= 80:  base += 0.18
                if row_sub.Lifecycle == "Renewal" and at == "Retention": base += 0.15
                if camp == "Device_Upgrade_iPhone" and row_sub.Upgrade_Eligible: base += 0.15
                scores[camp] = round(min(base + np.random.default_rng(abs(hash(selected)) % (2**31)).uniform(0, 0.1), 0.95), 3)
            chart = pd.DataFrame({"Campaign": list(scores.keys()), "Score": list(scores.values())})
            chart = chart.sort_values("Score", ascending=True)
            st.bar_chart(chart.set_index("Campaign")["Score"])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 – EVALUATION
# ══════════════════════════════════════════════════════════════════════════════
with tab_eval:
    st.markdown("### Stage 6 · CSV Evaluation Loop")

    eval_path = "outputs/eval_summary.csv"
    detail_path = "outputs/eval_detail.csv"

    if os.path.exists(eval_path):
        summary_df = pd.read_csv(eval_path)
        detail_df  = pd.read_csv(detail_path)

        st.markdown("#### Headline metrics")
        cols = st.columns(len(summary_df))
        for col, (_, row) in zip(cols, summary_df.iterrows()):
            col.metric(row["Metric"].replace("_", " "), row["Value"])

        st.markdown("#### Acceptance distribution (NBX vs baseline)")
        hist_data = detail_df[["p_baseline","p_nbx"]].copy()
        hist_data.columns = ["Baseline acceptance", "NBX acceptance"]
        st.bar_chart(hist_data.head(200))

        st.markdown("#### Evaluation detail (first 500 rows)")
        st.dataframe(detail_df.head(500), use_container_width=True)

        st.download_button(
            "⬇ Download full eval CSV",
            data=detail_df.to_csv(index=False).encode(),
            file_name="nbx_eval_detail.csv",
            mime="text/csv",
        )
    else:
        st.info("Run the pipeline first (▶ Run Pipeline tab).")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 – ARCHITECTURE
# ══════════════════════════════════════════════════════════════════════════════
with tab_arch:
    st.markdown("### 6-Stage Pipeline Architecture (Caner's Framework)")

    stages = [
        ("1 · Data Ingestion",       "BSS/CRM, billing, usage, device/network, engagement, care transcripts + sentiment (social/chatbot/complaints). No real PII — fully synthetic."),
        ("2 · Context Window",       "Unified per-subscriber object combining ALL signal types from every source system. Single dict passed downstream. Maps to SK Telecom's subscriber-context layer."),
        ("3 · Campaign Trigger",     "Next-token prediction: gradient-boosted propensity model scores every campaign for each subscriber and selects the one they are most likely to respond to next (behavioural sequence modelling)."),
        ("4 · LLM Reasoning",        "Claude API (claude-sonnet-4-6) generates:\n• WHY — a grounded rationale from the context window\n• HOW — a channel-compliant customer message\nFalls back to rule-based offline stand-in if no API key."),
        ("5 · Guard Rails",          "Margin guard (LOW LTV retention soft-cap), consent check (GDPR/PDPA), credit review (late payments + high value), network-issue hold (don't upsell a subscriber with active network complaints)."),
        ("6 · Evaluation",           "CSV loop: per-subscriber acceptance simulation · baseline vs NBX · revenue uplift · extrapolated annual QAR at 500k / 2M subscribers."),
    ]
    for title, desc in stages:
        with st.expander(f"**{title}**", expanded=False):
            st.markdown(desc)

    st.markdown("---")
    st.markdown("#### Reference architecture")
    st.markdown("""
| Layer | This prototype | SK Telecom V4 analogue |
|---|---|---|
| Data | Synthetic CRM + transcripts | BSS + care + social |
| Context | `build_context_window()` | Subscriber Context Object |
| Trigger | `CampaignTrigger` (GBT) | TWICE-Rec next-token prediction |
| LLM | Claude (Anthropic SDK) | Internal LLM + RARL |
| Rationale | WHY field | Self-Annotated Rationale Construction |
| Message | HOW field | Collaborative Preference Alignment |
| Guard | `guard_rail()` | Margin + consent rules |
| Eval | `evaluate()` → CSV | A/B click-through uplift |
""")

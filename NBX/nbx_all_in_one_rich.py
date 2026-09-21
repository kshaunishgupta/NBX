#!/usr/bin/env python3
"""
================================================================================
NBX - NEXT BEST EXPERIENCE : EXPANDED PoC (rich telecom data, single file)
================================================================================
LLM "Next Best Experience" layer ON TOP of the existing NBA contextual bandit,
fed by a telecom-grade context: BSS/CRM, billing, consumption, device,
engagement, care (call transcript & sentiment), and NBO/CVM outputs.

RUN
  pip install pandas numpy scikit-learn
  python nbx_all_in_one_rich.py

OPTIONAL REAL LLM (OpenAI-compatible; else offline stand-in)
  export NBX_LLM_PROVIDER=openai
  export NBX_LLM_BASE_URL=https://api.openai.com/v1
  export NBX_LLM_API_KEY=sk-...
  export NBX_LLM_MODEL=gpt-4o-mini
  # local/in-tenant:  NBX_LLM_PROVIDER=local  NBX_LLM_BASE_URL=http://localhost:11434/v1
================================================================================
"""
import os, re, json, urllib.request
import numpy as np, pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
# This PoC now lives alongside the Streamlit app (app.py + nbx_pipeline.py),
# which writes its own CSVs to ./outputs/. Use a subfolder so the two pipelines'
# results never mix -- they have different schemas and different row counts.
OUT="outputs/all_in_one"; os.makedirs(OUT, exist_ok=True)

SOURCE_SYSTEMS = {
  "BSS/CRM": ["AccountType","Language","Tenure_M","Segment","LTV_Band","Lifecycle","Days_To_Renewal","Consent_Flag"],
  "Billing": ["ARPU_QAR","ARPU_Trend","Late_Payments","Outstanding_QAR","Dunning_Flag","Payment_Method","MonthlyValueQAR"],
  "Consumption": ["Data_Cap_GB","Data_Used_GB","Data_Usage_Pct","Overage_Months","Voice_Min","Intl_Min","Roaming_Days","Usage_Trend","Top_App","Pct_5G"],
  "Device/Network": ["Device_Model","Device_Type","Device_Tier","Network_Capability","Device_Age_M","Upgrade_Eligible","Last_Upgrade_M",
                     "Avg_DL_Speed_Mbps","Dropped_Call_Rate","Latency_ms","Coverage_Score","Cell_Congestion","Network_Complaints_90d","Network_Issue_Flag"],
  "Engagement": ["Channel_Affinity","App_Logins_30d","Loyalty_Tier","Last_Campaign_Resp"],
  "Care/Unstructured": ["Recent_Complaint","Sentiment","NPS","Call_Transcript"],
  "NBA/NBO/CVM": ["ProductID","ActionType","Churn_Prob","NBO_Alt_Action","NBO_Reason"],
}
PII_FIELDS={"Call_Transcript"}

def generate_rich_telecom_data(num_rows=20000):
    rng=np.random.default_rng(42)
    msisdn=[f"9745{rng.integers(1000000,9999999)}" for _ in range(num_rows)]
    actions=["10GB_Aggressive_HVC","1GB_plus_100mins","200mins_Upsell","Home_Free_OoredooTV","Roaming_Daily_Pass",
             "5G_Unlimited_Upgrade","Retention_Save_25pct","Data_AddOn_5GB","Device_Upgrade_iPhone","Loyalty_Points_2x"]
    atmap={"10GB_Aggressive_HVC":"Retention","Retention_Save_25pct":"Retention","1GB_plus_100mins":"NBO","Data_AddOn_5GB":"NBO",
           "200mins_Upsell":"Upsell","5G_Unlimited_Upgrade":"Upsell","Home_Free_OoredooTV":"CrossSell","Roaming_Daily_Pass":"CrossSell",
           "Device_Upgrade_iPhone":"Upsell","Loyalty_Points_2x":"Retention"}
    price={"10GB_Aggressive_HVC":120,"Retention_Save_25pct":90,"1GB_plus_100mins":40,"Data_AddOn_5GB":50,"200mins_Upsell":35,
           "5G_Unlimited_Upgrade":200,"Home_Free_OoredooTV":150,"Roaming_Daily_Pass":60,"Device_Upgrade_iPhone":280,"Loyalty_Points_2x":0}
    segs=["HVC","Mass","Youth","Business"]; ltvs=["HIGH","MED","LOW"]; lifes=["Onboarding","Growth","Maturity","Renewal","AtRisk"]
    chs=["SMS","Push","Email","Call","WhatsApp"]; accts=["Postpaid","Prepaid","Hybrid"]
    devices=["iPhone 14","iPhone 15","Galaxy S23","Galaxy A54","Pixel 8","Huawei P60","Other"]
    pays=["Autopay_Card","Manual_Card","Cash","Bank_Transfer"]; apps=["YouTube","TikTok","Instagram","WhatsApp","Netflix","Gaming","Snapchat"]
    t_neg=["customer complained about slow data speeds at home","customer asked about high bill last month",
           "customer reported dropped calls in their area","customer unhappy after roaming charges abroad","customer threatened to switch to a competitor"]
    t_pos=["customer asked how to add more data","customer inquired about upgrading their phone","customer asked about international calling rates",
           "customer wanted to join the loyalty program","no recent call-center contact"]
    ca=rng.choice(actions,num_rows); rows=[]
    for i in range(num_rows):
        a=ca[i]; at=atmap[a]
        seg=rng.choice(segs,p=[.15,.55,.20,.10]); ltv=rng.choice(ltvs,p=[.2,.5,.3]); life=rng.choice(lifes,p=[.1,.3,.3,.15,.15]); acct=rng.choice(accts,p=[.55,.35,.10])
        churn=float(np.clip(rng.beta(2,5)+(0.3 if life=="AtRisk" else 0),0,1))
        cap=int(rng.choice([5,10,20,50,100])); used=float(np.clip(rng.beta(2,2)*cap*1.2,0,cap*1.3)); dpct=round(min(used/cap*100,130),1); ov=int(rng.integers(0,4))
        voice=int(rng.gamma(2,120)); intl=int(rng.gamma(1,15)) if rng.random()<.3 else 0; roam=int(rng.integers(0,8)) if rng.random()<.2 else 0
        utrend=rng.choice(["Rising","Stable","Falling"],p=[.3,.45,.25]); topapp=rng.choice(apps); g5=round(float(rng.random()),2)
        arpu=round(price.get(a,80)*0.5+rng.gamma(3,20),1); atrend=rng.choice(["Up","Flat","Down"],p=[.3,.4,.3])
        late=int(rng.binomial(3,0.15)); outst=round(float(rng.gamma(1,30)) if late else 0.0,1); dun=int(late>=2)
        dev=rng.choice(devices); dage=int(rng.integers(1,48)); elig=int(dage>=20); lupg=int(rng.integers(1,60))
        dtype=rng.choice(["Smartphone","Feature Phone","Router/MiFi","Tablet","IoT"],p=[.78,.06,.08,.05,.03])
        if dev in ("iPhone 15","Galaxy S23","Pixel 8","Huawei P60"): dtier,netcap="Flagship","5G"
        elif dev in ("iPhone 14","Galaxy A54"): dtier,netcap="Mid",("5G" if rng.random()<.7 else "4G")
        else: dtier,netcap=("Budget","4G")
        if dtype in ("Feature Phone","IoT"): netcap="4G"; dtier="Budget"
        cong=round(float(rng.beta(2,9)),2)
        dl=round(float(np.clip((180 if netcap=="5G" else 60)*(1-0.6*cong)*rng.uniform(0.5,1.2),3,400)),1)
        lat=int(np.clip((18 if netcap=="5G" else 40)+cong*120+rng.normal(0,8),8,300))
        dcr=round(float(np.clip(rng.beta(2,40)+cong*0.05,0,0.4)),3)
        cov=round(float(np.clip(1-cong*0.6-rng.beta(2,8),0,1)),2)
        net_comp=int(rng.binomial(2,0.04+cong*0.15))
        net_issue=int((dcr>0.12) or (cov<0.25) or (lat>160) or (net_comp>=1) or (dl<6))
        dr=int(rng.integers(1,60)) if life=="Renewal" else int(rng.integers(1,365)); ten=int(rng.integers(1,120))
        cha=rng.choice(chs,p=[.35,.25,.12,.13,.15]); logins=int(rng.poisson(8)); consent=int(rng.binomial(1,0.85))
        loy=rng.choice(["None","Silver","Gold","Platinum"],p=[.4,.3,.2,.1]); lcr=rng.choice(["Accepted","Ignored","Rejected","None"],p=[.15,.45,.15,.25])
        comp=int(rng.binomial(1,0.15+0.2*(churn>0.6)))
        sent=rng.choice(["Negative","Neutral","Positive"],p=[.25,.5,.25]) if comp else rng.choice(["Neutral","Positive"],p=[.6,.4])
        trans=rng.choice(t_neg) if (comp or sent=="Negative") else rng.choice(t_pos); nps=int(rng.integers(0,11))
        nbo_alt=rng.choice([x for x in actions if x!=a]); nbo_r=rng.choice(["High data usage trend","Renewal window","Churn risk detected","Device upgrade eligible","Loyalty reward due"])
        bp=0.12
        if at=="Retention" and churn>0.6: bp+=0.25
        if at=="Upsell" and churn>0.6: bp-=0.10
        if at=="Upsell" and dpct>80: bp+=0.18
        if at=="NBO" and seg=="Mass": bp+=0.08
        if life=="Renewal" and at=="Retention": bp+=0.15
        if ltv=="HIGH": bp+=0.05
        if a=="Device_Upgrade_iPhone" and elig: bp+=0.15
        if sent=="Negative": bp-=0.08
        if utrend=="Rising" and at=="Upsell": bp+=0.10
        if consent==0: bp-=0.05
        if net_issue and at=="Upsell": bp-=0.10
        if net_issue and at=="Retention": bp+=0.06
        acc=int(rng.binomial(1,np.clip(bp,0.01,0.95)))
        rows.append([msisdn[i],a,at,price[a],seg,ltv,life,acct,("EN" if rng.random()<.6 else "AR"),ten,round(churn,3),dr,
                     cap,round(used,1),dpct,ov,voice,intl,roam,utrend,topapp,g5,arpu,atrend,late,outst,dun,rng.choice(pays),
                     dev,dtype,dtier,netcap,dage,elig,lupg,dl,dcr,lat,cov,cong,net_comp,net_issue,
                     cha,logins,consent,loy,lcr,comp,sent,nps,trans,nbo_alt,nbo_r,acc])
    cols=["MSISDN","ProductID","ActionType","MonthlyValueQAR","Segment","LTV_Band","Lifecycle","AccountType","Language","Tenure_M","Churn_Prob","Days_To_Renewal",
          "Data_Cap_GB","Data_Used_GB","Data_Usage_Pct","Overage_Months","Voice_Min","Intl_Min","Roaming_Days","Usage_Trend","Top_App","Pct_5G","ARPU_QAR","ARPU_Trend",
          "Late_Payments","Outstanding_QAR","Dunning_Flag","Payment_Method","Device_Model","Device_Type","Device_Tier","Network_Capability","Device_Age_M","Upgrade_Eligible","Last_Upgrade_M",
          "Avg_DL_Speed_Mbps","Dropped_Call_Rate","Latency_ms","Coverage_Score","Cell_Congestion","Network_Complaints_90d","Network_Issue_Flag","Channel_Affinity","App_Logins_30d",
          "Consent_Flag","Loyalty_Tier","Last_Campaign_Resp","Recent_Complaint","Sentiment","NPS","Call_Transcript","NBO_Alt_Action","NBO_Reason","Accept_Ind"]
    return pd.DataFrame(rows,columns=cols)

CAT=["ProductID","Segment","LTV_Band","Lifecycle","AccountType","Usage_Trend","ARPU_Trend","Device_Model","Device_Type","Device_Tier","Network_Capability","Channel_Affinity","Loyalty_Tier","Last_Campaign_Resp","Top_App","Sentiment"]
NUM=["MonthlyValueQAR","Tenure_M","Churn_Prob","Days_To_Renewal","Data_Cap_GB","Data_Used_GB","Data_Usage_Pct","Overage_Months","Voice_Min","Intl_Min","Roaming_Days","Pct_5G",
     "ARPU_QAR","Late_Payments","Outstanding_QAR","Dunning_Flag","Device_Age_M","Upgrade_Eligible","Last_Upgrade_M",
     "Avg_DL_Speed_Mbps","Dropped_Call_Rate","Latency_ms","Coverage_Score","Cell_Congestion","Network_Complaints_90d","Network_Issue_Flag","App_Logins_30d","Consent_Flag","NPS"]
class NBABandit:
    def __init__(self):
        pre=ColumnTransformer([("cat",OneHotEncoder(handle_unknown="ignore"),CAT)],remainder="passthrough")
        self.model=Pipeline([("pre",pre),("clf",GradientBoostingClassifier(n_estimators=100,max_depth=3))])
    def fit(self,df):
        self.actions=sorted(df.ProductID.unique()); self.price=df.groupby("ProductID").MonthlyValueQAR.first().to_dict()
        self.atype=df.groupby("ProductID").ActionType.first().to_dict(); self.model.fit(df[CAT+NUM],df.Accept_Ind); return self
    def recommend(self,df):
        # batched scoring: build all candidate rows at once
        big=[]
        for _,row in df.iterrows():
            for a in self.actions:
                c=row.copy(); c.ProductID=a; c.MonthlyValueQAR=self.price[a]; c.ActionType=self.atype[a]; big.append(c)
        B=pd.DataFrame(big); probs=self.model.predict_proba(B[CAT+NUM])[:,1]
        na=len(self.actions); out=[]
        for i,(_,row) in enumerate(df.iterrows()):
            seg=probs[i*na:(i+1)*na]; bi=int(np.argmax(seg))
            out.append({"MSISDN":row.MSISDN,"Selected_Action":self.actions[bi],"ActionType":self.atype[self.actions[bi]],
                        "Propensity":round(float(seg[bi]),4),"MonthlyValueQAR":self.price[self.actions[bi]]})
        return pd.DataFrame(out)

class LLMClient:
    def __init__(self):
        self.provider=os.getenv("NBX_LLM_PROVIDER","none").lower(); self.base_url=os.getenv("NBX_LLM_BASE_URL","").rstrip("/")
        self.api_key=os.getenv("NBX_LLM_API_KEY",""); self.model=os.getenv("NBX_LLM_MODEL","gpt-4o-mini"); self.timeout=int(os.getenv("NBX_LLM_TIMEOUT","20"))
    @property
    def enabled(self): return self.provider not in ("none","") and bool(self.base_url)
    def chat_json(self,sysp,usrp,temperature=0.2,max_tokens=400):
        url=f"{self.base_url}/chat/completions"
        payload={"model":self.model,"messages":[{"role":"system","content":sysp},{"role":"user","content":usrp}],"temperature":temperature,"max_tokens":max_tokens,"response_format":{"type":"json_object"}}
        h={"Content-Type":"application/json"}
        if self.api_key: h["Authorization"]=f"Bearer {self.api_key}"
        if self.provider=="azure": h["api-key"]=self.api_key
        req=urllib.request.Request(url,data=json.dumps(payload).encode(),headers=h,method="POST")
        with urllib.request.urlopen(req,timeout=self.timeout) as r: body=json.loads(r.read().decode())
        return json.loads(body["choices"][0]["message"]["content"])

def redact(t): return re.sub(r"\b\d{6,}\b","[REDACTED]",str(t))

# ══════════════════════════════════════════════════════════════════════════════
# VECTOR RETRIEVAL LAYER  (Step 4 of the vector-DB learning module)
# ══════════════════════════════════════════════════════════════════════════════
# BEFORE: build_context() shoved the customer's whole unstructured blob
#         (Call_Transcript) into every prompt.
# AFTER:  we semantically retrieve only the top-3 most decision-relevant
#         interactions out of that customer's 5-10 historical touchpoints.
#
# This is OPT-IN. If the store was never built (or chromadb is not installed),
# build_context() silently falls back to the original single-transcript
# behaviour, so the existing pipeline keeps working unchanged.
_VSTORE = None            # active VectorStore, or None
_VHISTORY = {}            # customer_id -> full history (for the "what was NOT retrieved" log)
_VLOG = False             # print per-customer retrieval traces
_VTOPK = 3

def enable_vector_retrieval(customer_ids, top_k=3, log=False, n_min=5, n_max=10, transcripts=None):
    """
    Generate synthetic interaction histories for these customers, embed them,
    and switch build_context() over to retrieval mode. Returns the VectorStore.

    transcripts: optional {msisdn: Call_Transcript}. The customer's EXISTING
        care transcript is added to the corpus alongside the synthetic history,
        so retrieval ranks it against everything else instead of discarding it.
        Without this, a genuinely relevant real transcript can be silently
        replaced by a less relevant synthetic interaction.
    """
    global _VSTORE, _VHISTORY, _VLOG, _VTOPK
    import sys as _sys
    _sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from data.interaction_history import generate_all_histories
    from pipeline.vector_store import build_vector_store

    ids = [str(c) for c in customer_ids]
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

def _retrieve_unstructured(customer_id):
    """
    Return (top_snippets, retrieved_records) for one customer, or (None, []) when
    retrieval is disabled -- which triggers the original fallback path.
    """
    if _VSTORE is None:
        return None, []
    hits = _VSTORE.retrieve(str(customer_id), top_k=_VTOPK)
    if _VLOG:
        kept = {h["interaction_id"] for h in hits}
        skipped = [i for i in _VHISTORY.get(str(customer_id), []) if i["interaction_id"] not in kept]
        print(f"\n   [retrieval] customer {customer_id}: "
              f"{len(hits)} of {len(_VHISTORY.get(str(customer_id), []))} interactions used")
        for h in hits:
            print(f"      KEPT    sim={h['similarity']:+.3f}  {h['date']}  {h['channel']:<9} {h['text']}")
        for s in skipped:
            print(f"      dropped              {s['date']}  {s['channel']:<9} {s['text']}")
    return [redact(h["text"]) for h in hits], hits

def build_context(c,b):
    ctx = _build_context_base(c,b)
    # --- swap the "dump everything" transcript for retrieved, ranked evidence --
    snippets, hits = _retrieve_unstructured(getattr(b, "MSISDN", None))
    if snippets:
        # transcript stays a plain string (the offline reasoner reads it),
        # but it is now the single MOST RELEVANT interaction, not a raw dump.
        ctx["transcript"] = snippets[0]
        ctx["retrieved_interactions"] = [
            {"text": redact(h["text"]), "date": h["date"],
             "channel": h["channel"], "similarity": h["similarity"]}
            for h in hits
        ]
        ctx["retrieval_mode"] = f"vector_top{len(hits)}"
    else:
        ctx["retrieval_mode"] = "full_dump"   # original behaviour
    return ctx

def _build_context_base(c,b):
    return {"action":b.Selected_Action,"action_type":b.ActionType,"propensity":b.Propensity,"value_qar":b.MonthlyValueQAR,
            "segment":c.Segment,"ltv":c.LTV_Band,"lifecycle":c.Lifecycle,"account_type":c.AccountType,"language":c.Language,"tenure_m":c.Tenure_M,
            "churn":c.Churn_Prob,"days_renewal":c.Days_To_Renewal,"data_pct":c.Data_Usage_Pct,"data_used_gb":c.Data_Used_GB,"data_cap_gb":c.Data_Cap_GB,
            "overage":c.Overage_Months,"voice_min":c.Voice_Min,"intl_min":c.Intl_Min,"roaming_days":c.Roaming_Days,"usage_trend":c.Usage_Trend,"top_app":c.Top_App,
            "pct_5g":c.Pct_5G,"arpu":c.ARPU_QAR,"arpu_trend":c.ARPU_Trend,"late_payments":c.Late_Payments,"device":c.Device_Model,"device_age":c.Device_Age_M,
            "upgrade_eligible":c.Upgrade_Eligible,"last_upgrade_m":c.Last_Upgrade_M,
            "device_type":c.Device_Type,"device_tier":c.Device_Tier,"network_capability":c.Network_Capability,
            "dl_speed_mbps":c.Avg_DL_Speed_Mbps,"dropped_call_rate":c.Dropped_Call_Rate,"latency_ms":c.Latency_ms,
            "coverage_score":c.Coverage_Score,"cell_congestion":c.Cell_Congestion,"network_complaints_90d":c.Network_Complaints_90d,
            "network_issue":c.Network_Issue_Flag,"channel":c.Channel_Affinity,"app_logins":c.App_Logins_30d,"consent":c.Consent_Flag,
            "loyalty":c.Loyalty_Tier,"last_campaign_resp":c.Last_Campaign_Resp,"complaint":c.Recent_Complaint,"sentiment":c.Sentiment,"nps":c.NPS,
            "transcript":redact(c.Call_Transcript),"nbo_reason":c.NBO_Reason}

SYSTEM_PROMPT="""You are NBX, a telecom Next-Best-Experience copywriter for Ooredoo.
You receive a SELECTED ACTION (already chosen by the recommender) and a rich CUSTOMER CONTEXT
(BSS, billing, usage, device, engagement, care/sentiment, NBO reasoning).
HARD RULES:
1. Do NOT change or re-rank the action; use it exactly as given.
2. Only use facts in the context. Never invent numbers, savings, dates or features.
3. Every numeric claim must match the context exactly.
4. If churn_prob>=0.6 or sentiment is Negative, lead with empathy/retention, never a hard upsell.
5. Reflect the recent care interaction (transcript/sentiment) where relevant, tactfully.
5b. If network_issue is 1, acknowledge the experience issue first (do not pitch a heavy upsell);
    prefer retention/loyalty or a network-relevant action, never blame the customer.
6. Respect channel limits: SMS<=160, Push<=120, WhatsApp<=300, Email longer, Call=agent script.
7. Write in the customer's language (EN/AR) per context.
Return ONLY valid JSON: {"rationale":"...","message":"...","grounded":true,"used_fields":[...]}"""
def build_user_prompt(ctx):
    return (f"SELECTED ACTION (do not change): {ctx['action']} | type {ctx['action_type']} | value_qar {ctx['value_qar']}\n"
            f"CHANNEL: {ctx['channel']}\nNBO_REASON: {ctx['nbo_reason']}\n\nCUSTOMER CONTEXT (only use these facts):\n"+json.dumps(ctx,indent=2,default=str))
def validate_grounding(ctx,result):
    text=(result.get("rationale","")+" "+result.get("message","")).lower()
    allowed={str(int(ctx["overage"])),str(int(ctx["days_renewal"])),str(int(round(ctx["data_pct"]))),str(int(ctx["value_qar"])),
             str(int(round(ctx["data_used_gb"]))),str(int(ctx["data_cap_gb"])),str(int(round(ctx["arpu"]))),str(int(ctx["voice_min"])),str(int(ctx["nps"])),str(int(ctx["tenure_m"])),
             str(int(round(ctx["dl_speed_mbps"]))),str(int(ctx["latency_ms"])),str(int(ctx["network_complaints_90d"]))}
    for nraw in re.findall(r"\b\d{1,5}\b",text):
        n=nraw.lstrip("0") or "0"
        if n in allowed: continue
        if any(abs(int(n)-int(a))<=1 for a in allowed if a.isdigit()): continue
        return False,f"ungrounded number '{nraw}'"
    return True,"ok"

CHANNEL_LIMITS={"SMS":160,"Push":120,"WhatsApp":300,"Email":600,"Call":400}
_CLIENT=LLMClient()
def _offline_reason(x):
    b=[]
    if x.get("network_issue")==1:
        nb=[]
        if x["dropped_call_rate"]>0.05: nb.append(f"high dropped-call rate ({x['dropped_call_rate']:.0%})")
        if x["coverage_score"]<0.4: nb.append("weak coverage at their location")
        if x["latency_ms"]>120: nb.append(f"high latency ({x['latency_ms']}ms)")
        if x["dl_speed_mbps"]<10: nb.append(f"low speeds ({x['dl_speed_mbps']:.0f}Mbps)")
        if x["network_complaints_90d"]>=1: nb.append(f"{x['network_complaints_90d']} network complaint(s) in 90 days")
        b.append("is experiencing network issues ("+", ".join(nb or ["degraded experience"])+")")
    if x["sentiment"]=="Negative" or x["complaint"]==1: b.append(f"recently contacted care ({x['transcript']})")
    if x["overage"]>=1: b.append(f"exceeded the data cap in {x['overage']} of the last 3 months")
    if x["data_pct"]>=80: b.append(f"is a heavy data user ({x['data_pct']:.0f}% of {x['data_cap_gb']}GB)")
    if x["usage_trend"]=="Rising": b.append("has rising data usage")
    if x["days_renewal"]<=60: b.append(f"is {x['days_renewal']} days from renewal")
    if x["upgrade_eligible"]==1: b.append(f"is eligible for a device upgrade ({x['device']}, {x['device_age']}m old)")
    if x["churn"]>=0.6: b.append("shows elevated churn risk")
    if x["loyalty"] in ("Gold","Platinum"): b.append(f"is a {x['loyalty']} loyalty member")
    if not b: b.append(f"is an active {x['segment']} customer (tenure {x['tenure_m']}m)")
    return "Customer "+", ".join(b)+"."
def _offline_message(x,why):
    a,ch=x["action"],x["channel"]; empath=(x["sentiment"]=="Negative" or x["churn"]>=0.6 or x.get("network_issue")==1)
    if x.get("network_issue")==1:
        core=f"We noticed your service hasn't been at its best lately - to make it right, our {a.replace('_',' ')} is ready for you"
    elif empath: core=f"Thanks for being with us - to make things right, our {a.replace('_',' ')} is ready for you"
    elif x["action_type"]=="Upsell":
        if x["upgrade_eligible"] and "Device" in a: core=f"You're eligible to upgrade your {x['device']} - {a.replace('_',' ')} is ready"
        else: core=f"You're using {x['data_pct']:.0f}% of your {x['data_cap_gb']}GB - {a.replace('_',' ')} gives you headroom"
    elif x["action_type"]=="Retention":
        core=f"Based on your usage, our {a.replace('_',' ')} keeps you covered"
        if x["days_renewal"]<=60: core+=f" and your renewal is in {x['days_renewal']} days"
    elif x["action_type"]=="CrossSell": core=f"Add {a.replace('_',' ')} to get more from your plan"
    else: core=f"{a.replace('_',' ')} suits your usage"
    m={"SMS":f"Hi! {core}. Reply YES. Ooredoo","Push":f"{core} - tap to activate.","WhatsApp":f"Hi! {core}. Tap below to confirm. Ooredoo",
       "Email":f"Dear Customer,\n\n{why}\n\n{core}.\n\nOoredoo","Call":f"AGENT: {why} Recommend: {core}."}[ch]
    return m[:CHANNEL_LIMITS[ch]]
def margin_guard(x):
    if x["action_type"]=="Retention" and x["ltv"]=="LOW" and x["churn"]>=0.6: return "SOFTENED"
    if x["late_payments"]>=2 and x["value_qar"]>150: return "REVIEW_CREDIT"
    return "APPROVED"
def generate_one(x):
    if _CLIENT.enabled:
        try:
            r=_CLIENT.chat_json(SYSTEM_PROMPT,build_user_prompt(x)); ok,_=validate_grounding(x,r)
            msg=(r.get("message") or "").strip()[:CHANNEL_LIMITS[x["channel"]]]; why=(r.get("rationale") or "").strip()
            if ok and msg and why: return why,msg,"llm"
            why=_offline_reason(x); return why,_offline_message(x,why),"llm->fallback"
        except Exception:
            why=_offline_reason(x); return why,_offline_message(x,why),"llm->fallback"
    why=_offline_reason(x); return why,_offline_message(x,why),"offline"
def run_nbx(cust_df,bandit_df):
    cust=cust_df.drop_duplicates("MSISDN").set_index("MSISDN"); rows=[]
    for _,b in bandit_df.iterrows():
        x=build_context(cust.loc[b.MSISDN],b); why,msg,src=generate_one(x)
        rows.append({"msisdn":b.MSISDN,"action":x["action"],"action_type":x["action_type"],"channel":x["channel"],"sentiment":x["sentiment"],
                     "rationale_WHY":why,"message_HOW":msg,"guard_status":margin_guard(x),"gen_source":src})
    return pd.DataFrame(rows)

def value_simulation(rec,nbx,data):
    d=data.drop_duplicates("MSISDN")
    keep=["MSISDN","Churn_Prob","Data_Usage_Pct","Overage_Months","Days_To_Renewal","LTV_Band","Usage_Trend","Upgrade_Eligible","Sentiment","Loyalty_Tier","Consent_Flag","Network_Issue_Flag","Device_Type"]
    m=rec.merge(d[keep],on="MSISDN").merge(nbx[["msisdn","guard_status"]],left_on="MSISDN",right_on="msisdn")
    m["p_baseline"]=m.Propensity*0.85
    def mult(r):
        x=1.10
        if r.Overage_Months>=1 or r.Data_Usage_Pct>=80: x+=0.12
        if r.Days_To_Renewal<=60: x+=0.10
        if r.ActionType=="Retention" and r.Churn_Prob>=0.6: x+=0.10
        if r.Usage_Trend=="Rising" and r.ActionType=="Upsell": x+=0.06
        if r.Upgrade_Eligible==1 and "Device" in r.Selected_Action: x+=0.08
        if r.Sentiment=="Negative": x+=0.06
        if r.Network_Issue_Flag==1 and r.ActionType in ("Retention","CrossSell"): x+=0.07
        if r.Network_Issue_Flag==1 and r.ActionType=="Upsell": x-=0.04
        if r.Loyalty_Tier in ("Gold","Platinum"): x+=0.04
        x+=0.05
        if r.guard_status in ("SOFTENED","REVIEW_CREDIT"): x-=0.05
        if r.Consent_Flag==0: x=1.0
        return min(x,1.55)
    m["p_nbx"]=np.clip(m.p_baseline*m.apply(mult,axis=1),0,0.95); M=12; n=len(m)
    base=(m.p_baseline*m.MonthlyValueQAR*M).sum(); nbxr=(m.p_nbx*m.MonthlyValueQAR*M).sum()
    m.to_csv(f"{OUT}/value_simulation.csv",index=False)
    pd.DataFrame({"Metric":["Acceptance baseline","Acceptance NBX","Revenue uplift %","Incr annual rev QAR (sample)","Extrapolated +QAR/yr @500k","Extrapolated +QAR/yr @2M"],
                  "Value":[f"{m.p_baseline.mean():.1%}",f"{m.p_nbx.mean():.1%}",f"+{nbxr/base-1:.1%}",f"{nbxr-base:,.0f}",f"{(nbxr-base)*(500000/n):,.0f}",f"{(nbxr-base)*(2000000/n):,.0f}"]}).to_csv(f"{OUT}/value_summary.csv",index=False)
    return m.p_baseline.mean(),m.p_nbx.mean(),base,nbxr,n

def main():
    print("="*64); print("NBX EXPANDED PoC - rich telecom data"); print("="*64)
    print(f"[mode] LLM {'ENABLED ('+_CLIENT.model+')' if _CLIENT.enabled else 'OFFLINE stand-in'}")
    print("1) Generating rich telecom data (44 fields)...")
    df=generate_rich_telecom_data(20000); df.to_csv(f"{OUT}/telecom_nbx_rich.csv",index=False)
    print(f"   {df.shape[0]} rows x {df.shape[1]} cols | accept {df.Accept_Ind.mean():.1%}")
    print("2) Training NBA bandit on rich features (WHAT)...")
    bandit=NBABandit().fit(df); score=df.drop_duplicates("MSISDN").head(800).reset_index(drop=True)
    rec=bandit.recommend(score); rec.to_csv(f"{OUT}/nba_recommendations.csv",index=False)
    # Optional: swap the "dump the whole transcript" step for semantic retrieval.
    #   NBX_VECTOR_RETRIEVAL=1 python3 nbx_all_in_one_rich.py
    #   NBX_VECTOR_LOG=1       also prints kept-vs-dropped interactions per customer
    if os.getenv("NBX_VECTOR_RETRIEVAL","").lower() in ("1","true","yes"):
        print("2b) Vector retrieval ENABLED - embedding synthetic interaction histories...")
        enable_vector_retrieval(rec.MSISDN.tolist(), top_k=3,
                                log=os.getenv("NBX_VECTOR_LOG","").lower() in ("1","true","yes"),
                                transcripts=dict(zip(score.MSISDN, score.Call_Transcript)))
    print("3) NBX layer on top, rich context incl. transcript/sentiment/NBO (WHY+HOW)...")
    nbx=run_nbx(score,rec); nbx.to_csv(f"{OUT}/nbx_experience.csv",index=False)
    print("   sample WHY:",nbx.iloc[0].rationale_WHY); print("   sample HOW:",nbx.iloc[0].message_HOW)
    print("4) Value simulation...")
    ba,na,base,nbxr,n=value_simulation(rec,nbx,df)
    print("="*64); print(f"   Acceptance : {ba:.1%} -> {na:.1%}"); print(f"   Revenue uplift : +{nbxr/base-1:.1%}")
    print(f"   Incremental annual revenue (sample {n}): QAR {nbxr-base:,.0f}")
    print(f"   Extrapolated: +QAR {(nbxr-base)*(2000000/n):,.0f}/yr at 2M customers"); print("="*64); print(f"Outputs in ./{OUT}/")

if __name__=="__main__": main()

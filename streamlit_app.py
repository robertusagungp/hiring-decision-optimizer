import streamlit as st
import requests
import os
import pandas as pd

# =========================================================
# LOAD GROQ API KEY (Streamlit Cloud secrets OR local env)
# =========================================================
if "GROQ_API_KEY" in st.secrets:
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
else:
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")

GROQ_ENDPOINT = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = "llama-3.3-70b-versatile"


# =========================================================
# LANGUAGE (ID/EN)
# =========================================================
LANG = st.sidebar.selectbox("Language / Bahasa", ["English", "Bahasa Indonesia"], index=0)

TXT = {
    "English": {
        "app_title": "Hiring Decision Optimizer (Advanced)",
        "subtitle": "Rule-based scoring + Groq explanation + role-fit auto-detect + shortlist ranking",
        "candidate_block": "Candidate Input",
        "candidate_name": "Candidate Name",
        "job_position": "Target Job Position (manual)",
        "auto_role": "Auto-detect role fit",
        "exp": "Years of Experience",
        "lead": "Leadership Experience (years)",
        "skill": "Skill Match (%)",
        "test": "Technical Test Score (0-100)",
        "interview": "Interview Score (0-100)",
        "education": "Education Level",
        "curr_salary": "Current Salary",
        "exp_salary": "Expected Salary",
        "budget": "Salary Budget (max)",
        "evaluate": "Evaluate",
        "add_shortlist": "Add to Shortlist",
        "clear_shortlist": "Clear Shortlist",
        "result": "Result",
        "score": "Score",
        "decision": "Decision",
        "role_reco": "Recommended Role",
        "role_fit": "Role Fit",
        "confidence": "Confidence",
        "flags": "Risk Flags",
        "ai_explain": "AI Explanation (Groq)",
        "shortlist": "Shortlist: Multi-candidate Comparison",
        "ranked": "Ranked Recommendations",
        "download_csv": "Download shortlist as CSV",
        "top3": "Top 3 candidates",
        "no_key": "⚠️ GROQ_API_KEY not configured. AI explanation disabled.",
        "api_err": "⚠️ Groq API error",
        "select_role_note": "Tip: If you enable auto-detect, the app will recommend the best-fit level (Junior/Officer/Senior/Supervisor/Manager).",
    },
    "Bahasa Indonesia": {
        "app_title": "Hiring Decision Optimizer (Advanced)",
        "subtitle": "Rule-based scoring + penjelasan Groq + auto-deteksi role fit + ranking shortlist",
        "candidate_block": "Input Kandidat",
        "candidate_name": "Nama Kandidat",
        "job_position": "Target Posisi (manual)",
        "auto_role": "Auto-detect role fit",
        "exp": "Pengalaman Kerja (tahun)",
        "lead": "Pengalaman Leadership (tahun)",
        "skill": "Kecocokan Skill (%)",
        "test": "Skor Tes Teknis (0-100)",
        "interview": "Skor Interview (0-100)",
        "education": "Pendidikan",
        "curr_salary": "Gaji Saat Ini",
        "exp_salary": "Ekspektasi Gaji",
        "budget": "Budget Gaji (maks)",
        "evaluate": "Evaluasi",
        "add_shortlist": "Tambahkan ke Shortlist",
        "clear_shortlist": "Hapus Shortlist",
        "result": "Hasil",
        "score": "Skor",
        "decision": "Keputusan",
        "role_reco": "Role Rekomendasi",
        "role_fit": "Kecocokan Role",
        "confidence": "Confidence",
        "flags": "Risk Flags",
        "ai_explain": "Penjelasan AI (Groq)",
        "shortlist": "Shortlist: Perbandingan Multi Kandidat",
        "ranked": "Ranking Rekomendasi",
        "download_csv": "Download shortlist sebagai CSV",
        "top3": "Top 3 kandidat",
        "no_key": "⚠️ GROQ_API_KEY belum diset. Penjelasan AI dimatikan.",
        "api_err": "⚠️ Groq API error",
        "select_role_note": "Tip: Jika auto-detect dinyalakan, sistem akan merekomendasikan level terbaik (Junior/Officer/Senior/Supervisor/Manager).",
    }
}[LANG]


# =========================================================
# ROLE REQUIREMENTS (simple but practical)
# =========================================================
ROLES = ["Junior", "Officer", "Senior", "Supervisor", "Manager"]

ROLE_REQUIREMENTS = {
    "Junior":     {"min_exp": 0,  "ideal_exp": 2,  "min_lead": 0},
    "Officer":    {"min_exp": 2,  "ideal_exp": 4,  "min_lead": 0},
    "Senior":     {"min_exp": 4,  "ideal_exp": 7,  "min_lead": 0},
    "Supervisor": {"min_exp": 6,  "ideal_exp": 10, "min_lead": 1},
    "Manager":    {"min_exp": 8,  "ideal_exp": 15, "min_lead": 3},
}

EDU_SCORE = {
    "High School": 55,
    "Diploma": 65,
    "Bachelor": 75,
    "Master": 90,
    "PhD": 100
}


# =========================================================
# ROLE AUTO-DETECT
# =========================================================
def auto_detect_role(experience: float, leadership: float):
    """
    Return (recommended_role, confidence_0_100, explanation)
    Rule: choose highest role that matches minimum requirements.
    """
    best_role = "Junior"

    for role in ROLES:
        req = ROLE_REQUIREMENTS[role]
        if experience >= req["min_exp"] and leadership >= req["min_lead"]:
            best_role = role

    # Confidence: based on how close to ideal_exp for that role
    ideal = ROLE_REQUIREMENTS[best_role]["ideal_exp"]
    # If ideal is 0 (not possible here), guard
    if ideal <= 0:
        conf = 60
    else:
        # 0..100 based on ratio; cap
        ratio = min(experience / ideal, 1.2)  # allow slight overflow
        conf = int(max(35, min(95, ratio * 80)))  # 35..95

    # Add leadership boost for senior roles
    if best_role in ["Supervisor", "Manager"]:
        conf = int(min(98, conf + min(10, leadership * 2)))

    explanation = f"{best_role} (exp={experience}, lead={leadership})"
    return best_role, conf, explanation


def role_fit_label(target_role: str, recommended_role: str):
    """
    Compare target role vs recommended role.
    """
    idx_target = ROLES.index(target_role)
    idx_reco = ROLES.index(recommended_role)
    gap = idx_target - idx_reco

    if gap == 0:
        return "Excellent", 0
    if gap == 1:
        return "Slight Stretch", gap
    if gap >= 2:
        return "Over-stretched", gap
    if gap == -1:
        return "Over-qualified (mild)", gap
    return "Over-qualified", gap


# =========================================================
# ADVANCED SCORING (rule-based, interpretable)
# =========================================================
def clamp(x, lo=0.0, hi=100.0):
    return max(lo, min(hi, x))

def score_candidate(payload: dict):
    """
    Returns:
      score (0..100),
      decision (Strong Hire/Hire/Consider/No Hire),
      breakdown dict,
      risk_flags list,
      aux dict (salary_fit, raise_pct, role_fit_penalty, etc)
    """

    # Base fields
    role_target = payload["target_role"]
    role_reco = payload["recommended_role"]

    exp = float(payload["experience"])
    lead = float(payload["leadership"])
    skill = float(payload["skill_match"])
    test = float(payload["test_score"])
    interview = float(payload["interview_score"])
    edu = EDU_SCORE.get(payload["education"], 75)

    curr_salary = float(payload["current_salary"])
    exp_salary = float(payload["expected_salary"])
    budget = float(payload["salary_budget"])

    # Experience score relative to ideal for TARGET role
    ideal_exp = ROLE_REQUIREMENTS[role_target]["ideal_exp"]
    exp_score = clamp((exp / ideal_exp) * 100 if ideal_exp > 0 else 60)

    # Leadership score relative to ideal for TARGET role (weighted more for higher roles)
    lead_req = ROLE_REQUIREMENTS[role_target]["min_lead"]
    if role_target in ["Supervisor", "Manager"]:
        lead_score = clamp((lead / max(1.0, lead_req + 2)) * 100)
    else:
        lead_score = clamp((lead / 3.0) * 100)  # small bonus even for IC roles

    # Salary fit score (expected vs budget)
    if budget <= 0:
        salary_fit = 55
        salary_flag = "Salary budget not provided"
    else:
        if exp_salary <= budget:
            salary_fit = 100
            salary_flag = ""
        else:
            over = (exp_salary - budget) / budget
            salary_fit = clamp(100 - over * 140)  # harsher penalty when above budget
            salary_flag = "Expected salary exceeds budget"

    # Raise realism score (expected vs current)
    raise_pct = None
    if curr_salary > 0 and exp_salary > 0:
        raise_pct = ((exp_salary - curr_salary) / curr_salary) * 100
        if raise_pct <= 20:
            raise_score = 100
        elif raise_pct <= 35:
            raise_score = 80
        elif raise_pct <= 50:
            raise_score = 60
        else:
            raise_score = 35
    else:
        raise_score = 60  # unknown
        raise_pct = None

    # Role fit penalty (if target is too high vs recommended)
    fit_label, gap = role_fit_label(role_target, role_reco)
    if gap >= 2:
        role_fit_penalty = 12
    elif gap == 1:
        role_fit_penalty = 6
    elif gap == 0:
        role_fit_penalty = 0
    else:
        # over-qualified doesn't penalize much (could be okay)
        role_fit_penalty = 2

    # Weighting: higher role expects more leadership + interview
    if role_target in ["Supervisor", "Manager"]:
        weights = {
            "exp": 0.16,
            "skill": 0.16,
            "test": 0.14,
            "interview": 0.20,
            "edu": 0.08,
            "lead": 0.12,
            "salary_fit": 0.10,
            "raise": 0.04,
        }
    else:
        weights = {
            "exp": 0.18,
            "skill": 0.20,
            "test": 0.18,
            "interview": 0.18,
            "edu": 0.10,
            "lead": 0.06,
            "salary_fit": 0.07,
            "raise": 0.03,
        }

    total = (
        weights["exp"] * exp_score +
        weights["skill"] * skill +
        weights["test"] * test +
        weights["interview"] * interview +
        weights["edu"] * edu +
        weights["lead"] * lead_score +
        weights["salary_fit"] * salary_fit +
        weights["raise"] * raise_score
    )

    total = clamp(total - role_fit_penalty)
    total = round(total, 2)

    # Risk flags
    flags = []
    if exp < ROLE_REQUIREMENTS[role_target]["min_exp"]:
        flags.append("Below minimum experience for target role")
    if role_target in ["Supervisor", "Manager"] and lead < ROLE_REQUIREMENTS[role_target]["min_lead"]:
        flags.append("Below minimum leadership for target role")
    if skill < 55:
        flags.append("Low skill match")
    if test < 55:
        flags.append("Low technical test score")
    if interview < 55:
        flags.append("Low interview score")
    if salary_flag:
        flags.append(salary_flag)
    if raise_pct is not None and raise_pct > 50:
        flags.append("Large expected raise vs current salary")

    # Decision thresholds
    if total >= 86:
        decision = "Strong Hire"
    elif total >= 72:
        decision = "Hire"
    elif total >= 58:
        decision = "Consider"
    else:
        decision = "No Hire"

    breakdown = {
        "experience_score": round(exp_score, 1),
        "leadership_score": round(lead_score, 1),
        "education_score": round(edu, 1),
        "salary_fit_score": round(salary_fit, 1),
        "raise_score": round(raise_score, 1),
        "role_fit_penalty": role_fit_penalty,
    }

    aux = {
        "salary_fit_score": round(salary_fit, 1),
        "raise_pct": None if raise_pct is None else round(raise_pct, 1),
        "role_fit_label": fit_label,
        "role_gap": gap,
    }

    return total, decision, breakdown, flags, aux


# =========================================================
# GROQ EXPLANATION (bilingual)
# =========================================================
def groq_explain(payload: dict, score: float, decision: str, breakdown: dict, flags: list):
    if not GROQ_API_KEY:
        return TXT["no_key"]

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }

    if LANG == "Bahasa Indonesia":
        sys = "Kamu adalah HR expert yang objektif dan profesional. Jangan mengarang data yang tidak diberikan."
        prompt = f"""
Data kandidat:
- Nama: {payload["candidate_name"]}
- Target posisi: {payload["target_role"]}
- Role rekomendasi (auto-detect): {payload["recommended_role"]}
- Pengalaman: {payload["experience"]} tahun
- Leadership: {payload["leadership"]} tahun
- Skill match: {payload["skill_match"]}/100
- Tes teknis: {payload["test_score"]}/100
- Interview: {payload["interview_score"]}/100
- Pendidikan: {payload["education"]}
- Gaji saat ini: {payload["current_salary"]}
- Ekspektasi gaji: {payload["expected_salary"]}
- Budget: {payload["salary_budget"]}

Hasil rule-based:
- Skor: {score}
- Keputusan: {decision}
- Breakdown: {breakdown}
- Risk flags: {flags}

Tolong buat output dengan format:
1) Ringkasan keputusan (2-3 kalimat)
2) Kekuatan utama (bullet)
3) Risiko/concern utama (bullet)
4) Saran next step (1-2 kalimat; misal: negosiasi gaji, assign case study, panel interview, dsb)

Gunakan bahasa Indonesia yang profesional dan ringkas.
"""
    else:
        sys = "You are an objective, professional HR expert. Do not invent facts not provided."
        prompt = f"""
Candidate data:
- Name: {payload["candidate_name"]}
- Target role: {payload["target_role"]}
- Auto-detected recommended role: {payload["recommended_role"]}
- Experience: {payload["experience"]} years
- Leadership: {payload["leadership"]} years
- Skill match: {payload["skill_match"]}/100
- Technical test: {payload["test_score"]}/100
- Interview: {payload["interview_score"]}/100
- Education: {payload["education"]}
- Current salary: {payload["current_salary"]}
- Expected salary: {payload["expected_salary"]}
- Budget: {payload["salary_budget"]}

Rule-based result:
- Score: {score}
- Decision: {decision}
- Breakdown: {breakdown}
- Risk flags: {flags}

Return in this format:
1) Decision summary (2-3 sentences)
2) Key strengths (bullets)
3) Key risks/concerns (bullets)
4) Next step recommendation (1-2 sentences)

Keep it concise and professional.
"""

    body = {
        "model": GROQ_MODEL,
        "messages": [
            {"role": "system", "content": sys},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.25,
    }

    try:
        r = requests.post(GROQ_ENDPOINT, headers=headers, json=body, timeout=40)
        if r.status_code == 200:
            return r.json()["choices"][0]["message"]["content"]
        return f"{TXT['api_err']}: {r.text}"
    except Exception as e:
        return f"{TXT['api_err']}: {str(e)}"


# =========================================================
# SHORTLIST STATE
# =========================================================
if "shortlist" not in st.session_state:
    st.session_state.shortlist = []


def add_to_shortlist(row: dict):
    st.session_state.shortlist.append(row)


def clear_shortlist():
    st.session_state.shortlist = []


# =========================================================
# UI
# =========================================================
st.set_page_config(page_title=TXT["app_title"], page_icon="👔", layout="wide")

st.title("👔 " + TXT["app_title"])
st.caption(TXT["subtitle"])

st.sidebar.info(TXT["select_role_note"])

# Input area
st.subheader(TXT["candidate_block"])

c1, c2, c3 = st.columns([1.2, 1.0, 1.0])

with c1:
    candidate_name = st.text_input(TXT["candidate_name"], value="Candidate A")
    auto_role_on = st.checkbox(TXT["auto_role"], value=True)

with c2:
    target_role = st.selectbox(TXT["job_position"], ROLES, index=1)
    education = st.selectbox(TXT["education"], list(EDU_SCORE.keys()), index=2)

with c3:
    experience = st.slider(TXT["exp"], 0, 25, 4)
    leadership = st.slider(TXT["lead"], 0, 25, 1)

c4, c5, c6, c7 = st.columns(4)
with c4:
    skill_match = st.slider(TXT["skill"], 0, 100, 70)
with c5:
    test_score = st.slider(TXT["test"], 0, 100, 75)
with c6:
    interview_score = st.slider(TXT["interview"], 0, 100, 80)
with c7:
    # money inputs as float; user can put IDR numbers directly
    current_salary = st.number_input(TXT["curr_salary"], min_value=0.0, value=0.0, step=500000.0)
    expected_salary = st.number_input(TXT["exp_salary"], min_value=0.0, value=0.0, step=500000.0)

salary_budget = st.number_input(TXT["budget"], min_value=0.0, value=0.0, step=500000.0)

# Auto detect recommended role
recommended_role, conf, reco_explain = auto_detect_role(experience, leadership)

if auto_role_on:
    target_role_effective = recommended_role
else:
    target_role_effective = target_role

fit_lbl, gap = role_fit_label(target_role_effective, recommended_role)

# Buttons
b1, b2, b3 = st.columns([1, 1, 2])
with b1:
    do_eval = st.button(TXT["evaluate"], use_container_width=True)
with b2:
    do_add = st.button(TXT["add_shortlist"], use_container_width=True)
with b3:
    do_clear = st.button(TXT["clear_shortlist"], use_container_width=True)

if do_clear:
    clear_shortlist()
    st.success("Shortlist cleared." if LANG == "English" else "Shortlist dihapus.")

# Prepare payload
payload = {
    "candidate_name": candidate_name.strip() if candidate_name.strip() else "(anonymous)",
    "target_role": target_role_effective,
    "manual_target_role": target_role,
    "recommended_role": recommended_role,
    "role_confidence": conf,
    "experience": experience,
    "leadership": leadership,
    "skill_match": skill_match,
    "test_score": test_score,
    "interview_score": interview_score,
    "education": education,
    "current_salary": current_salary,
    "expected_salary": expected_salary,
    "salary_budget": salary_budget,
    "auto_role_enabled": auto_role_on,
}

# Evaluate scoring
score, decision, breakdown, flags, aux = score_candidate({
    "candidate_name": payload["candidate_name"],
    "target_role": payload["target_role"],
    "recommended_role": payload["recommended_role"],
    "experience": payload["experience"],
    "leadership": payload["leadership"],
    "skill_match": payload["skill_match"],
    "test_score": payload["test_score"],
    "interview_score": payload["interview_score"],
    "education": payload["education"],
    "current_salary": payload["current_salary"],
    "expected_salary": payload["expected_salary"],
    "salary_budget": payload["salary_budget"],
})

# Show role fit panel
st.divider()
rc1, rc2, rc3, rc4 = st.columns([1.2, 1, 1, 1.2])
rc1.metric(TXT["role_reco"], recommended_role)
rc2.metric(TXT["confidence"], f"{conf}%")
rc3.metric(TXT["role_fit"], aux["role_fit_label"])
rc4.write(
    f"**Target role used for scoring:** `{payload['target_role']}`\n\n"
    f"**Manual target role:** `{payload['manual_target_role']}`\n\n"
    f"**Auto-detect enabled:** `{payload['auto_role_enabled']}`"
)

# If evaluate pressed, show full result and AI explanation
if do_eval:
    st.subheader(TXT["result"])
    m1, m2, m3 = st.columns([1, 1, 2])
    m1.metric(TXT["score"], score)
    m2.metric(TXT["decision"], decision)

    # quick insights
    insights = []
    if aux["raise_pct"] is not None:
        insights.append(f"Raise%: {aux['raise_pct']}%")
    insights.append(f"Salary fit: {aux['salary_fit_score']}/100")
    insights.append(f"Role fit: {aux['role_fit_label']}")
    m3.write("**Signals:** " + " | ".join(insights))

    st.write("**Breakdown:**")
    st.json(breakdown)

    st.write(f"**{TXT['flags']}:**")
    st.write(flags if flags else ["(none)"])

    st.subheader(TXT["ai_explain"])
    with st.spinner("Generating explanation..."):
        explanation = groq_explain(payload, score, decision, breakdown, flags)
    st.write(explanation)

# Add to shortlist
if do_add:
    row = {
        "candidate_name": payload["candidate_name"],
        "target_role_used": payload["target_role"],
        "manual_target_role": payload["manual_target_role"],
        "recommended_role": payload["recommended_role"],
        "role_confidence": payload["role_confidence"],
        "experience": payload["experience"],
        "leadership": payload["leadership"],
        "skill_match": payload["skill_match"],
        "test_score": payload["test_score"],
        "interview_score": payload["interview_score"],
        "education": payload["education"],
        "current_salary": payload["current_salary"],
        "expected_salary": payload["expected_salary"],
        "salary_budget": payload["salary_budget"],
        "score": score,
        "decision": decision,
        "salary_fit_score": aux["salary_fit_score"],
        "raise_pct": aux["raise_pct"],
        "role_fit": aux["role_fit_label"],
        "role_gap": aux["role_gap"],
        "flags": "; ".join(flags) if flags else "",
    }
    add_to_shortlist(row)
    st.success("Added to shortlist." if LANG == "English" else "Ditambahkan ke shortlist.")

# Shortlist comparison + ranking
st.divider()
st.subheader(TXT["shortlist"])

if len(st.session_state.shortlist) == 0:
    st.info("Shortlist empty." if LANG == "English" else "Shortlist masih kosong.")
else:
    df = pd.DataFrame(st.session_state.shortlist)

    # Ranking system (tie-breakers)
    # 1) score desc
    # 2) salary_fit_score desc
    # 3) interview_score desc
    # 4) test_score desc
    df_ranked = df.sort_values(
        by=["score", "salary_fit_score", "interview_score", "test_score"],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)

    df_ranked.insert(0, "rank", df_ranked.index + 1)

    st.subheader(TXT["ranked"])

    # Show Top 3 highlight
    top_n = min(3, len(df_ranked))
    st.write(f"**{TXT['top3']}**")
    top_cards = st.columns(top_n)
    for i in range(top_n):
        r = df_ranked.iloc[i]
        with top_cards[i]:
            st.metric(f"#{int(r['rank'])} {r['candidate_name']}", f"{r['score']}")
            st.write(f"**{TXT['decision']}:** {r['decision']}")
            st.write(f"**{TXT['role_reco']}:** {r['recommended_role']}")
            st.write(f"**{TXT['role_fit']}:** {r['role_fit']}")
            st.write(f"Salary fit: {r['salary_fit_score']}/100")

    st.dataframe(df_ranked, use_container_width=True, hide_index=True)

    # Download CSV
    csv_bytes = df_ranked.to_csv(index=False).encode("utf-8")
    st.download_button(
        TXT["download_csv"],
        data=csv_bytes,
        file_name="shortlist_ranked.csv",
        mime="text/csv",
    )

import streamlit as st
import requests
import os

# =============================
# CONFIG — LOAD GROK API KEY
# =============================

# Support Streamlit Cloud secrets
if "GROK_API_KEY" in st.secrets:
    GROK_API_KEY = st.secrets["GROK_API_KEY"]
else:
    GROK_API_KEY = os.getenv("GROK_API_KEY")


# =============================
# RULE-BASED SCORING FUNCTION
# =============================

def score_candidate(experience, skill_match, test_score, interview_score):
    """
    Simple hiring scoring system
    """

    # Convert experience to score (max 100)
    experience_score = min(experience * 10, 100)

    # Use values directly
    skill_score = skill_match
    test_score = test_score
    interview_score = interview_score

    # Weighted average
    total_score = (
        0.25 * experience_score +
        0.25 * skill_score +
        0.25 * test_score +
        0.25 * interview_score
    )

    total_score = round(total_score, 2)

    # Decision logic
    if total_score >= 80:
        decision = "Strong Hire"
    elif total_score >= 65:
        decision = "Hire"
    elif total_score >= 50:
        decision = "Consider"
    else:
        decision = "No Hire"

    return total_score, decision


# =============================
# GROK EXPLANATION FUNCTION
# =============================

def grok_explain(candidate, score, decision):
    """
    Call Grok API to explain hiring decision
    """

    # Safe fallback if no API key
    if not GROK_API_KEY:
        return "⚠️ GROK_API_KEY not configured. Showing rule-based result only."

    url = "https://api.x.ai/v1/chat/completions"

    headers = {
        "Authorization": f"Bearer {GROK_API_KEY}",
        "Content-Type": "application/json"
    }

    prompt = f"""
You are an HR hiring assistant.

Candidate profile:
- Experience: {candidate['experience']} years
- Skill match: {candidate['skill_match']} / 100
- Technical test score: {candidate['test_score']} / 100
- Interview score: {candidate['interview_score']} / 100

Overall score: {score}
Decision: {decision}

Explain briefly why this candidate should or should not be hired.
Keep explanation professional and concise.
"""

    data = {
        "model": "grok-2-latest",
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],
        "temperature": 0.3
    }

    try:

        response = requests.post(
            url,
            headers=headers,
            json=data,
            timeout=30
        )

        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"]

        else:
            return f"⚠️ Grok API error: {response.status_code}"

    except Exception as e:
        return f"⚠️ Connection error: {str(e)}"


# =============================
# STREAMLIT UI
# =============================

st.set_page_config(
    page_title="Hiring Decision Optimizer",
    page_icon="👔",
    layout="centered"
)

st.title("👔 Hiring Decision Optimizer")

st.write("Rule-based hiring scoring with Grok AI explanation")

st.divider()

# Input fields

experience = st.slider(
    "Years of Experience",
    min_value=0,
    max_value=15,
    value=3
)

skill_match = st.slider(
    "Skill Match (%)",
    min_value=0,
    max_value=100,
    value=70
)

test_score = st.slider(
    "Technical Test Score",
    min_value=0,
    max_value=100,
    value=75
)

interview_score = st.slider(
    "Interview Score",
    min_value=0,
    max_value=100,
    value=80
)

st.divider()

# Evaluate button

if st.button("Evaluate Candidate"):

    # Calculate score
    score, decision = score_candidate(
        experience,
        skill_match,
        test_score,
        interview_score
    )

    st.subheader("Result")

    st.metric("Score", score)
    st.metric("Decision", decision)

    candidate = {
        "experience": experience,
        "skill_match": skill_match,
        "test_score": test_score,
        "interview_score": interview_score
    }

    st.divider()

    st.subheader("AI Explanation (Grok)")

    with st.spinner("Generating explanation..."):
        explanation = grok_explain(candidate, score, decision)

    st.write(explanation)


# =============================
# FOOTER
# =============================

st.divider()

st.caption("Hiring Decision Optimizer • Rule-based + Grok AI")

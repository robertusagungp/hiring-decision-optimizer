import streamlit as st
import requests
import os

# =============================
# LOAD GROQ API KEY
# =============================

# Streamlit Cloud secrets first
if "GROQ_API_KEY" in st.secrets:
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
else:
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")


# =============================
# RULE-BASED SCORING
# =============================

def score_candidate(experience, skill_match, test_score, interview_score):

    experience_score = min(experience * 10, 100)

    total_score = (
        0.25 * experience_score +
        0.25 * skill_match +
        0.25 * test_score +
        0.25 * interview_score
    )

    total_score = round(total_score, 2)

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
# GROQ LLM EXPLANATION
# =============================

def groq_explain(candidate, score, decision):

    if not GROQ_API_KEY:
        return "⚠️ GROQ_API_KEY not configured."

    url = "https://api.groq.com/openai/v1/chat/completions"

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
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

Explain briefly and professionally why this candidate should or should not be hired.
"""

    data = {
        "model": "llama3-70b-8192",
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
            return f"⚠️ Groq API error: {response.text}"

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

st.write("Rule-based hiring scoring with Groq AI explanation")

st.divider()

experience = st.slider(
    "Years of Experience",
    0, 15, 3
)

skill_match = st.slider(
    "Skill Match (%)",
    0, 100, 70
)

test_score = st.slider(
    "Technical Test Score",
    0, 100, 75
)

interview_score = st.slider(
    "Interview Score",
    0, 100, 80
)

st.divider()

if st.button("Evaluate Candidate"):

    score, decision = score_candidate(
        experience,
        skill_match,
        test_score,
        interview_score
    )

    st.subheader("Result")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Score", score)

    with col2:
        st.metric("Decision", decision)

    candidate = {
        "experience": experience,
        "skill_match": skill_match,
        "test_score": test_score,
        "interview_score": interview_score
    }

    st.divider()

    st.subheader("AI Explanation (Groq)")

    with st.spinner("Generating explanation..."):
        explanation = groq_explain(candidate, score, decision)

    st.write(explanation)


# =============================
# FOOTER
# =============================

st.divider()

st.caption("Powered by Groq • LLaMA3-70B")

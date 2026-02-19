import streamlit as st
import requests
import os
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY="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"

# -------------------------
# RULE BASED SCORING
# -------------------------

def score_candidate(experience, test_score, interview_score, skill_match):
    exp_score = min(experience * 10, 100)
    skill_score = skill_match
    test_score = test_score
    interview_score = interview_score

    total = (
        0.25 * exp_score +
        0.25 * skill_score +
        0.25 * test_score +
        0.25 * interview_score
    )

    if total >= 80:
        decision = "Strong Hire"
    elif total >= 65:
        decision = "Hire"
    elif total >= 50:
        decision = "Consider"
    else:
        decision = "No Hire"

    return round(total, 2), decision


# -------------------------
# GROK LLM EXPLANATION
# -------------------------

def grok_explain(candidate, score, decision):

    if not GROK_API_KEY:
        return "No GROK_API_KEY found."

    url = "https://api.x.ai/v1/chat/completions"

    headers = {
        "Authorization": f"Bearer {GROK_API_KEY}",
        "Content-Type": "application/json"
    }

    prompt = f"""
Candidate profile:
Experience: {candidate['experience']} years
Skill match: {candidate['skill_match']} / 100
Technical test: {candidate['test_score']} / 100
Interview score: {candidate['interview_score']} / 100

Score: {score}
Decision: {decision}

Explain hiring recommendation briefly.
"""

    data = {
        "model": "grok-2-latest",
        "messages": [
            {"role": "user", "content": prompt}
        ]
    }

    response = requests.post(url, headers=headers, json=data)

    if response.status_code == 200:
        return response.json()["choices"][0]["message"]["content"]

    return f"Grok API error: {response.text}"


# -------------------------
# STREAMLIT UI
# -------------------------

st.title("Hiring Decision Optimizer")

st.write("Rule-based hiring scoring + Grok explanation")

experience = st.slider("Years of experience", 0, 15, 3)

skill_match = st.slider("Skill match %", 0, 100, 70)

test_score = st.slider("Technical test score", 0, 100, 75)

interview_score = st.slider("Interview score", 0, 100, 80)

if st.button("Evaluate Candidate"):

    score, decision = score_candidate(
        experience,
        test_score,
        interview_score,
        skill_match
    )

    st.subheader("Result")

    st.write("Score:", score)
    st.write("Decision:", decision)

    candidate = {
        "experience": experience,
        "skill_match": skill_match,
        "test_score": test_score,
        "interview_score": interview_score
    }

    explanation = grok_explain(candidate, score, decision)

    st.subheader("Grok Explanation")

    st.write(explanation)

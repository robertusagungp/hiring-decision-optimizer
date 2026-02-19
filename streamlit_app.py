import streamlit as st
import requests
import os

# =============================
# LOAD GROQ API KEY
# =============================

if "GROQ_API_KEY" in st.secrets:
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
else:
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")


# =============================
# LANGUAGE SYSTEM
# =============================

LANGUAGES = {
    "English": {
        "title": "Hiring Decision Optimizer",
        "position": "Job Position",
        "experience": "Years of Experience",
        "skill": "Skill Match (%)",
        "test": "Technical Test Score",
        "interview": "Interview Score",
        "education": "Education Level",
        "current_salary": "Current Salary",
        "expected_salary": "Expected Salary",
        "salary_budget": "Salary Budget",
        "leadership": "Leadership Experience (years)",
        "evaluate": "Evaluate Candidate",
        "result": "Result",
        "score": "Score",
        "decision": "Decision",
        "explanation": "AI Explanation",
    },
    "Bahasa Indonesia": {
        "title": "Sistem Evaluasi Kandidat",
        "position": "Posisi Pekerjaan",
        "experience": "Pengalaman Kerja (tahun)",
        "skill": "Kecocokan Skill (%)",
        "test": "Skor Tes Teknis",
        "interview": "Skor Interview",
        "education": "Pendidikan",
        "current_salary": "Gaji Saat Ini",
        "expected_salary": "Ekspektasi Gaji",
        "salary_budget": "Budget Gaji",
        "leadership": "Pengalaman Leadership (tahun)",
        "evaluate": "Evaluasi Kandidat",
        "result": "Hasil",
        "score": "Skor",
        "decision": "Keputusan",
        "explanation": "Penjelasan AI",
    }
}


# =============================
# ROLE REQUIREMENTS
# =============================

ROLE_REQUIREMENTS = {

    "Junior": {
        "min_experience": 0,
        "ideal_experience": 2,
        "salary_multiplier": 1.2
    },

    "Officer": {
        "min_experience": 2,
        "ideal_experience": 4,
        "salary_multiplier": 1.3
    },

    "Senior": {
        "min_experience": 4,
        "ideal_experience": 7,
        "salary_multiplier": 1.4
    },

    "Supervisor": {
        "min_experience": 6,
        "ideal_experience": 10,
        "salary_multiplier": 1.5
    },

    "Manager": {
        "min_experience": 8,
        "ideal_experience": 15,
        "salary_multiplier": 1.6
    }
}


# =============================
# EDUCATION SCORE
# =============================

EDUCATION_SCORE = {
    "High School": 50,
    "Diploma": 65,
    "Bachelor": 75,
    "Master": 90,
    "PhD": 100
}


# =============================
# ADVANCED SCORING SYSTEM
# =============================

def score_candidate(data):

    role = data["position"]
    role_req = ROLE_REQUIREMENTS[role]

    experience = data["experience"]
    skill = data["skill"]
    test = data["test"]
    interview = data["interview"]
    education = EDUCATION_SCORE[data["education"]]
    leadership = data["leadership"]

    current_salary = data["current_salary"]
    expected_salary = data["expected_salary"]
    budget = data["budget"]

    # Experience score
    exp_score = min((experience / role_req["ideal_experience"]) * 100, 100)

    # Leadership score (important for higher roles)
    leadership_score = min((leadership / role_req["ideal_experience"]) * 100, 100)

    # Salary fit score
    if expected_salary <= budget:
        salary_score = 100
    else:
        salary_score = max(0, 100 - ((expected_salary - budget) / budget) * 100)

    # Salary increase realism
    if current_salary > 0:
        raise_percent = ((expected_salary - current_salary) / current_salary) * 100
        if raise_percent <= 30:
            raise_score = 100
        elif raise_percent <= 50:
            raise_score = 70
        else:
            raise_score = 40
    else:
        raise_score = 50

    total = (
        0.15 * exp_score +
        0.20 * skill +
        0.15 * test +
        0.15 * interview +
        0.10 * education +
        0.10 * leadership_score +
        0.10 * salary_score +
        0.05 * raise_score
    )

    total = round(total, 2)

    if total >= 85:
        decision = "Strong Hire"
    elif total >= 70:
        decision = "Hire"
    elif total >= 55:
        decision = "Consider"
    else:
        decision = "No Hire"

    return total, decision


# =============================
# GROQ EXPLANATION
# =============================

def groq_explain(data, score, decision, language):

    if not GROQ_API_KEY:
        return "GROQ_API_KEY not configured"

    url = "https://api.groq.com/openai/v1/chat/completions"

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    if language == "Bahasa Indonesia":
        prompt = f"""
Anda adalah HR expert.

Data kandidat:
Posisi: {data['position']}
Pengalaman: {data['experience']} tahun
Skill match: {data['skill']}
Test score: {data['test']}
Interview score: {data['interview']}
Pendidikan: {data['education']}
Gaji saat ini: {data['current_salary']}
Ekspektasi gaji: {data['expected_salary']}
Budget: {data['budget']}

Skor: {score}
Keputusan: {decision}

Berikan penjelasan profesional.
"""
    else:

        prompt = f"""
You are an HR expert.

Candidate data:
Position: {data['position']}
Experience: {data['experience']}
Skill match: {data['skill']}
Test score: {data['test']}
Interview score: {data['interview']}
Education: {data['education']}
Current salary: {data['current_salary']}
Expected salary: {data['expected_salary']}
Budget: {data['budget']}

Score: {score}
Decision: {decision}

Explain professionally.
"""

    data_api = {
        "model": "llama-3.3-70b-versatile",
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.3
    }

    r = requests.post(url, headers=headers, json=data_api)

    if r.status_code == 200:
        return r.json()["choices"][0]["message"]["content"]

    return r.text


# =============================
# UI
# =============================

language = st.selectbox("Language / Bahasa", ["English", "Bahasa Indonesia"])

T = LANGUAGES[language]

st.title(T["title"])

position = st.selectbox(
    T["position"],
    ["Junior", "Officer", "Senior", "Supervisor", "Manager"]
)

experience = st.slider(T["experience"], 0, 20, 3)

skill = st.slider(T["skill"], 0, 100, 70)

test = st.slider(T["test"], 0, 100, 75)

interview = st.slider(T["interview"], 0, 100, 80)

education = st.selectbox(
    T["education"],
    list(EDUCATION_SCORE.keys())
)

leadership = st.slider(T["leadership"], 0, 20, 0)

current_salary = st.number_input(T["current_salary"], 0)

expected_salary = st.number_input(T["expected_salary"], 0)

budget = st.number_input(T["salary_budget"], 0)


if st.button(T["evaluate"]):

    data = {
        "position": position,
        "experience": experience,
        "skill": skill,
        "test": test,
        "interview": interview,
        "education": education,
        "leadership": leadership,
        "current_salary": current_salary,
        "expected_salary": expected_salary,
        "budget": budget
    }

    score, decision = score_candidate(data)

    st.subheader(T["result"])

    st.metric(T["score"], score)
    st.metric(T["decision"], decision)

    st.subheader(T["explanation"])

    explanation = groq_explain(data, score, decision, language)

    st.write(explanation)

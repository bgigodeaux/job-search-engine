# webapp/pages/3_Jobs.py
import asyncio
import html
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import streamlit as st


# ---------- Paths ----------
HERE = Path(__file__).resolve()
REPO_ROOT = HERE.parents[1] if (HERE.parents[1] / "app").exists() else HERE.parents[2]
DATA_FILE = REPO_ROOT / "app" / "data" / "raw_jobs.json"


# ---------- Data helpers ----------
def load_jobs() -> List[Dict[str, Any]]:
    if not DATA_FILE.exists():
        return []
    try:
        return json.loads(DATA_FILE.read_text(encoding="utf-8"))
    except Exception:
        return []


def save_jobs(jobs: List[Dict[str, Any]]) -> None:
    DATA_FILE.parent.mkdir(parents=True, exist_ok=True)
    DATA_FILE.write_text(json.dumps(jobs, indent=2, ensure_ascii=False), encoding="utf-8")


# ---------- Feature renderer (simple & robust) ----------
def render_job_features_simple(feats: Any) -> None:
    """Render features with keys:
    - extracted_skills: List[str]
    - seniority_level: str
    - required_experience_years: int/float
    - job_summary_for_embedding: str
    """
    # Convert Pydantic model -> dict if needed
    if hasattr(feats, "model_dump"):
        feats = feats.model_dump()
    elif hasattr(feats, "dict"):
        feats = feats.dict()

    summary = feats.get("job_summary_for_embedding") or feats.get("job_summary") or ""
    seniority = feats.get("seniority_level", "N/A")
    years = feats.get("required_experience_years")
    skills = feats.get("extracted_skills", []) or []

    # Light CSS for chips/cards
    st.markdown(
        """
        <style>
          .chip {background:#1E88E5;color:#fff;padding:3px 8px;margin:2px;
                 border-radius:12px;font-size:12px;display:inline-block;}
          .card {background:#fff;border:1px solid #eee;border-radius:12px;
                 padding:14px 16px;margin:8px 0;}
        </style>
        """,
        unsafe_allow_html=True,
    )

    if summary:
        st.markdown("### Summary")
        st.markdown(f"> {summary}")

    c1, c2 = st.columns(2)
    c1.metric("Seniority", seniority)
    c2.metric("Required experience", f"{years} yrs" if years is not None else "N/A")

    st.markdown("### Skills")
    if skills:
        chips = " ".join(f"<span class='chip'>{html.escape(str(s))}</span>" for s in skills)
        st.markdown(chips, unsafe_allow_html=True)
    else:
        st.caption("No skills extracted.")

    with st.expander("Raw features JSON"):
        st.json(feats, expanded=False)


# ---------- Services (lazy import for stability on reruns) ----------
@st.cache_resource(show_spinner=False)
def get_services():
    from app.service.llm_manager import LLMManager
    from app.model.schemas import RawJob

    return LLMManager(), RawJob


# ---------- Page ----------
st.set_page_config(page_title="Jobs", layout="wide")
st.title("🧩 Job Management")

# Bootstrap session state
if "jobs" not in st.session_state:
    st.session_state.jobs = load_jobs()

llm_manager, RawJob = get_services()

# Sidebar: select or create job
job_titles = [j.get("job_title", "Untitled") for j in st.session_state.jobs]
choices = ["— Create New Job —"] + job_titles
picked = st.sidebar.selectbox("Select a job", choices)

if picked == "— Create New Job —":
    job_data: Dict[str, Any] = {}
    current_idx = -1
else:
    current_idx = job_titles.index(picked)
    job_data = dict(st.session_state.jobs[current_idx])

# Form
with st.form("job_form"):
    st.subheader("Job Details")
    colA, colB, colC = st.columns([1.2, 1, 1])
    title = colA.text_input("Job Title", value=job_data.get("job_title", ""))
    company = colB.text_input("Company Name", value=job_data.get("company_name", ""))
    location = colC.text_input("Location", value=job_data.get("location", ""))

    description = st.text_area(
        "Job Description",
        value=job_data.get("job_description", ""),
        height=220,
    )
    skills_csv = st.text_input(
        "Required Skills (comma-separated)",
        value=", ".join(job_data.get("required_skills", [])),
        help="e.g., Python, FastAPI, Docker",
    )

    c1, c2, c3 = st.columns(3)
    analyze_btn = c1.form_submit_button("🔎 Analyze Job (AI)", type="primary")
    save_btn = c2.form_submit_button("💾 Save Job")
    delete_btn = c3.form_submit_button("🗑️ Delete", disabled=(current_idx == -1))

# Actions
if analyze_btn:
    try:
        payload = {
            "job_title": title,
            "company_name": company,
            "location": location,
            "job_description": description,
            "required_skills": [s.strip() for s in skills_csv.split(",") if s.strip()],
            "budget": job_data.get("budget", {}),
        }
        raw_job = RawJob(**payload)
        with st.spinner("Generating engineered features…"):
            feats = asyncio.run(llm_manager.generate_job_features(raw_job))
        if feats:
            st.success("Features generated")
            render_job_features_simple(feats)
        else:
            st.error("Failed to generate features for this job.")
    except Exception as e:
        st.error(f"AI error: {e}")

if save_btn:
    updated = {
        "job_title": title,
        "company_name": company,
        "location": location,
        "job_description": description,
        "required_skills": [s.strip() for s in skills_csv.split(",") if s.strip()],
        "budget": job_data.get("budget", {}),
        "employment_type": job_data.get("employment_type", ""),
    }
    if current_idx == -1:
        st.session_state.jobs.append(updated)
    else:
        st.session_state.jobs[current_idx] = updated
    save_jobs(st.session_state.jobs)
    st.success("Job saved ✅")

if delete_btn and current_idx != -1:
    del st.session_state.jobs[current_idx]
    save_jobs(st.session_state.jobs)
    st.success("Job deleted 🗑️")
    st.stop()  # simple refresh after delete

# Preview (simple card)
st.markdown("### Preview")
st.write(f"**{title or 'Untitled'}** @ {company or '—'}  •  {location or '—'}")
if description:
    st.caption(description)
reqs = [s.strip() for s in skills_csv.split(",") if s.strip()]
if reqs:
    st.markdown("**Required skills:** " + ", ".join(reqs))

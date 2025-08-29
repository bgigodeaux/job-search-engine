# webapp/pages/4_Search.py
import asyncio
import html
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import streamlit as st


# ---------- Paths ----------
HERE = Path(__file__).resolve()
REPO_ROOT = HERE.parents[1] if (HERE.parents[1] / "app").exists() else HERE.parents[2]
DATA_DIR = REPO_ROOT / "app" / "data"
JOBS_FILE = DATA_DIR / "raw_jobs.json"
PROCESSED_CANDS_FILE = DATA_DIR / "processed_candidates.json"


# ---------- Small helpers ----------
def load_jobs() -> List[Dict[str, Any]]:
    if not JOBS_FILE.exists():
        return []
    try:
        return json.loads(JOBS_FILE.read_text(encoding="utf-8"))
    except Exception:
        return []


def tag_chip(text: str, color: str = "#1E88E5") -> str:
    t = html.escape(str(text))
    return (
        f"<span style='background:{color};color:#fff;padding:3px 8px;margin:2px;"
        f"border-radius:12px;font-size:12px;display:inline-block'>{t}</span>"
    )


def render_tags(tags: Optional[List[str]], color: str = "#1E88E5") -> None:
    if not tags:
        return
    st.markdown(" ".join(tag_chip(t, color) for t in tags), unsafe_allow_html=True)


def safe(obj: Any, *keys, default=None):
    cur = obj
    for k in keys:
        if cur is None:
            return default
        if isinstance(cur, dict):
            cur = cur.get(k, default)
        else:
            cur = getattr(cur, k, default)
    return cur


def kvtbl(
    d: Dict[str, Any], keys: List[str], labels: Optional[Dict[str, str]] = None
) -> pd.DataFrame:
    labels = labels or {}
    rows = []
    for k in keys:
        rows.append({"Field": labels.get(k, k), "Value": d.get(k, "")})
    return pd.DataFrame(rows)


def list_of_dicts_to_df(
    items: Optional[List[Dict[str, Any]]], cols: List[str], labels: Optional[Dict[str, str]] = None
) -> pd.DataFrame:
    labels = labels or {}
    if not items:
        return pd.DataFrame(columns=[labels.get(c, c) for c in cols])
    df = pd.DataFrame(
        [{c: (item.get(c, "") if isinstance(item, dict) else "") for c in cols} for item in items]
    )
    df.rename(columns={c: labels.get(c, c) for c in cols}, inplace=True)
    return df


# ---------- Services ----------
@st.cache_resource(show_spinner=False)
def get_search_service():
    # Import lazily to avoid path/rerun hiccups
    from app.service.llm_manager import LLMManager
    from app.service.search_service import SearchService
    from app.model.schemas import RawJob

    llm = LLMManager()
    service = SearchService(llm_manager=llm, processed_candidates_path=PROCESSED_CANDS_FILE)
    return service, RawJob


# ---------- Page ----------
st.set_page_config(page_title="Search Engine", layout="wide")
st.title("🔎 Search Engine — Find Top Candidates")

# Light CSS
st.markdown(
    """
    <style>
      .card {background:#fff;border:1px solid #eee;border-radius:12px;padding:14px 16px;margin:12px 0;}
      .muted {color:#6b7280;}
      .scorebar {background:#f1f5f9;border-radius:8px; height:10px; width:100%; overflow:hidden;}
      .scorefill {background:#22c55e; height:10px;}
      .rowline {display:flex; gap:16px; flex-wrap:wrap;}
    </style>
    """,
    unsafe_allow_html=True,
)

# Bootstrap jobs in session
if "jobs" not in st.session_state:
    st.session_state.jobs = load_jobs()

service, RawJob = get_search_service()

# Guards
if not st.session_state.jobs:
    st.warning("No jobs found. Please create a job in the **Jobs** page first.")
    st.stop()

if not PROCESSED_CANDS_FILE.exists():
    st.warning(
        "Processed candidate data not found. Please generate **app/data/processed_candidates.json** first."
    )
    st.stop()

# ---------- Sidebar controls ----------
job_labels = [
    f"{j.get('job_title', 'Untitled')} @ {j.get('company_name', 'N/A')}"
    for j in st.session_state.jobs
]
with st.sidebar:
    st.header("Search Settings")
    job_idx = st.selectbox(
        "Select a job", range(len(job_labels)), format_func=lambda i: job_labels[i]
    )
    top_n = st.slider("Top N candidates", min_value=10, max_value=200, value=100, step=10)
    run_btn = st.button("🚀 Find Matches", use_container_width=True)

# ---------- Run search ----------
if run_btn:
    selected_job = st.session_state.jobs[job_idx]
    raw_job_model = RawJob(**selected_job)

    # --- Job requirements at the top ---
    st.subheader("📋 Job Requirements")
    jr_cols = st.columns([1.2, 1, 1])
    jr_cols[0].markdown(f"**Title:** {selected_job.get('job_title','—')}")
    jr_cols[1].markdown(f"**Company:** {selected_job.get('company_name','—')}")
    jr_cols[2].markdown(f"**Location:** {selected_job.get('location','—')}")
    if selected_job.get("required_skills"):
        st.caption("Required skills")
        render_tags(selected_job.get("required_skills", []), color="#8E24AA")
    if selected_job.get("job_description"):
        st.caption("Description")
        st.write(selected_job.get("job_description"))

    st.markdown("---")

    with st.spinner("Running hybrid search…"):
        ranked_candidates = asyncio.run(service.find_top_candidates(raw_job_model, top_n=top_n))

    if not ranked_candidates:
        st.info("No suitable candidates found after filtering.")
        st.stop()

    st.subheader(f"Top {len(ranked_candidates)} Matches")

    # Render each candidate card
    for i, rc in enumerate(ranked_candidates, 1):
        score = float(safe(rc, "score", default=0.0) or 0.0)
        cand = safe(rc, "candidate", default=None)
        feats = safe(cand, "engineered_features", default=None)
        orig = safe(cand, "original_data", default=None)

        first = safe(orig, "first_name", default="") or ""
        last = safe(orig, "last_name", default="") or ""
        full_name = f"{first} {last}".strip() or f"Candidate #{i}"

        recent_title = safe(feats, "recent_job_title", default="—")
        recent_company = safe(feats, "recent_company", default="—")
        years = safe(feats, "total_years_of_experience", default=None)
        skills = safe(feats, "skill_keywords", default=[]) or []

        # Card header
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown(f"### {i}. {html.escape(full_name)}")

        # Score line + bar
        pct = f"{score:.2%}"
        st.markdown(f"**Semantic Match Score:** {pct}")
        fill_pct = max(0, min(int(score * 100), 100))
        st.markdown(
            f"<div class='scorebar'><div class='scorefill' style='width:{fill_pct}%;'></div></div>",
            unsafe_allow_html=True,
        )

        # Quick facts row
        cols = st.columns(3)
        cols[0].markdown(f"**Recent:** {html.escape(str(recent_title))}")
        cols[1].markdown(f"**Company:** {html.escape(str(recent_company))}")
        cols[2].markdown(
            f"**Experience:** {years:.1f} yrs"
            if isinstance(years, (int, float))
            else "**Experience:** —"
        )

        # Skills (engineered)
        st.caption("Skills (engineered)")
        render_tags(skills, color="#1E88E5")

        # -------- Easy-to-read full profile (tables) --------
        st.markdown("#### Full Profile")

        # 1) Profile key/value
        profile_kv = {
            "Name": full_name,
            "Email": safe(orig, "email", default=""),
            "Phone": safe(orig, "phone", default=""),
            "Age": safe(orig, "age", default=""),
            "Birthdate": safe(orig, "birthdate", default=""),
            "Address": safe(orig, "address", default=""),
        }
        st.table(pd.DataFrame(list(profile_kv.items()), columns=["Field", "Value"]))

        # 2) Experiences table
        exp_cols = ["role", "company", "start_date", "end_date", "description"]
        exp_labels = {
            "role": "Role",
            "company": "Company",
            "start_date": "Start",
            "end_date": "End",
            "description": "Description",
        }
        exp_df = list_of_dicts_to_df(safe(orig, "experiences", default=[]), exp_cols, exp_labels)
        st.markdown("**Experiences**")
        st.dataframe(exp_df, use_container_width=True)

        # 3) Education table
        edu_cols = ["degree", "institution", "year_of_graduation", "description"]
        edu_labels = {
            "degree": "Degree",
            "institution": "Institution",
            "year_of_graduation": "Year",
            "description": "Description",
        }
        edu_df = list_of_dicts_to_df(safe(orig, "education", default=[]), edu_cols, edu_labels)
        st.markdown("**Education**")
        st.dataframe(edu_df, use_container_width=True)

        # 4) Engineered features table
        feats_summary = {
            "Recent Title": recent_title,
            "Recent Company": recent_company,
            "Years of Experience": years if years is not None else "",
            "Seniority": safe(feats, "seniority_level", default=""),
            "Education Level": safe(feats, "education_level", default=""),
            "Skill Keywords": ", ".join(skills),
        }
        st.markdown("**Engineered Features**")
        st.table(pd.DataFrame(list(feats_summary.items()), columns=["Field", "Value"]))

        st.markdown("</div>", unsafe_allow_html=True)
else:
    st.info("Pick a job on the left, set `Top N`, then click **Find Matches**.")

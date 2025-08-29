# webapp/2_Candidates.py
import json
from pathlib import Path
from typing import List, Dict, Any

import pandas as pd
import streamlit as st


# ---------- Configuration & Paths ----------
HERE = Path(__file__).resolve()
# Works whether this file is under webapp/ or webapp/pages/
if (HERE.parents[1] / "app").exists():
    REPO_ROOT = HERE.parents[1]
else:
    REPO_ROOT = HERE.parents[2]
DATA_DIR = REPO_ROOT / "app" / "data"
DATA_FILE = DATA_DIR / "processed_candidates.json"


# ---------- Small UI helpers ----------
def render_tags(tag_list: List[str], color: str = "#1E88E5"):
    """Render a list of tags inline with lightweight styling."""
    if not tag_list:
        return
    tags_html = " ".join(
        [
            f"<span style='background-color:{color}; color:white; padding:3px 8px; "
            f"margin:2px; border-radius:12px; font-size:12px; display:inline-block;'>{t}</span>"
            for t in tag_list
        ]
    )
    st.markdown(tags_html, unsafe_allow_html=True)


def safe_get(d: Dict[str, Any], path: List[str], default=None):
    cur = d
    for k in path:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


# ---------- Data loading ----------
@st.cache_data(show_spinner=False)
def load_candidates() -> List[Dict[str, Any]]:
    if not DATA_FILE.exists():
        return []
    with open(DATA_FILE, "r") as f:
        data = json.load(f)
    # File can be either a list of candidates or a single dict with keys. Normalize to list.
    if isinstance(data, dict) and "original_data" in data:
        return [data]
    if isinstance(data, list):
        return data
    return []


# ---------- Page ----------
st.set_page_config(page_title="Candidates", layout="wide")
st.title("👥 Candidates")

cands = load_candidates()
if not cands:
    st.warning(
        f"No data found. Put a processed file at: `{DATA_FILE}`\n\n"
        "Expected structure per item: { original_data: {...}, engineered_features: {...}, embedding: [...] }"
    )
    st.stop()

# ---------- Build quick filter facets ----------
all_engineered_skills = sorted(
    {
        skill
        for c in cands
        for skill in (safe_get(c, ["engineered_features", "skill_keywords"], []) or [])
    }
)
all_seniority = sorted(
    {
        str(safe_get(c, ["engineered_features", "seniority_level"], ""))
        for c in cands
        if safe_get(c, ["engineered_features", "seniority_level"], "")
    }
)

# Sidebar filters
with st.sidebar:
    st.header("Filters")
    q = st.text_input("Search name/email/company/role")
    min_years = st.slider("Min. experience (years)", 0.0, 30.0, 0.0, 0.5)
    chosen_seniority = st.multiselect("Seniority", options=all_seniority, default=[])
    chosen_skills = st.multiselect(
        "Engineered skills include", options=all_engineered_skills, default=[]
    )
    skill_mode = st.radio("Skill filter mode", ["Any", "All"], horizontal=True)


# ---------- Filter logic ----------
def matches_query(c: Dict[str, Any], q: str) -> bool:
    if not q:
        return True
    ql = q.lower()
    raw = c.get("original_data", {})
    text_blobs = [
        raw.get("first_name", ""),
        raw.get("last_name", ""),
        raw.get("email", ""),
        raw.get("address", ""),
    ]
    for exp in raw.get("experiences", []) or []:
        text_blobs += [exp.get("company", ""), exp.get("role", ""), exp.get("description", "")]
    return any(ql in (t or "").lower() for t in text_blobs)


def matches_years(c: Dict[str, Any], min_yrs: float) -> bool:
    yrs = safe_get(c, ["engineered_features", "total_years_of_experience"], 0.0) or 0.0
    try:
        return float(yrs) >= float(min_yrs)
    except Exception:
        return False


def matches_seniority(c: Dict[str, Any], sels: List[str]) -> bool:
    if not sels:
        return True
    s = str(safe_get(c, ["engineered_features", "seniority_level"], "")) or ""
    return s in set(sels)


def matches_skills(c: Dict[str, Any], skills: List[str], mode: str) -> bool:
    if not skills:
        return True
    have = set(safe_get(c, ["engineered_features", "skill_keywords"], []) or [])
    want = set(skills)
    return (have & want) != set() if mode == "Any" else want.issubset(have)


filtered = [
    c
    for c in cands
    if matches_query(c, q)
    and matches_years(c, min_years)
    and matches_seniority(c, chosen_seniority)
    and matches_skills(c, chosen_skills, skill_mode)
]


# ---------- List view ----------
def row_for_table(c: Dict[str, Any]) -> Dict[str, Any]:
    raw = c.get("original_data", {})
    feats = c.get("engineered_features", {}) or {}
    name = f"{raw.get('first_name','')} {raw.get('last_name','')}".strip()
    top_sk = ", ".join((feats.get("skill_keywords") or [])[:5])
    return {
        "Name": name or "N/A",
        "Email": raw.get("email", ""),
        "Years": feats.get("total_years_of_experience", ""),
        "Seniority": feats.get("seniority_level", ""),
        "Top skills": top_sk,
    }


table = pd.DataFrame([row_for_table(c) for c in filtered])
st.caption(f"Showing {len(filtered)} of {len(cands)} candidates")
if not table.empty:
    st.dataframe(table, use_container_width=True, hide_index=True)
else:
    st.info("No candidates match your filters.")

# ---------- Candidate picker ----------
options = [
    f"{safe_get(c, ['original_data','first_name'],'')} {safe_get(c, ['original_data','last_name'],'')}".strip()
    or f"Candidate #{i+1}"
    for i, c in enumerate(filtered)
]
if not options:
    st.stop()

selected_idx = st.selectbox(
    "Select a candidate", range(len(filtered)), format_func=lambda i: options[i]
)
candidate = filtered[selected_idx]
raw = candidate.get("original_data", {}) or {}
feats = candidate.get("engineered_features", {}) or {}

# ---------- Detail: side-by-side ----------
left, right = st.columns(2, gap="large")

with left:
    st.subheader("📄 Candidate Details (Raw)")
    st.markdown(f"**Name:** {raw.get('first_name','')} {raw.get('last_name','')}")
    cols_a = st.columns(3)
    cols_a[0].markdown(f"**Email:** {raw.get('email','N/A')}")
    cols_a[1].markdown(f"**Phone:** {raw.get('phone','N/A')}")
    cols_a[2].markdown(f"**Age:** {raw.get('age','?')}")
    st.markdown(f"**Address:** {raw.get('address','N/A')}")

    st.markdown("#### Skills")
    render_tags(raw.get("skills", []), color="#1E88E5")

    st.markdown("#### Experience")
    experiences = raw.get("experiences", []) or []
    if not experiences:
        st.caption("No experience listed.")
    for exp in experiences:
        st.markdown(
            f"- **{exp.get('role','')}** at *{exp.get('company','')}* "
            f"({exp.get('start_date','?')} – {exp.get('end_date','Present')})"
        )
        if exp.get("description"):
            st.caption(exp["description"])

    st.markdown("#### Education")
    education = raw.get("education", []) or []
    if not education:
        st.caption("No education listed.")
    for edu in education:
        line = f"- **{edu.get('degree','')}**, {edu.get('institution','')}"
        year = edu.get("year_of_graduation")
        if year:
            line += f" ({year})"
        st.markdown(line)
        if edu.get("description"):
            st.caption(edu["description"])

with right:
    st.subheader("🤖 Engineered Features")
    summary = feats.get("candidate_summary", "")
    if summary:
        st.markdown(f"> {summary}")

    cols_b = st.columns(3)
    yrs = feats.get("total_years_of_experience", "N/A")
    cols_b[0].metric("Experience", f"{yrs} yrs" if yrs != "N/A" else "N/A")
    cols_b[1].metric("Seniority", feats.get("seniority_level", "N/A"))
    cols_b[2].metric("Education Level", feats.get("education_level", "N/A"))

    st.markdown("#### Skills (Engineered)")
    render_tags(feats.get("skill_keywords", []), color="#43A047")

    # Compare raw vs engineered skills
    raw_sk = set(raw.get("skills", []) or [])
    eng_sk = set(feats.get("skill_keywords", []) or [])
    if raw_sk or eng_sk:
        st.markdown("#### Skill Overlap")
        common = sorted(raw_sk & eng_sk)
        added = sorted(eng_sk - raw_sk)
        missing = sorted(raw_sk - eng_sk)
        st.write("**Common**")
        render_tags(common, color="#546E7A")
        st.write("**Added by AI**")
        render_tags(added, color="#8E24AA")
        st.write("**Only in Raw**")
        render_tags(missing, color="#90A4AE")

# Optional raw JSON expanders
with st.expander("Raw JSON: original_data", expanded=False):
    st.json(raw, expanded=False)
with st.expander("Raw JSON: engineered_features", expanded=False):
    st.json(feats, expanded=False)

import asyncio
import json
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from app.service.llm_manager import LLMManager
from app.service.search_service import SearchService
from app.model.schemas import RawJob, RawCandidate


# --- Path & Environment ---
PROJECT_ROOT = Path(__file__).resolve().parents[1]  # repo root
DATA_DIR = PROJECT_ROOT / "app" / "data"
load_dotenv(PROJECT_ROOT / ".env")

# --- Page Configuration ---
st.set_page_config(page_title="AI Recommendation Engine", layout="wide")
st.title("🧠 AI Recommendation Engine")


# --- Service Initialization (with Caching) ---
@st.cache_resource
def get_services():
    try:
        llm_manager = LLMManager()
        search_service = SearchService(
            llm_manager,
            processed_candidates_path=DATA_DIR / "processed_candidates.json",
        )
        return llm_manager, search_service, RawJob, RawCandidate
    except ImportError as e:
        st.error(
            f"Failed to import a required module: {e}. "
            "Please ensure you run Streamlit from the project root directory and all dependencies are installed."
        )
    except Exception as e:
        st.error(f"An error occurred during service initialization: {e}")
    return None, None, None, None


llm_manager, search_service, RawJob, RawCandidate = get_services()


# --- Data Loading and State Management ---
def load_data():
    if "jobs" not in st.session_state:
        try:
            with open(DATA_DIR / "raw_jobs.json", "r") as f:
                st.session_state.jobs = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            st.session_state.jobs = []

    if "candidates" not in st.session_state:
        try:
            with open(DATA_DIR / "processed_candidates.json", "r") as f:
                st.session_state.candidates = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            st.session_state.candidates = []


def save_jobs():
    try:
        with open(DATA_DIR / "raw_jobs.json", "w") as f:
            json.dump(st.session_state.jobs, f, indent=2)
        st.success("Jobs saved successfully!")
    except IOError as e:
        st.error(f"Failed to save jobs: {e}")


load_data()


# --- Sidebar Navigation ---
with st.sidebar:
    st.header("Navigation")
    page = st.radio("Go to", ["Candidate Browsing", "Job Management", "Search Engine"])

import streamlit as st


def render_tags(tag_list, color="blue"):
    """Render a list of tags in a single line with some styling."""
    if not tag_list:
        return
    tags_html = " ".join(
        [
            f"<span style='background-color:{color}; color:white; padding:3px 8px; "
            f"margin:2px; border-radius:12px; font-size:12px;'>{t}</span>"
            for t in tag_list
        ]
    )
    st.markdown(tags_html, unsafe_allow_html=True)


# --- Page 1: Candidate Browsing ---
if page == "Candidate Browsing":
    st.header("Browse Candidate Profiles")

    if not st.session_state.candidates:
        st.warning(
            "No candidate data found. Please add 'processed_candidates.json' to the data directory."
        )
    else:
        # Candidate selector
        selected_idx = st.selectbox(
            "Select a candidate",
            range(len(st.session_state.candidates)),
            format_func=lambda i: f"{st.session_state.candidates[i]['original_data'].get('first_name','')} "
            f"{st.session_state.candidates[i]['original_data'].get('last_name','')}",
        )
        candidate = st.session_state.candidates[selected_idx]

        raw = candidate["original_data"]
        feats = candidate.get("engineered_features", {})

        # --- Two-column layout ---
        col1, col2 = st.columns(2)

        # LEFT COLUMN: Original Data
        with col1:
            st.markdown("### 📄 Candidate Details (Raw)")
            st.markdown(
                f"**Name:** {raw.get('first_name','')} {raw.get('last_name','')}"
            )
            st.markdown(f"**Email:** {raw.get('email','N/A')}")
            st.markdown(f"**Phone:** {raw.get('phone','N/A')}")
            st.markdown(f"**Address:** {raw.get('address','N/A')}")
            st.markdown(
                f"**Age:** {raw.get('age','?')}  |  **Birthdate:** {raw.get('birthdate','N/A')}"
            )

            st.markdown("#### Skills")
            render_tags(raw.get("skills", []), color="#1E88E5")

            st.markdown("#### Experience")
            for exp in raw.get("experiences", []):
                st.markdown(
                    f"- **{exp.get('role','')}** at *{exp.get('company','')}*  "
                    f"({exp.get('start_date','?')} – {exp.get('end_date','Present')})"
                )
                if exp.get("description"):
                    st.caption(exp["description"])

            st.markdown("#### Education")
            for edu in raw.get("education", []):
                st.markdown(
                    f"- **{edu.get('degree','')}**, {edu.get('institution','')} "
                    f"({edu.get('year_of_graduation','')})"
                )
                if edu.get("description"):
                    st.caption(edu["description"])

        # RIGHT COLUMN: Engineered Features
        with col2:
            st.markdown("### 🤖 Engineered Features")
            if feats:
                st.markdown(f"**Summary:** {feats.get('candidate_summary','')}")
                st.metric(
                    "Experience", f"{feats.get('total_years_of_experience','?')} yrs"
                )
                st.metric("Seniority", feats.get("seniority_level", "N/A"))
                st.metric("Education Level", feats.get("education_level", "N/A"))

                st.markdown("#### Skills (Engineered)")
                render_tags(feats.get("skill_keywords", []), color="#43A047")
            else:
                st.info("No engineered features available.")


# --- Page 2: Job Management ---
elif page == "Job Management":
    st.header("Create or Update Job Postings")

    job_titles = [j.get("job_title", "Untitled") for j in st.session_state.jobs]
    job_options = ["--- Create New Job ---"] + job_titles
    selected_job_title = st.selectbox(
        "Select a job to edit or create a new one", job_options
    )

    if selected_job_title == "--- Create New Job ---":
        job_data, current_job_index = {}, -1
    else:
        current_job_index = job_titles.index(selected_job_title)
        job_data = st.session_state.jobs[current_job_index]

    with st.form(key="job_form"):
        st.subheader("Job Details")
        title = st.text_input("Job Title", value=job_data.get("job_title", ""))
        company = st.text_input("Company Name", value=job_data.get("company_name", ""))
        location = st.text_input("Location", value=job_data.get("location", ""))
        description = st.text_area(
            "Job Description", value=job_data.get("job_description", ""), height=200
        )
        skills = st.text_input(
            "Required Skills (comma-separated)",
            value=", ".join(job_data.get("required_skills", [])),
        )

        col1, col2 = st.columns(2)
        analyze_button = col1.form_submit_button(
            label="Analyze Job (Feature Engineering)"
        )
        save_button = col2.form_submit_button(label="Save Job")

    if analyze_button:
        if llm_manager and RawJob:
            with st.spinner("Generating features using AI..."):
                form_job_data = {
                    "job_title": title,
                    "company_name": company,
                    "location": location,
                    "job_description": description,
                    "required_skills": [s.strip() for s in skills.split(",")],
                }
                raw_job_model = RawJob(**form_job_data)
                features = asyncio.run(llm_manager.generate_job_features(raw_job_model))
                if features:
                    st.subheader("Engineered Features")
                    st.json(features.model_dump())
                else:
                    st.error("Failed to generate features for the job.")
        else:
            st.error("AI Services not available.")

    if save_button:
        updated_job_data = {
            "job_title": title,
            "company_name": company,
            "location": location,
            "job_description": description,
            "required_skills": [s.strip() for s in skills.split(",")],
            "budget": job_data.get("budget", {}),
            "employment_type": job_data.get("employment_type", ""),
        }
        if current_job_index == -1:
            st.session_state.jobs.append(updated_job_data)
        else:
            st.session_state.jobs[current_job_index] = updated_job_data
        save_jobs()


# --- Page 3: Search Engine ---
elif page == "Search Engine":
    st.header("Find Top Candidates for a Job")

    if not st.session_state.jobs:
        st.warning(
            "No jobs found. Please create a job in the 'Job Management' page first."
        )
    elif search_service and not search_service.candidates:
        st.warning(
            "Processed candidate data not found. Please generate 'processed_candidates.json' first."
        )
    elif search_service and RawJob:
        job_options = [
            f"{j.get('job_title', 'Untitled')} @ {j.get('company_name', 'N/A')}"
            for j in st.session_state.jobs
        ]
        selected_job_idx = st.selectbox(
            "Select a job to find matches for",
            range(len(job_options)),
            format_func=lambda i: job_options[i],
        )

        if st.button("🔍 Find Top 100 Candidates"):
            selected_job = st.session_state.jobs[selected_job_idx]
            raw_job_model = RawJob(**selected_job)

            with st.spinner("Running hybrid search... This may take a moment."):
                ranked_candidates = asyncio.run(
                    search_service.find_top_candidates(raw_job_model)
                )

            if not ranked_candidates:
                st.info("No suitable candidates found after filtering.")
            else:
                st.subheader(f"Top {len(ranked_candidates)} Matches")
                for i, ranked_cand in enumerate(ranked_candidates, 1):
                    cand_features = ranked_cand.candidate.engineered_features
                    cand_original = ranked_cand.candidate.original_data

                    st.markdown(
                        f"""
                        ---
                        ### {i}. {cand_original.first_name} {cand_original.last_name}
                        **Semantic Match Score:** `{ranked_cand.score:.2%}`
                        """
                    )
                    st.text(
                        f"Title: {cand_features.recent_job_title} @ {cand_features.recent_company}"
                    )
                    st.text(
                        f"Experience: {cand_features.total_years_of_experience:.1f} years"
                    )
                    st.text(f"Skills: {', '.join(cand_features.skill_keywords)}")

                    with st.expander("View Full Profile"):
                        st.json(cand_original.model_dump())
    else:
        st.error(
            "Search Service is not available. Check logs for initialization errors."
        )

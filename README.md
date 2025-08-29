# 🧠 Manatal Recommendation Engine

This project is an **AI-powered recommendation engine** for matching job descriptions with top candidates.  
It combines **LLMs for feature engineering**, **hybrid search (semantic + keyword)**, and a **Streamlit webapp** for interactive exploration.

---

## 📂 Project Structure
├── app
│   ├── api/                  # FastAPI backend
│   ├── model/                # Pydantic schemas
│   ├── service/              # LLM + search services
│   ├── data/                 # Candidate & job JSON datasets
│   └── prompts/              # Prompt templates
├── webapp/                   # Streamlit frontend
│   ├── 1_👋_Welcome.py        # Landing page
│   ├── 2_Candidates.py       # Candidate browsing
│   ├── 3_Jobs.py             # Job management
│   └── 4_Search.py           # Candidate-job matching
├── requirements.txt
├── pyproject.toml / poetry.lock
└── README.md

---

## ⚡ Features
- Upload or manage **raw jobs** and **raw candidates**
- **LLM-driven feature engineering** (skills, seniority, summaries)
- **Hybrid candidate search** (semantic + keyword ranking)
- Interactive **Streamlit dashboard**
- REST API (FastAPI) for programmatic access

---

## 🛠️ Setup

### With Poetry
```bash
direnv reload
```
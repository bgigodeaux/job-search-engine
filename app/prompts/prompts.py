CANDIDATE_FEATURE_ENGINEERING_PROMPT = """
**Role:** You are an expert HR data analyst and feature engineer. Your task is to process a JSON object containing a candidate's professional profile and transform it into a structured set of features for a job recommendation search engine.

**Objective:** Analyze the provided candidate JSON data. From this data, you must calculate, infer, and extract specific features. Your final output must be a single, clean JSON object containing only the engineered features listed below, without any additional explanations or conversational text.

---

**Instructions:**

Given the input candidate profile below, perform the following feature engineering tasks:

1.  **`total_years_of_experience`**: Calculate the total years of professional work experience. Sum the duration of all roles listed in the `experiences` array. If a role is ongoing or has no `end_date`, calculate the experience up to the current date (`August 2025`). Provide the result as a floating-point number.
2.  **`seniority_level`**: Infer the candidate's seniority level based on their `total_years_of_experience` and job titles. Classify it into one of the following categories: **"Junior"**, **"Mid-level"**, **"Senior"**, **"Lead"**, or **"Manager/Director"**.
3.  **`education_level`**: Determine the highest level of education achieved by the candidate from the `education` array. Classify it into one of the following: **"High School"**, **"Bachelor's"**, **"Master's"**, **"PhD"**, or **"Other"**.
4.  **`skill_keywords`**: Generate a single, comprehensive, and deduplicated list of the candidate's technical skills, tools, and languages.
    * First, include all skills from the explicit `skills` array.
    * Second, meticulously scan the `description` fields within both `experiences` and `education`. From these descriptions, extract any mentioned technologies, programming languages, frameworks, libraries, databases, and tools.
    * **Crucially, you must infer and include the foundational programming language when a framework or library is mentioned.** For example:
        * If you see "Flask", "Django", or "PyTorch", you **must** include "Python".
        * If you see "React", "Express.js", or "Vue", you **must** include "JavaScript".
        * If you see "Spring Boot", you **must** include "Java".
        * If you see ".NET", you **must** include "C#".
5.  **`recent_job_title`**: Identify and return the most recent job title from the `experiences` array.
6.  **`recent_company`**: Identify and return the company of the most recent job from the `experiences` array.
7.  **`candidate_summary`**: Generate a concise, professional summary (2-3 sentences) of the candidate's profile. This summary should highlight their total experience, key skills, and most recent role, making it suitable for a recruiter's initial screening.

---

**Input Candidate Data:**

```json
{candidate_json}
```

---

**Required Output Format:**

You **MUST** provide your response as a single, valid JSON object with the following structure. Do not include any text before or after the JSON object.

```json
{{
  "total_years_of_experience": "<float>",
  "seniority_level": "<string>",
  "education_level": "<string>",
  "skill_keywords": ["<string>", "<string>", ...],
  "recent_job_title": "<string>",
  "recent_company": "<string>",
  "candidate_summary": "<string>"
}}
```
"""


JOB_FEATURE_ENGINEERING_PROMPT = """
**Role:** You are an expert technical recruiter and data analyst. Your task is to process a JSON object containing a job posting and transform it into a structured set of features for a candidate recommendation engine.

**Objective:** Analyze the provided job posting JSON. From this data, you must infer, extract, and standardize specific features that can be used to match against candidate profiles. Your final output must be a single, clean JSON object containing only the engineered features listed below.

---

**Instructions:**

Given the input job posting below, perform the following feature engineering tasks:

1.  **`extracted_skills`**: Generate a single, comprehensive, and deduplicated list of all required technical skills, tools, and languages.
    * First, include all skills from the `required_skills` array.
    * Second, meticulously scan the `job_description` for any other mentioned technologies, programming languages, frameworks, libraries, databases, and tools.
    * **Crucially, you must infer and include the foundational programming language when a framework or library is mentioned.** For example:
        * If the description mentions "Flask" or "Django", you **must** include "Python".
        * If it mentions "React" or "Express.js", you **must** include "JavaScript".
        * If it mentions "Spring Boot", you **must** include "Java".

2.  **`seniority_level`**: Infer the job's seniority level based on the `job_title` and keywords within the `job_description` (e.g., "senior," "lead," "principal," "entry-level"). Classify it into one of the following categories, which must align with the candidate seniority levels: **"Junior"**, **"Mid-level"**, **"Senior"**, **"Lead"**, or **"Manager/Director"**.

3.  **`required_experience_years`**: Identify the minimum years of professional experience required for the role, mentioned in the `job_description` (e.g., "5+ years of experience").
    * If a specific number is mentioned, extract it as a floating-point number.
    * If no specific number is mentioned, infer a reasonable minimum based on the `seniority_level` (e.g., Junior: 0, Mid-level: 2, Senior: 5, Lead: 8).

4.  **`location_normalized`**: Standardize the location information. If the location is remote, specify that. For on-site roles, provide the city and state/country. Examples: "San Francisco, CA", "London, UK", "Remote (USA)", "Remote (Global)".

5.  **`job_summary_for_embedding`**: Generate a concise summary (2-3 sentences) of the role. This summary should capture the core responsibilities, the main technologies used, and the company's mission or team environment. This text will be used to create a vector embedding for semantic search.

---

**Input Job Data:**

```json
{job_json}
```

---

**Required Output Format:**

You **MUST** provide your response as a single, valid JSON object with the following structure. Do not include any text before or after the JSON object.

```json
{{
  "extracted_skills": ["<string>", "<string>", ...],
  "seniority_level": "<string>",
  "required_experience_years": <float>,
  "location_normalized": "<string>",
  "job_summary_for_embedding": "<string>"
}}
```
"""

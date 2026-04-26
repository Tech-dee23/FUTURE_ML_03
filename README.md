# Task 3 – Resume / Candidate Screening System

## Overview
An ML‑based system that automatically screens and ranks resumes against a given job description.  
It highlights missing skills, helping recruiters shortlist the best‑fit candidates faster.

## Approach
- **Text Vectorisation**: TF‑IDF over job description + resumes.
- **Scoring**: Cosine similarity between job and each resume.
- **Skill Gap Detection**: Keyword matching against a list of required skills (Python, Django, REST, PostgreSQL, AWS, Docker).

## Results
- Candidates are ranked by relevance score (see `candidate_ranking.png`).
- Missing skills are clearly listed for each candidate.

### Example Ranking
| Candidate   | Score | Missing Skills           |
|-------------|-------|--------------------------|
| Candidate_A | 0.75  | None                     |
| Candidate_D | 0.68  | django                   |
| Candidate_C | 0.22  | python, django, rest, postgresql, aws, docker |
| ...         |       |                          |

## Business Value
- Reduces manual screening time.
- Identifies skill gaps instantly for up‑skilling or filtering.
- Transparent score allows recruiters to set a cut‑off threshold.

## Visual
![Candidate Ranking](candidate_ranking.png)

## Tools
Python, Pandas, Scikit‑learn, Matplotlib
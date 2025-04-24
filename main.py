from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional
import fitz  # PyMuPDF
import os
import google.generativeai as genai
import re
import uvicorn
# from dotenv import load_dotenv

# Load .env if available
# load_dotenv()

# Configure Gemini
genai.configure(api_key="AIzaSyBDoheyHjMBEbJAvE43khAD3-rMWHvJTrI")
for model in genai.list_models():
    print(model.name)


app = FastAPI()

# Allow frontend/devtools (optional)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 📄 Extract text from PDF or TXT
def extract_text(file: UploadFile) -> str:
    if file.filename.endswith(".pdf"):
        doc = fitz.open(stream=file.file.read(), filetype="pdf")
        return "\n".join([page.get_text() for page in doc])
    else:
        return file.file.read().decode("utf-8")

# 🧠 Gemini prompt & analysis
def get_analysis_from_gemini(transcript: str) -> dict:
    model = genai.GenerativeModel("models/gemini-2.0-flash")
#     prompt = f"""
# Given the following AI interview transcript, analyze the interviewee's **strengths** and **weaknesses** across both technical and behavioral areas.
# Also provide an overall summary of the **technical ability** of the interviewee based on their performance.

# Respond with a JSON object in the following format:

# {{
#   "technical_strengths": [...],
#   "technical_weaknesses": [...],
#   "behavioral_strengths": [...]
#   "behavioral_weaknesses": [...],
#   "technical_ability": "<summary>"
# }}

# Transcript:
# {transcript}
# """
    prompt = f"""
You are evaluating an AI-conducted interview transcript. Extract **only** the truly *applied* technical and behavioral strengths and weaknesses.

**Technical Strengths** must meet ALL:
1. **Applied Demonstration**: The candidate gives a concrete example or describes an action they actually took.
2. **Outcome or Reasoning**: They explain why they did it or what benefit/result it produced.
3. **Tool/Concept in Context**: They use a specific tool or concept in that example — not just name it.

If **no** applied technical strength is demonstrated, set:
"technical_strengths": "No applied technical strengths were demonstrated during the interview."

**Technical Weaknesses** must meet ALL:
1. **Failure to Apply**: Struggles to use or explain a concept when asked.
2. **Unclear or Vague**: Answers are imprecise or show confusion.
3. **Missed Core Areas**: Avoids or can’t handle essential parts of a topic.

If **no** applied technical weakness is observed, set:
"technical_weaknesses": "No clear technical weaknesses were demonstrated during the interview."

---

— **Behavioral Strengths**:
  1. If the candidate gives a real example of leadership, teamwork, conflict resolution, etc., treat that as an applied strength.
  2. Otherwise, infer from their communication style: clarity of explanation, responsiveness to questions, willingness to ask clarifying questions, calmness under pressure, adaptability, engagement, etc.
  3. Always tie your inference back to specific parts of the transcript (e.g., “They asked a clarifying question about loading times after the first technical question, showing engagement.”).

— **Behavioral Weaknesses**:
  1. If there’s an applied “I did X” story that shows a shortcoming, list it.
  2. Otherwise, infer from hesitations, repetition, defensiveness, lack of follow-up questions, over-reliance on one example, etc., citing transcript evidence.

---

Finally, add **two summary fields**:
- **technical_ability**: An evidence-based paragraph summarizing their overall technical capability (no fluff).
- **behavioral_ability**: An evidence-based paragraph summarizing their overall behavioral skills and potential areas for growth.

Return exactly this JSON structure, with each value a single paragraph:

{{
  "technical_strengths": "<…>",
  "technical_weaknesses": "<…>",
  "behavioral_strengths": "<…>",
  "behavioral_weaknesses": "<…>",
  "technical_ability": "<…>",
  "behavioral_ability": "<…>"
}}

Transcript:
{transcript}
"""

    response = model.generate_content(prompt)
    cleaned = re.sub(r"```json|```", "", response.text).strip()
    return eval(cleaned)  # Ensure response is valid JSON (use json.loads for stricter parsing)


@app.get("/")
def read_root():
    return {"message": "Hello from FastAPI!"}

# API route
@app.post("/analyze-transcript")
async def analyze_transcript(file: UploadFile = File(...)):
    transcript = extract_text(file)
    analysis = get_analysis_from_gemini(transcript)
    return {"analysis": analysis}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
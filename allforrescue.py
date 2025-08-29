#app.py:
import streamlit as st
import os
import json
import pandas as pd
import shutil
import queue
import wave
import av
import whisper
import traceback
from datetime import datetime

from streamlit_webrtc import webrtc_streamer, WebRtcMode
from resume_matcher_app import main

# Load Whisper model once
model = whisper.load_model("base")

# Queue to store audio frames
audio_frames = queue.Queue()

# Transcription function
def transcribe_audio(audio_path):
    result = model.transcribe(audio_path)
    return result["text"]

# Audio frame callback
def audio_frame_callback(frame: av.AudioFrame):
    try:
        # Convert to mono 16-bit PCM and enqueue
        samples = frame.to_ndarray()
        # Optional debug output
        print(f"🔊 Received frame with shape: {samples.shape}, dtype: {samples.dtype}")
        
        # Ensure it's 1D and int16
        if samples.ndim > 1:
            samples = samples.mean(axis=0)  # downmix to mono
        samples = samples.astype("int16")

        audio_frames.put(samples.tobytes())
    except Exception as e:
        print("❌ Error in audio callback:", e)
    return frame

# UI setup
st.set_page_config(layout="wide")
st.title("📄 Resume Matcher")

# ========== Upload Resumes ==========
st.header("1️⃣ Upload Resumes")
resume_files = st.file_uploader("Upload your resumes (PDF, DOCX, TXT)", accept_multiple_files=True)
resume_dir = "resumes"
os.makedirs(resume_dir, exist_ok=True)

if resume_files:
    for file in resume_files:
        with open(os.path.join(resume_dir, file.name), "wb") as f:
            f.write(file.read())
    st.success(f"✅ Uploaded {len(resume_files)} resume(s)")

# ========== Job Description Input ==========
st.header("2️⃣ Enter Job Requirements")

description = st.text_area("🧾 Job Description")
languages = st.text_input("Languages (comma-separated)", "en,fr")
degree = st.selectbox("Required Degree", ['phd', 'master', 'bachelor', 'associate'])
fields = st.text_input("Preferred Fields (comma-separated)", "marketing, business administration, communications")
min_exp = st.slider("Min Experience (years)", 0, 20, 2)
max_exp = st.slider("Max Experience (years)", 0, 30, 5)
skills = st.text_input("Required Skills (comma-separated)", "seo, content creation")

if st.button("✅ Match Resumes"):
    job = {
        "description": description,
        "required_languages": [l.strip().lower() for l in languages.split(',')],
        "required_degree": degree,
        "preferred_fields": [f.strip().lower() for f in fields.split(',')],
        "min_experience_years": min_exp,
        "max_experience_years": max_exp,
        "required_skills": [s.strip().lower() for s in skills.split(',')]
    }

    job_file_path = "job.json"
    with open(job_file_path, "w") as f:
        json.dump(job, f)

    try:
        uploaded_filenames = [file.name for file in resume_files]
        summary_df, detail_df = main(resume_dir, job_file_path, uploaded_filenames)

        
        st.success("✅ Matching complete!")
        
        st.subheader("📊 Summary Scores")
        st.dataframe(summary_df)

    # Save resume match history
        history_entry = {
           "timestamp": datetime.now().isoformat(),
           "job": job,
           "num_resumes": len(resume_files) if resume_files else 0,
           "matched_candidates": summary_df.to_dict(orient="records")
        }

        os.makedirs("history", exist_ok=True)
        history_file = "history/match_history.json"
        if os.path.exists(history_file):
            with open(history_file, "r") as f:
               all_history = json.load(f)
        else:
            all_history = []

        all_history.append(history_entry)
        with open(history_file, "w") as f:
            json.dump(all_history, f, indent=2)

        st.success("🗃️ Match history saved!")

        st.subheader("🔍 Match Details")
        st.dataframe(detail_df)

    except Exception as e:
        st.error(f"🚨 Error during resume matching: {e}")
        st.text(traceback.format_exc())

# ========== Audio File Upload ==========
st.header("📁 Upload an Audio Note")
audio_file = st.file_uploader("Upload an audio note (mp3, wav, m4a)", type=["mp3", "wav", "m4a"])

if audio_file is not None:
    temp_audio_path = f"temp_{audio_file.name}"
    with open(temp_audio_path, "wb") as f:
        f.write(audio_file.read())

    st.info("🔍 Transcribing audio...")
    transcribed_text = transcribe_audio(temp_audio_path)
    st.success("✅ Transcription complete!")
    st.text_area("📝 Transcribed Job Description", transcribed_text, height=150)

    audio_history = {
        "timestamp": datetime.now().isoformat(),
        "filename": audio_file.name,
        "transcription": transcribed_text
    }

    os.makedirs("history", exist_ok=True)
    transcription_file = "history/audio_transcriptions.json"
    if os.path.exists(transcription_file):
        with open(transcription_file, "r") as f:
            transcriptions = json.load(f)
    else:
        transcriptions = []

    transcriptions.append(audio_history)
    with open(transcription_file, "w") as f:
        json.dump(transcriptions, f, indent=2)

    st.success("🗃️ Transcription history saved!")

    # Optionally use this description
    description = transcribed_text
    st.write(f"📂 Saved to: {temp_audio_path}")
    st.write(f"📏 File exists? {os.path.exists(temp_audio_path)}")

# ========== Microphone Audio Recorder ==========
st.header("🎙️ Record audio using your microphone")

st.info("⏳ Please allow microphone access and wait 2–3 seconds after it connects before speaking.")

ctx = webrtc_streamer(
    key="audio-recorder",
    mode=WebRtcMode.SENDRECV,
    audio_frame_callback=audio_frame_callback,
    media_stream_constraints={"audio": True, "video": False},
    async_processing=True,
)

# Add state feedback
st.write(f"🔄 WebRTC state: `{ctx.state}`")
st.write(f"🎛️ Playing: `{ctx.state.playing}`")
st.write(f"🎙️ Frames captured: `{audio_frames.qsize()}`")

if ctx.state.playing:
    st.success("🎤 Recording in progress...")
else:
    st.warning("⚠️ Waiting for microphone access or connection...")

st.write(f"📦 Captured frames: {audio_frames.qsize()}")

if st.button("💾 Save Recorded Audio"):
    if not audio_frames.empty():
        # Save all audio frames to WAV
        all_audio = b"".join(list(audio_frames.queue))
        audio_frames.queue.clear()

        with wave.open("recorded_audio.wav", "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # 16-bit
            wf.setframerate(48000)
            wf.writeframes(all_audio)

        st.success("✅ Audio saved as recorded_audio.wav")

        # Play audio
        with open("recorded_audio.wav", "rb") as f:
            audio_bytes = f.read()
        st.audio(audio_bytes, format="audio/wav")

        # Transcribe audio
        transcription = transcribe_audio("recorded_audio.wav")
        st.text_area("📝 Transcription", transcription, height=150)
    else:
        st.warning("❌ No audio recorded yet. Please try again.")
st.header("📜 View History")

if st.checkbox("🔎 Show Resume Match History"):
    try:
        with open("history/match_history.json") as f:
            match_logs = json.load(f)


        if match_logs:
           for i, entry in enumerate(reversed(match_logs[-5:])):  # Show last 5
              st.markdown(f"### 📌 Match #{len(match_logs)-i} — {entry['timestamp']}")
              with st.expander("📋 Job Description", expanded=False):
                st.json(entry["job"])

              st.write(f"📄 {entry['num_resumes']} resumes matched")
              df = pd.DataFrame(entry["matched_candidates"])
              st.dataframe(df)

              st.download_button(
                label="⬇️ Download Results as CSV",
                data=df.to_csv(index=False).encode("utf-8"),
                file_name="resume_match_results.csv",
                mime="text/csv"
        )
        else:
          st.warning("History is empty.")

    except Exception as e:
        st.error(f"❌ Error reading history: {e}")

if st.checkbox("🔎 Show Audio Transcription History"):
    try:
        with open("history/audio_transcriptions.json") as f:
            trans_logs = json.load(f)
        st.json(trans_logs[-5:])  # last 5 entries
    except:
        st.warning("No transcription history found.")
#resume_matcher_app.py:
import os
import matplotlib.pyplot as plt
import matplotlib
import json

import re
import logging
import math
import argparse
from datetime import datetime
import streamlit as st
import dateparser
import pdfplumber
import docx
import spacy
from langdetect import detect, detect_langs, DetectorFactory
from sentence_transformers import SentenceTransformer, util
from spacy.matcher import PhraseMatcher
import pandas as pd
from transformers import pipeline

import whisper

# Load the model once
whisper_model = whisper.load_model("base")  # tu peux aussi utiliser "small", "medium", "large"

def transcribe_audio(audio_path):
    """
    Convert audio file to text using Whisper
    """
    try:
        result = whisper_model.transcribe(audio_path)
        return result["text"]
    except Exception as e:
        return f"[ERROR] Transcription failed: {e}"


summarizer = pipeline(
    "summarization",
    model="E:/bart_finetuned_resumes/bart_finetuned_resumes",       # path to your folder E:\bart_finetuned_resumes\bart_finetuned_resumes changed here
    tokenizer="E:/bart_finetuned_resumes/bart_finetuned_resumes",   # same path for tokenizer files
    device=-1                                # -1 for CPU, or 0 for GPU
)
# --------------------- CONFIGURATION ---------------------
class Config:
    WEIGHTS = {'semantic': 0.35, 'skills': 0.35, 'experience': 0.1, 'education': 0.15, 'language': 0.5}
    DEGREE_HIERARCHY = {'phd':4,'doctorat':4,'doctorate':4,'master':3,'msc':3,'maîtrise':3,
                        'bachelor':2,'licence':2,'b.s.':2,'m.s.':3,'associate':1,'bts':1,'dut':1}
    LANG_MAP = {'french':'fr','français':'fr','english':'en','arabic':'ar','arabe':'ar','german':'de','allemand':'de',
                'spanish':'es','espagnol':'es','italian':'it','chinese':'zh','mandarin':'zh','japanese':'ja'}
    MODEL_NAME = 'sentence-transformers/distiluse-base-multilingual-cased-v2'
    CHUNK_TOKENS = 512

logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s:%(name)s: %(message)s')
logger = logging.getLogger('ResumeMatcher')

# -------------- UTILITIES --------------
def extract_text(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    if ext == '.txt':
        with open(path,'r',encoding='utf-8',errors='ignore') as f: return f.read()
    if ext == '.pdf':
        try:
            with pdfplumber.open(path) as pdf: return '\n'.join(p.extract_text() or '' for p in pdf.pages)
        except Exception as e:
            logger.warning(f"PDF extraction failed for {path}: {e}"); return ''
    if ext == '.docx':
        try:
            doc = docx.Document(path); return '\n'.join(p.text for p in doc.paragraphs)
        except Exception as e:
            logger.warning(f"DOCX extraction failed for {path}: {e}"); return ''
    return ''

def chunk_text(text: str, max_tokens: int):
    words = text.split()
    for i in range(0,len(words),max_tokens): yield ' '.join(words[i:i+max_tokens])

def parse_date_str(s:str): return dateparser.parse(s) if s else None

# ---------- SCORING COMPONENTS ----------
def score_language(text: str, required: set):
    """
    Language scoring:
    - Only uses declared languages in resume (no detected fallback).
    - Score is proportion of required languages declared.
    - If no required languages specified, returns 1.0.
    """
    declared = set()
    # Look for a 'Languages' or 'Langues' section
    m = re.search(r"(?:langues?|languages?)\s*[:\-–]?\s*(.*?)(?:\n{2}|$)",
                  text, re.IGNORECASE | re.DOTALL)
    if m:
        # Extract tokens and map to codes
        for tok in re.findall(r"\b(\w+)\b", m.group(1)):
            code = Config.LANG_MAP.get(tok.lower())
            if code:
                declared.add(code)

    # Compute score strictly from declared
    if not required:
        score = 1.0
    else:
        score = len(declared & required) / len(required) if declared else 0.0

    return round(score, 4), sorted(declared)


def score_skills(text: str, required_skills: list):
    nlp = spacy.blank('fr')
    matcher = PhraseMatcher(nlp.vocab, attr='LOWER')
    # Create patterns only for required skills
    patterns = [nlp.make_doc(skill) for skill in required_skills]
    matcher.add('SKILLS', patterns)
    doc = nlp(text.lower())
    matches = matcher(doc)
    found_skills = set()
    for _, start, end in matches:
        found_skills.add(doc[start:end].text)
    # Score = number of matched required skills / total required skills
    score = len(found_skills) / len(required_skills) if required_skills else 1.0
    return round(score, 4), list(found_skills)

def extract_education(text:str):
    edus=[]; lines=text.lower().split('\n')
    for ln in lines:
        for deg,_ in Config.DEGREE_HIERARCHY.items():
            if deg in ln:
                m=re.search(fr"{deg}.*?(?:en|of)?\s*([\w\s]+)",ln)
                edus.append((deg,m.group(1).strip() if m else ''))
    return edus

def score_education(edus, req_lvl, req_domains, sbert):
    """
    Strict education scoring:
     - 1.0 only if degree level ≥ required AND at least one domain semantically matches.
     - Otherwise 0.0.
    Returns (score, matched_list).
    """
    lvl_req = Config.DEGREE_HIERARCHY.get(req_lvl, 0)
    matched = []
    for deg, dom in edus:
        lvl = Config.DEGREE_HIERARCHY.get(deg, 0)
        if lvl >= lvl_req and dom and req_domains:
            # check semantic domain match
            dom_emb = sbert.encode(dom, convert_to_tensor=True)
            for rd in req_domains:
                if util.cos_sim(dom_emb, sbert.encode(rd, convert_to_tensor=True)).item() >= 0.7:
                    matched.append((deg, dom))
                    return 1.0, matched
    return 0.0, []
def extract_experience(text:str):
    entries=[]
    pat=re.compile(r"(?P<title>.*?)\s*(?:–|-|to)\s*(?P<start>[A-Za-z]{3,9} \d{4})\s*(?:–|-|to)\s*(?P<end>[A-Za-z]{3,9} \d{4}|present)",re.I)
    for m in pat.finditer(text):
        s=parse_date_str(m.group('start')); e=datetime.now() if 'present' in m.group('end').lower() else parse_date_str(m.group('end'))
        entries.append((m.group('title').strip(),'',s,e))
    return entries

def score_experience(exps, min_y, max_y, req_domains, sbert):
    """
    Improved experience scoring:
    - Only counts roles where domain matches.
    - Applies time decay: more recent experience counts more.
    - Normalizes total weighted years against max_y to a [0,1] score.
    Returns (score, matched_experiences).
    """
    total = 0.0
    matched_exps = []
    now = datetime.now()
    # decay rate: half-life ~ 5 years -> lambda = ln(2)/5 ~0.1386
    lambda_decay = math.log(2) / 5
    for title, desc, start, end in exps:
        if not (start and end):
            continue
        years = (end - start).days / 365.0
        # check domain relevance
        emb = sbert.encode(f"{title} {desc}", convert_to_tensor=True)
        domain_match = False
        for dom in req_domains:
            if util.cos_sim(emb, sbert.encode(dom, convert_to_tensor=True)).item() >= 0.7:
                domain_match = True
                break
        if domain_match and years > 0:
            # time decay based on how long ago the experience ended
            age = (now - end).days / 365.0
            decay = math.exp(-lambda_decay * age)
            weighted = years * decay
            total += weighted
            matched_exps.append((title, round(years, 2)))
    # normalize
    score = round(min(total / max_y, 1.0), 4) if max_y > 0 else 0.0
    return score, matched_exps

def score_semantic(text,job_txt,sbert):
    best=0.0
    for chunk in chunk_text(text,Config.CHUNK_TOKENS):
        sim=util.cos_sim(sbert.encode(chunk,convert_to_tensor=True),
                         sbert.encode(job_txt,convert_to_tensor=True)).item()
        best=max(best,sim)
    return round(best,4)

# --------------------- MAIN ---------------------
"""def main(resume_dir:str,job_input):
    DetectorFactory.seed=0
    sbert=SentenceTransformer(Config.MODEL_NAME)
    nlp=spacy.load('fr_core_news_sm')
    import json
    job=json.load(open(job_input))
    summary_rows=[]; detail_rows=[]
    for fn in os.listdir(resume_dir):
        if not fn.lower().endswith(('.pdf','.docx','.txt')): continue
        txt=extract_text(os.path.join(resume_dir,fn))
        
        if not txt: continue
        lang_sc,langs=score_language(txt,set(job['required_languages']))
        skill_sc, skills = score_skills(txt, job['required_skills'])
        edu_list=extract_education(txt)
        edu_sc,matched_edu=score_education(edu_list,job['required_degree'],job['preferred_fields'],sbert)
        exp_sc,years=score_experience(extract_experience(txt),job['min_experience_years'],job['max_experience_years'],job['preferred_fields'],sbert)
        sem_sc=score_semantic(txt,job['description'],sbert)
        final=round(Config.WEIGHTS['semantic']*sem_sc+Config.WEIGHTS['skills']*skill_sc+
                    Config.WEIGHTS['experience']*exp_sc+Config.WEIGHTS['education']*edu_sc+
                    Config.WEIGHTS['language']*lang_sc,4)
        summary_rows.append({'file':fn,'semantic':sem_sc,'skills':skill_sc,'experience':exp_sc,'education':edu_sc,'language':lang_sc,'final':final})
        detail_rows.append({'file':fn,'languages':','.join(langs),'matched_skills':','.join(skills),'matched_education':','.join(f"{d}:{f}" for d,f in matched_edu)})
    df_sum=pd.DataFrame(summary_rows).sort_values('final',ascending=False)
    df_det=pd.DataFrame(detail_rows).set_index('file')
    return summary_rows, detail_rows"""
def main(resume_dir: str, job_input, file_list=None):
    # seed & models
    DetectorFactory.seed = 0
    sbert = SentenceTransformer(Config.MODEL_NAME)
    nlp   = spacy.load("fr_core_news_sm")

    # load the JSON config
    with open(job_input, "r", encoding="utf-8") as f:
        job = json.load(f)

    summary_rows = []
    detail_rows  = []

    resume_files = [f for f in file_list if f.lower().endswith((".pdf", ".docx", ".txt"))] if file_list else [
    f for f in os.listdir(resume_dir) if f.lower().endswith((".pdf", ".docx", ".txt"))
]


    for fn in resume_files:

        # extract raw text
        txt = extract_text(os.path.join(resume_dir, fn))
        if not txt:
            continue

        # ——— your summarizer ———
        try:
            summary_obj     = summarizer(txt, max_length=150, min_length=30, do_sample=False)
            resume_summary  = summary_obj[0]["summary_text"]
        except Exception as e:
            resume_summary  = f"[SUMMARY ERROR] {e}"

        # ——— scoring ———
        lang_sc, langs     = score_language(txt, set(job["required_languages"]))
        skill_sc, skills   = score_skills(txt, job["required_skills"])
        edu_list           = extract_education(txt)
        edu_sc, matched_ed = score_education(edu_list, job["required_degree"], job["preferred_fields"], sbert)
        exp_sc, years      = score_experience(
                                 extract_experience(txt),
                                 job["min_experience_years"],
                                 job["max_experience_years"],
                                 job["preferred_fields"],
                                 sbert
                             )
        sem_sc             = score_semantic(txt, job["description"], sbert)

        final_score = round(
            Config.WEIGHTS["semantic"]   * sem_sc +
            Config.WEIGHTS["skills"]     * skill_sc +
            Config.WEIGHTS["experience"] * exp_sc +
            Config.WEIGHTS["education"]  * edu_sc +
            Config.WEIGHTS["language"]   * lang_sc,
            4
        )

        # ——— collect rows ———
        summary_rows.append({
            "file":     fn,
            "summary":  resume_summary,
            "semantic": sem_sc,
            "skills":   skill_sc,
            "experience": exp_sc,
            "education":  edu_sc,
            "language":   lang_sc,
            "final":      final_score
        })

        detail_rows.append({
            "file":           fn,
            "languages":      ",".join(langs),
            "matched_skills": ",".join(skills),
            "matched_education": "; ".join(f"{d}:{fld}" for d, fld in matched_ed)
        })

    # turn into DataFrames
    df_sum = pd.DataFrame(summary_rows).sort_values("final", ascending=False)
    df_det = pd.DataFrame(detail_rows).set_index("file")

    return df_sum, df_det

    # Display summaries
    print("=== SUMMARY SCORES ===")
    print(df_sum.to_string(index=False))
    # Display details
    print("\n=== MATCHED DETAILS ===")
    print(df_det.to_string())
    # Bar chart

"""
    # ✅ Fix: force safe font
    # Set font at top (before any plotting)
    matplotlib.rcParams['font.family'] = 'DejaVu Sans'


    fig, ax = plt.subplots()
    df_sum.set_index("file")["final"].plot(kind="bar", ax=ax)
    ax.set_title("Resume Final Scores")
    ax.set_ylabel("Score")
    plt.xticks(rotation=45)
    plt.tight_layout()

    #st.pyplot(fig)
"""
"""if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--resumes',required=True); p.add_argument('--job',required=True)
    a=p.parse_args(); main(a.resumes,a.job)"""
#resume_dir = "/content/OO/OO"
#job_file   = "/content/job.json"

#main(resume_dir, job_file_path)
#mimtkAAAexper(skill/edu works)

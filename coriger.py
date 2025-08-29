import os
import matplotlib.pyplot as plt
import matplotlib
import json
import nltk
from nltk.corpus import stopwords
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
from transformers import BartTokenizer, BartForConditionalGeneration

MODEL_PATH = 'E:/TESTOR/fine_tuned_bart2F'

@st.cache_resource
def load_model():
    tokenizer = BartTokenizer.from_pretrained(MODEL_PATH)
    model = BartForConditionalGeneration.from_pretrained(MODEL_PATH)
    model.eval()
    return model, tokenizer

model, tokenizer = load_model()

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


#------------------- CONFIGURATION ---------------------
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

def chunk_text(text: str, max_tokens: int):
    words = text.split()
    for i in range(0,len(words),max_tokens): yield ' '.join(words[i:i+max_tokens])

def parse_date_str(s:str): return dateparser.parse(s) if s else None


def translate_text(text, source_lang='en', target_lang='fr'):
    """
    Traduit un texte donné de la langue source à la langue cible.
    """
    try:
        translator = GoogleTranslator(source=source_lang, target=target_lang)
        return translator.translate(text)
    except Exception as e:
        logging.error(f"Erreur de traduction : {e}")
        return text  # Retourne le texte original en cas d'erreur


# Fonction pour extraire le texte d'un fichier à l'aide de Docling
def extract_text_with_docling(file_path):
    try:
        converter = DocumentConverter()
        result = converter.convert(file_path)
        return result.document.export_to_markdown().strip()
    except Exception as e:
        logging.error(f"Erreur Docling : {e}")
        return None
    
# Fonction pour convertir une image en PDF
def convert_image_to_pdf(image_path):
    pdf_path = os.path.splitext(image_path)[0] + ".pdf"
    with open(pdf_path, "wb") as pdf_file:
        pdf_file.write(img2pdf.convert(image_path))
    return pdf_path


def detect_language(text):
    try:
        return detect(text)
    except Exception as e:
        logging.error(f"Erreur de détection de langue : {e}")
        return 'fr'  # Par défaut, suppose que le texte est en français

# Assurez-vous que les stopwords sont téléchargés
nltk.download('stopwords')

# Définir les stop words en français et en anglais
stop_words = set(stopwords.words('french') + stopwords.words('english'))

def nettoyer_texte(texte):
    # Suppression des chaînes spécifiques comme <!-- image -->
    texte = re.sub(r'<!-- image -->', ' ', texte)

    # Suppression des caractères non pertinents tout en conservant les retours à la ligne
    texte = re.sub(r'[^\w\s@.#\+\-/%&:\n]', ' ', texte)  # Garde les mots, espaces, retours à la ligne, et les symboles utiles

    # Réduction des espaces multiples tout en conservant les retours à la ligne
    texte = re.sub(r'\s+', ' ', texte)  # Remplace les espaces multiples par un seul espace

    # Suppression des espaces avant et après le texte
    texte = texte.strip()

    # Suppression des stop words
    mots = texte.split()
    texte_sans_stop_words = ' '.join([mot for mot in mots if mot.lower() not in stop_words])

    return texte_sans_stop_words

# Fonction pour traiter les fichiers
def process_file(file_path):
    file_extension = os.path.splitext(file_path)[1].lower()

    # Convertir les images en PDF avant traitement
    if file_extension in ['.jpg', '.jpeg', '.png']:
        file_path = convert_image_to_pdf(file_path)

    
    # Extraire le texte avec Docling
    extracted_text = extract_text_with_docling(file_path)

    # Vérifier si le texte est en anglais et le traduire en français
    if extracted_text:
        if detect_language(extracted_text) == 'en':  # Détection de la langue (ajouter cette fonction)
            extracted_text = translate_text(extracted_text, source_lang='en', target_lang='fr')

        cleaned_text = nettoyer_texte(extracted_text)
    return cleaned_text
# ---------- SCORING COMPONENTS ----------
def score_language(text: str, required: set):
  
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
def generate_summary(resume_text):
    if not resume_text or len(resume_text.strip()) < 100:
        return "[Résumé trop court ou manquant]"

    inputs = tokenizer(resume_text, return_tensors="pt", max_length=1024, truncation=True)

    summary_ids = model.generate(
        **inputs,
        max_length=150,
        min_length=30,
        num_beams=4,
        early_stopping=True
    )

    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

def extract_education(text:str):
    edus=[]; lines=text.lower().split('\n')
    for ln in lines:
        for deg,_ in Config.DEGREE_HIERARCHY.items():
            if deg in ln:
                m=re.search(fr"{deg}.*?(?:en|of)?\s*([\w\s]+)",ln)
                edus.append((deg,m.group(1).strip() if m else ''))
    return edus

def score_education(edus, req_lvl, req_domains, sbert):

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
        txt = extract_text_with_docling(os.path.join(resume_dir, fn))
    # Check if extracted text is too short or an error message
        if not txt or len(txt.strip()) < 50 or txt.startswith("[ERROR]") or txt.startswith("[UNSUPPORTED]"):
          resume_summary = "[Résumé trop court ou manquant]"
        else:
            try:
               resume_summary = generate_summary(txt)
            except Exception as e:
               resume_summary = f"[SUMMARY ERROR] {e}"

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

# -*- coding: utf-8 -*-
# education_user_preferred.py
# Extraction d'éducation + scoring en utilisant preferred_fields fournis par l'utilisateur.
import logging
logging.basicConfig(level=logging.DEBUG)

import os, re, html, unicodedata, logging
from typing import List, Tuple, Dict, Optional
from rapidfuzz import fuzz

# optional SBERT
try:
    from sentence_transformers import SentenceTransformer, util
except Exception:
    SentenceTransformer = None
    util = None

# logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("edu_user_pref")

# ============================
# Vocabulaire utile (DEGREES + institutions)
# ============================
DEGREE_NORMALIZATION = {
    "phd": ["phd", "ph.d", "doctorat", "docteur", "dr", "thèse"],
    "master": [
        "master", "mastère", "mastere", "maîtrise", "maitrise", "msc", "mba", "m2", "m1",
        "master spécialisé", "mastère spécialisé", "master specialise", "mastère specialise",
        "master professionnel", "master pro", "mastère professionnel", "mastère pro"
    ],
    "engineer": [
        "ingénieur", "ingenieur", "ing", "ingénieur d'état", "ingenieur d'etat",
        "diplôme d’ingénieur", "diplome d'ingenieur", "cycle d’ingénieur", "cycle d'ingenieur"
    ],
    "bachelor": ["bachelor", "licence", "licence pro", "licence professionnelle", "b.sc", "b.s"],
    "baccalaureat": ["baccalauréat", "baccalaureat", "bac"],
    "deug": ["deug", "deust", "dut", "bts"]
}

DEGREE_VARIANTS = [v for vs in DEGREE_NORMALIZATION.values() for v in vs]
DEGREE_KEYWORDS_RE = rf"(?P<deg>\b(?:{'|'.join([re.escape(v) for v in DEGREE_VARIANTS])})\b)"
YEAR_RE_SIMPLE = re.compile(r"\b(19|20)\d{2}\b")

MOROCCAN_INSTITUTIONS = [
    "université hassan ii", "universite hassan ii", "université mohammed v", "universite mohammed v",
    "université cadi ayyad", "universite cadi ayyad", "université ibn zohr", "al akhawayn",
    "encg", "ensa", "emi", "ensem", "inpt", "ensam", "ecole nationale", "faculté des sciences"
]

# helper: expand user short aliases (optionnel)
SHORT_FIELD_ALIASES = {
    "data": ["data science", "data engineering", "science des données"],
    "ia": ["intelligence artificielle", "ai", "artificial intelligence"],
    "dev": ["développement logiciel", "génie logiciel"]
}

# ============================
# Text extraction minimale (pdf/docx/txt/img)
# ============================
# NOTE: dans Colab installe pdfplumber & python-docx si nécessaire
# Remplacer ton ancienne extract_text_from_file par ceci
"""
def extract_text_from_pdf(path: str) -> str:
    #Try text extraction with pdfplumber, else fallback to OCR (pdf2image + pytesseract).
    text_parts = []
    # 1) try pdfplumber (fast for born-digital PDFs)
    try:
        import pdfplumber
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(page_text)
        txt = "\n\n".join(text_parts).strip()
        if txt and len(txt) > 20:
            return txt  # success
    except Exception as e:
        logger.info("pdfplumber not usable or failed: %s", e)

    # 2) fallback -> render pages to images and OCR them
    try:
        from pdf2image import convert_from_path
        from PIL import Image
        import pytesseract
        images = convert_from_path(path, dpi=250)  # dpi tradeoff: 200-300
        ocr_texts = []
        for i, img in enumerate(images):
            try:
                page_txt = pytesseract.image_to_string(img, lang='fra+eng')
                if page_txt:
                    ocr_texts.append(page_txt)
            except Exception as e:
                logger.warning("pytesseract failed on page %d: %s", i, e)
        return "\n\n".join(ocr_texts).strip()
    except Exception as e:
        logger.warning("PDF OCR fallback failed: %s", e)
        return ""

def extract_text_from_docx(path: str) -> str:
    try:
        import docx
        doc = docx.Document(path)
        return "\n".join([p.text for p in doc.paragraphs]).strip()
    except Exception as e:
        logger.warning("docx extraction failed: %s", e)
        return ""

def extract_text_from_image(path: str) -> str:
    try:
        from PIL import Image
        import pytesseract
        img = Image.open(path)
        return pytesseract.image_to_string(img, lang='fra+eng').strip()
    except Exception as e:
        logger.warning("Image OCR failed: %s", e)
        return ""

def extract_text_from_file(path: str) -> str:
    #Unified extractor: pdf/docx/txt/image with OCR fallback for scanned PDFs.
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        return extract_text_from_pdf(path)
    elif ext in [".docx", ".doc"]:
        return extract_text_from_docx(path)
    elif ext == ".txt":
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                return f.read()
        except Exception as e:
            logger.warning("txt read failed: %s", e)
            return ""
    elif ext in [".png", ".jpg", ".jpeg", ".tiff", ".bmp"]:
        return extract_text_from_image(path)
    else:
        # try pdf attempt as last resort
        return extract_text_from_pdf(path)
"""
# ============================
# Normalisation & sanitation
# ============================
def normalize_text(s: str) -> str:
    if not s: return ""
    s = html.unescape(s)
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = re.sub(r"\s+", " ", s).strip().lower()
    return s

def sanitize_section(text: str) -> str:
    if not text: return ""
    t = html.unescape(text)
    t = re.sub(r"https?://\S+|www\.\S+|\S+@\S+\.\S+", " ", t)
    t = re.sub(r"\+?\d[\d\s\-\.\(\)]{5,}", " ", t)
    t = re.sub(r"\s{2,}", " ", t).strip()
    return t

# ============================
# Patterns extraction
# ============================
STRICT_PATTERN = re.compile(
    rf"{DEGREE_KEYWORDS_RE}"
    r"(?:\s+(?:en|de|dans|in|of|pour))?"
    r"\s+(?P<field>[A-Za-zÀ-ÖØ-öø-ÿ0-9&\-\s/]{2,120})",
    flags=re.IGNORECASE
)

FALLBACK_PATTERN = re.compile(
    rf"{DEGREE_KEYWORDS_RE}"
    r"\s*(?:[:\-\u2013]|\s+)"
    r"(?P<field2>[A-Za-zÀ-ÖØ-öø-ÿ0-9&\-\s/]{2,120})",
    flags=re.IGNORECASE
)

def extract_degree_field_year(text: str):
    out = []
    if not text: return out
    t = normalize_text(sanitize_section(text))
    for m in STRICT_PATTERN.finditer(t):
        deg = m.group("deg") or ""
        field = (m.group("field") or "").strip()
        y = YEAR_RE_SIMPLE.search(m.group(0))
        year = y.group(0) if y else None
        out.append((deg.strip(), field.strip(), year))
    for m in FALLBACK_PATTERN.finditer(t):
        deg = m.group("deg") or ""
        field = (m.group("field2") or "").strip()
        y = YEAR_RE_SIMPLE.search(m.group(0))
        year = y.group(0) if y else None
        tup = (deg.strip(), field.strip(), year)
        if tup not in out:
            out.append(tup)
    return out

def extract_education_section(text: str) -> str:
    if not text: return ""
    m = re.search(
        r"(?s)(?:^|\n)(?:formation|formations|dipl[oô]mes?|études|parcours)\s*[:\-–]?\s*(.*?)(?=(?:\n{2,}|\bexp[eé]rienc|\bcomp[eé]tences|\blangues|\bprojets|\bcertificat|$))",
        text, flags=re.IGNORECASE)
    return sanitize_section(m.group(1).strip()) if m else sanitize_section(text)

# ============================
# merge candidates
# ============================
def canonical_degree_from_chunk(chunk: str) -> Optional[str]:
    if not chunk: return None
    low = normalize_text(chunk)
    for canon, variants in DEGREE_NORMALIZATION.items():
        for v in variants:
            if v in low:
                return canon
    return None

def merge_edu_candidates(*lists):
    out = []; seen = set()
    for lst in lists:
        if not lst: continue
        for it in lst:
            if isinstance(it, dict):
                deg_raw = it.get("degree_raw",""); major = it.get("major",""); years = it.get("years",[])
                chunk = it.get("chunk","")
            elif isinstance(it, (list, tuple)):
                if len(it) >= 3:
                    deg_raw, major, years = it[0], it[1], it[2]
                elif len(it) == 2:
                    deg_raw, major = it[0], it[1]; years=[]
                else:
                    deg_raw = it[0]; major=""; years=[]
                chunk = f"{deg_raw} | {major}"
            else:
                continue
            deg_canon = canonical_degree_from_chunk(deg_raw) or canonical_degree_from_chunk(major)
            key = (str(deg_canon or "").lower(), str(deg_raw or "").lower(), str(major or "").lower())
            if key in seen: continue
            seen.add(key)
            out.append({"chunk": chunk, "degree_canon": deg_canon, "degree_raw": deg_raw, "major": major, "institution": "", "years": years if isinstance(years,(list,tuple)) else ([years] if years else [])})
    return out

# ============================
# expansion helper for user's preferred_fields
# ============================
def expand_user_fields(user_fields: List[str]) -> List[str]:
    out = []
    for f in user_fields or []:
        nf = normalize_text(f)
        out.append(nf)
        if nf in SHORT_FIELD_ALIASES:
            out.extend(SHORT_FIELD_ALIASES[nf])
    # dedup preserve order
    seen = set(); result = []
    for x in out:
        if x not in seen:
            seen.add(x); result.append(x)
    return result

# ============================
# scoring function (utilise preferred_fields passés par l'utilisateur)
# ============================
def score_education_user(edu_candidates: List[Dict], job_required_level: str, preferred_fields: List[str], sbert_model=None, sim_threshold=0.7):
    """
    edu_candidates: output de merge_edu_candidates
    preferred_fields: liste fournie par l'utilisateur (strings)
    """
    pref_fields_expanded = expand_user_fields(preferred_fields)
    pref_embs = None
    if sbert_model and pref_fields_expanded:
        try:
            pref_embs = sbert_model.encode([normalize_text(f) for f in pref_fields_expanded], convert_to_tensor=True, show_progress_bar=False)
        except Exception as e:
            logger.warning("sbert encode failed: %s", e)
            pref_embs = None

    best = {"score": 0.0, "candidate": None, "method": None, "sim": None, "matched_field": None}
    order = ["baccalaureat","bachelor","deug","licence","engineer","master","phd"]

    for cand in edu_candidates:
        deg_raw = cand.get("degree_raw","") or ""
        major = cand.get("major","") or ""
        deg_canon = (cand.get("degree_canon") or "") or canonical_degree_from_chunk(deg_raw)
        # basic level check
        level_ok = True
        if job_required_level:
            try:
                cand_idx = order.index(deg_canon) if deg_canon in order else -1
                req_idx = order.index(job_required_level) if job_required_level in order else -1
                if cand_idx>=0 and req_idx>=0:
                    level_ok = (cand_idx >= req_idx)
            except Exception:
                level_ok = True

        candidate_field = (major or deg_raw or "").strip()
        candidate_field_norm = normalize_text(candidate_field)

        # exact match on user's fields
        for pf in pref_fields_expanded:
            if pf and pf in candidate_field_norm:
                score = 1.0 if level_ok else 0.8
                if score > best["score"]:
                    best.update({"score": score, "candidate": cand, "method": "keyword", "sim": score, "matched_field": pf})
                break

        # fuzzy match
        if best["score"] < 1.0 and candidate_field:
            for pf in pref_fields_expanded:
                fscore = fuzz.token_set_ratio(candidate_field_norm, pf) / 100.0
                if fscore >= 0.75:
                    score = 0.9 if level_ok else 0.7
                    if score > best["score"]:
                        best.update({"score": score, "candidate": cand, "method": "fuzzy", "sim": fscore, "matched_field": pf})

        # SBERT semantic (optionnel)
        if best["score"] < 1.0 and pref_embs is not None and candidate_field:
            try:
                cand_emb = sbert_model.encode(normalize_text(candidate_field), convert_to_tensor=True)
                sims = util.cos_sim(cand_emb, pref_embs)
                best_sim = float(sims.max()); best_idx = int(sims.argmax())
                if best_sim >= sim_threshold:
                    score = 1.0 if level_ok else 0.85
                    pf = pref_fields_expanded[best_idx]
                    if score > best["score"]:
                        best.update({"score": score, "candidate": cand, "method": "sbert", "sim": best_sim, "matched_field": pf})
            except Exception as e:
                logger.debug("sbert fail: %s", e)

    return (best["score"], best) if best["candidate"] else (0.0, None)

# ============================
# Process wrapper: un test simple avec preferred_fields en paramètre
# ============================
def process_resume_for_education(path: str, job_required_level: str, preferred_fields: List[str], use_sbert: bool = False):
    text = extract_text_from_file(path)
    if not text:
        logger.warning("Aucun texte extrait pour %s", path)
        return None
    section = extract_education_section(text)
    regex_candidates = extract_degree_field_year(text)
    section_candidates = extract_degree_field_year(section)
    merged = merge_edu_candidates(regex_candidates, section_candidates)

    # optionally load SBERT model once if requested
    sbert_model = None
    if use_sbert and SentenceTransformer is not None:
        try:
            sbert_model = SentenceTransformer("all-MiniLM-L6-v2")
        except Exception as e:
            logger.warning("SBERT load failed: %s", e); sbert_model = None

    score, best = score_education_user(merged, job_required_level, preferred_fields, sbert_model=sbert_model)
    return {"path": path, "merged": merged, "score": score, "best": best, "section_preview": section[:800]}
# new helper: process from raw text instead of file path
def process_resume_for_education_from_text(text: str, job_required_level: str, preferred_fields: List[str], sbert_model=None):
    """
    Analyse l'éducation à partir d'un texte déjà extrait (évite ré-extraction).
    Retourne dict similaire à process_resume_for_education.
    """
    if not text:
        logger.warning("process_resume_for_education_from_text: text empty")
        return None
    section = extract_education_section(text)
    regex_candidates = extract_degree_field_year(text)
    section_candidates = extract_degree_field_year(section)
    merged = merge_edu_candidates(regex_candidates, section_candidates)

    # no SBERT reload here — on reçoit sbert_model en paramètre (peut être None)
    score, best = score_education_user(merged, job_required_level, preferred_fields, sbert_model=sbert_model)
    return {"path": None, "merged": merged, "score": score, "best": best, "section_preview": section[:800]}

# ============================
# Exemple d'utilisation:
# ============================
"""if __name__ == "__main__":
    # remplacer par le chemin de ton CV uploadé dans Colab
    path = "/content/cv_intermediaire.pdf"
    job_required_level = "licence"   # ex: 'master', 'licence', 'phd' ou '' si non précisé
    # preferred_fields saisis par l'utilisateur (exemple)
    preferred_fields = ["Commerce", "marketing", "Statistique"]
    res = process_resume_for_education(path, job_required_level, preferred_fields, use_sbert=False)
    if res is None:
          print("❗ Aucun texte n'a été extrait du fichier. Vérifie que le PDF n'est pas protégé/endommagé ou que l'upload a réussi.")
    else:
         print("Score:", res["score"])
         print("Best match:", res["best"])
         print("Merged candidates:", res["merged"][:8])
         print("Section preview:", res["section_preview"][:400])
#education
"""
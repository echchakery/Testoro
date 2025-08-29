# pages/2_🧑_Espace_Candidat.py
from resume_matcher_app import main  # Ton module de matching
import streamlit as st
import os
import json
import uuid
import tempfile
import re
import zipfile
import pandas as pd
from typing import Any
from datetime import datetime

st.set_page_config(layout="wide")
st.title("📑 Analyse Automatique des CVs:")
# -----------------------
# Helpers for safe IO
# -----------------------

import os
import re
import tempfile
import zipfile
from datetime import datetime

# -----------------------
# Helpers for safe IO
# -----------------------
def _safe_filename(name: str) -> str:
    """Sanitize filename but keep extension."""
    name = os.path.basename(name or "uploaded")
    parts = name.split(".")
    if len(parts) > 1:
        ext = parts[-1]
        base = ".".join(parts[:-1])
        base = re.sub(r"[^A-Za-z0-9\-_]", "_", base)
        ext = re.sub(r"[^A-Za-z0-9]", "", ext)
        safe = f"{base[:150]}.{ext}"
    else:
        safe = re.sub(r"[^A-Za-z0-9\-_\.]", "_", name)[:200]
    return safe

def safe_save_uploaded_file(uploaded_file, out_dir="candidates") -> str:
    os.makedirs(out_dir, exist_ok=True)
    safe_name = _safe_filename(uploaded_file.name)
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    dest = os.path.join(out_dir, f"{timestamp}_{safe_name}")

    tmp_path = None
    try:
        fd, tmp_path = tempfile.mkstemp(prefix="upload_", dir=out_dir)
        os.close(fd)
        with open(tmp_path, "wb") as tmpf:
            try:
                tmpf.write(uploaded_file.getbuffer())
            except AttributeError:
                tmpf.write(uploaded_file.read())
        os.replace(tmp_path, dest)
        return dest
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except:
                pass
"""
import os
import uuid
import tempfile

def safe_save_uploaded_file(uploaded_file, out_dir="candidates") -> str:
    os.makedirs(out_dir, exist_ok=True)

    original_name = uploaded_file.name
    dest = os.path.join(out_dir, original_name)

    # If a file with the same name already exists, add a unique suffix
    if os.path.exists(dest):
        base, ext = os.path.splitext(original_name)
        unique_suffix = uuid.uuid4().hex[:8]  # short unique id
        dest = os.path.join(out_dir, f"{base}_{unique_suffix}{ext}")

    tmp_path = None
    try:
        fd, tmp_path = tempfile.mkstemp(prefix="upload_", dir=out_dir)
        os.close(fd)
        with open(tmp_path, "wb") as tmpf:
            try:
                tmpf.write(uploaded_file.getbuffer())
            except AttributeError:
                tmpf.write(uploaded_file.read())
        os.replace(tmp_path, dest)
        return dest
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except:
                pass
"""
# -----------------------
# Handle single upload or ZIP
# -----------------------
def handle_uploaded_file(uploaded_file, out_dir="candidates"):
    """Return list of saved filenames for single file or ZIP."""
    saved_files = []

    if uploaded_file.name.lower().endswith(".zip"):
        # Handle ZIP upload
        with tempfile.TemporaryDirectory() as tmpdir:
            zip_path = os.path.join(tmpdir, uploaded_file.name)
            with open(zip_path, "wb") as f:
                f.write(uploaded_file.read())

            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                for member in zip_ref.namelist():
                    if member.lower().endswith((".pdf", ".docx", ".txt", ".jpg", ".jpeg", ".png")):
                        extracted = zip_ref.extract(member, tmpdir)
                        with open(extracted, "rb") as f_in:
                            class FakeUpload:
                                def __init__(self, name, data):
                                    self.name = name
                                    self._data = data
                                def read(self): return self._data
                            fake_file = FakeUpload(os.path.basename(member), f_in.read())
                            dest = safe_save_uploaded_file(fake_file, out_dir)
                            saved_files.append(os.path.basename(dest))
    else:
        # Handle single file
        dest = safe_save_uploaded_file(uploaded_file, out_dir)
        saved_files.append(os.path.basename(dest))

    return saved_files

def atomic_write_json(obj, path, encoding="utf-8", ensure_ascii=False, indent=2):
    """Write JSON atomically."""
    d = os.path.dirname(path) or "."
    os.makedirs(d, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix="tmp_json_", dir=d)
    os.close(fd)
    try:
        with open(tmp_path, "w", encoding=encoding) as f:
            json.dump(obj, f, ensure_ascii=ensure_ascii, indent=indent)
        os.replace(tmp_path, path)
    finally:
        if os.path.exists(tmp_path):
            try: os.remove(tmp_path)
            except: pass

def _to_jsonable(x: Any):
    """Make Python/numpy/pandas types JSON safe."""
    import numpy as _np
    if x is None:
        return None
    if isinstance(x, datetime):
        return x.isoformat()
    if _np is not None and isinstance(x, (_np.integer,)):
        return int(x)
    if _np is not None and isinstance(x, (_np.floating,)):
        return float(x)
    if isinstance(x, (str, int, float, bool)):
        return x
    if isinstance(x, (list, tuple, set)):
        return [_to_jsonable(i) for i in x]
    if isinstance(x, dict):
        return {str(k): _to_jsonable(v) for k, v in x.items()}
    try:
        return x.tolist()
    except Exception:
        return str(x)

# -----------------------
# Load job offers
# -----------------------
JOB_OFFERS_DIR = "job_offers"
job_files = sorted(os.listdir(JOB_OFFERS_DIR)) if os.path.exists(JOB_OFFERS_DIR) else []

if not job_files:
    st.info("Aucune offre disponible pour le moment.")
else:
    for file in job_files:
        try:
            with open(os.path.join(JOB_OFFERS_DIR, file), encoding="utf-8") as f:
                job = json.load(f)
        except Exception as e:
            st.error(f"Impossible de lire l'offre {file}: {e}")
            continue

        with st.expander(f"📄 Offre : {job.get('job_title', 'Titre non spécifié')}"):
            st.write((job.get("description") or "")[:300] + "...")
            st.markdown(f"""
            - 🎓 Diplôme requis : **{job.get('required_degree','non spécifié')}**
            - 📚 Domaines préférés : {', '.join(job.get('preferred_fields', []))}
            - 📚 Domaines d'éxperience : {', '.join(job.get('EXP job', []))}
            - 🌐 Langues : {", ".join(job.get('required_languages', []))}
            - 🛠️ Compétences : {", ".join(job.get('required_skills', []))}
            - 🔢 Expérience : {job.get('min_experience_years', 0)} à {job.get('max_experience_years', 0)} ans
            """)

            # -----------------------
            # Upload CV or ZIP
            # -----------------------
            with st.form(f"apply_{job.get('job_id', file)}"):
                uploaded_cv = st.file_uploader(
                    "📎 Déposez votre CV (PDF, DOCX, TXT, JPG, PNG) ou un ZIP de plusieurs CVs",
                    type=["pdf", "docx", "txt", "jpg", "png", "zip"],
                    key=job.get('job_id', file)
                )

                submit = st.form_submit_button("📩 Postuler")

                if submit and uploaded_cv:
                    saved_files = []  # all saved resumes for this job

                    try:
                        if uploaded_cv.name.lower().endswith(".zip"):
                            # --- Handle ZIP upload ---
                            with tempfile.TemporaryDirectory() as tmpdir:
                                zip_path = os.path.join(tmpdir, uploaded_cv.name)
                                with open(zip_path, "wb") as f:
                                    f.write(uploaded_cv.read())

                                with zipfile.ZipFile(zip_path, "r") as zip_ref:
                                    for member in zip_ref.namelist():
                                        if member.lower().endswith((".pdf", ".docx", ".txt", ".jpg", ".jpeg", ".png")):
                                            extracted = zip_ref.extract(member, tmpdir)
                                            with open(extracted, "rb") as f_in:
                                                class FakeUpload:
                                                    def __init__(self, name, data):
                                                        self.name = name
                                                        self._data = data
                                                    def read(self): return self._data
                                                fake_file = FakeUpload(os.path.basename(member), f_in.read())
                                                dest = safe_save_uploaded_file(fake_file, out_dir="candidates")
                                                saved_files.append(os.path.basename(dest))
                        else:
                            # --- Handle single file upload ---
                            dest = safe_save_uploaded_file(uploaded_cv, out_dir="candidates")
                            saved_files.append(os.path.basename(dest))

                        st.success(f"CV(s) sauvegardé(s): {', '.join(saved_files)}")

                    except Exception as e:
                        st.error(f"Erreur lors de la sauvegarde des CVs: {e}")
                        continue

                    # --- Run matching on all saved resumes ---
                    try:
                        summary_df, detail_df = main("candidates", os.path.join(JOB_OFFERS_DIR, file), saved_files)
                    except Exception as e:
                        st.error(f"Erreur lors du matching: {e}")
                        import traceback; st.text(traceback.format_exc())
                        continue

                    # --- Process each resume separately ---
                    for saved_basename in saved_files:
                        try:
                            summary = summary_df.loc[summary_df["file"] == saved_basename].to_dict(orient="records")[0]
                        except Exception:
                            summary = {"final": None}

                        summary["classification"] = summary.get("predicted_class", "Non classé")
                        summary["original_filename"] = uploaded_cv.name if len(saved_files) == 1 else saved_basename

                        matched_domains, matched_skills, matched_languages = [], [], []
                        try:
                            if detail_df is not None and not detail_df.empty:
                                if saved_basename in detail_df.index:
                                    row = detail_df.loc[saved_basename]
                                else:
                                    row = detail_df[detail_df["file"] == saved_basename].iloc[0] if "file" in detail_df.columns else {}
                                if not row.empty:
                                    raw = row.get("matched_experience_domains", None)
                                    if raw:
                                        if isinstance(raw, str):
                                            try: matched_domains = json.loads(raw)
                                            except: matched_domains = [raw]
                                        elif isinstance(raw, (list, dict)): matched_domains = raw

                                    raw_sk = row.get("matched_skills_list", None)
                                    if raw_sk:
                                        if isinstance(raw_sk, str):
                                            try: matched_skills = json.loads(raw_sk)
                                            except: matched_skills = [s.strip() for s in raw_sk.split(",") if s.strip()]
                                        elif isinstance(raw_sk, (list, tuple)): matched_skills = list(raw_sk)

                                    raw_langs = row.get("languages", None)
                                    if raw_langs:
                                        if isinstance(raw_langs, str):
                                            try: matched_languages = json.loads(raw_langs)
                                            except: matched_languages = [raw_langs]
                                        elif isinstance(raw_langs, (list, tuple)): matched_languages = list(raw_langs)
                        except Exception:
                            pass

                        summary["matched_experience_domains"] = matched_domains
                        summary["matched_skills"] = matched_skills
                        summary["matched_languages"] = matched_languages

                        det_row = {}
                        try:
                            if detail_df is not None and not detail_df.empty:
                                if saved_basename in detail_df.index:
                                    det_row = detail_df.loc[saved_basename].to_dict()
                                elif "file" in detail_df.columns:
                                    row = detail_df[detail_df["file"] == saved_basename]
                                    if not row.empty:
                                        det_row = row.iloc[0].to_dict()
                        except Exception:
                            det_row = {}

                        det_row_jsonable = {k: _to_jsonable(v) for k, v in (det_row or {}).items()}
                        det_row_jsonable.setdefault("matched_experience_domains", _to_jsonable(matched_domains))
                        det_row_jsonable.setdefault("matched_skills_list", _to_jsonable(matched_skills))
                        det_row_jsonable.setdefault("languages", _to_jsonable(matched_languages))
                        det_row_jsonable.setdefault(
                            "matched_education_list",
                            _to_jsonable(det_row_jsonable.get("matched_education_list", det_row_jsonable.get("matched_education", [])))
                        )

                        match_data = {
                            "timestamp": datetime.now().isoformat(),
                            "job_id": job.get("job_id"),
                            "resume_file": saved_basename,
                            "original_filename": summary["original_filename"],
                            "match_score": summary,
                            "match_details": det_row_jsonable
                        }

                        try:
                            os.makedirs("applications", exist_ok=True)
                            app_file = os.path.join("applications", f"{saved_basename}_{job.get('job_id', file)}.json")
                            atomic_write_json(_to_jsonable(match_data), app_file, ensure_ascii=False, indent=2)
                            st.success(f"✅ Candidature envoyée avec succès pour {saved_basename}")
                        except Exception as e:
                            st.error(f"Erreur lors de l'enregistrement de la candidature: {e}")
                            import traceback; st.text(traceback.format_exc())


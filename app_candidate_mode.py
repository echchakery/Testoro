import streamlit as st
import os
import json
import uuid
import pandas as pd
from datetime import datetime
from resume_matcher_app import main  # <-- ton module de matching existant

# === CONFIG ===
st.set_page_config(layout="wide")
st.title("📄 Resume Matcher — Mode RH & Candidat")

os.makedirs("job_offers", exist_ok=True)
os.makedirs("candidates", exist_ok=True)
os.makedirs("applications", exist_ok=True)

# ==========================
# 1️⃣ SECTION RH : Créer une offre
# ==========================
st.header("👔 RH : Créer une Offre d'Emploi")

with st.form("job_offer_form"):
    description = st.text_area("📝 Description du poste", height=150)
    languages = st.text_input("🌐 Langues requises (ex: fr,en)", "fr,en")
    degree = st.selectbox("🎓 Niveau requis", ['phd', 'master', 'bachelor', 'associate'])
    fields = st.text_input("📚 Domaines préférés", "finance, sales, marketing")
    min_exp = st.slider("🔢 Expérience minimum (années)", 0, 20, 2)
    max_exp = st.slider("🔢 Expérience maximum (années)", 0, 30, 8)
    skills = st.text_input("🛠️ Compétences clés", "excel, crm, financial analysis")

    submit = st.form_submit_button("✅ Créer l'offre")

    if submit and description:
        job_id = str(uuid.uuid4())
        job_data = {
            "job_id": job_id,
            "description": description,
            "required_languages": [l.strip().lower() for l in languages.split(',')],
            "required_degree": degree,
            "preferred_fields": [f.strip().lower() for f in fields.split(',')],
            "min_experience_years": min_exp,
            "max_experience_years": max_exp,
            "required_skills": [s.strip().lower() for s in skills.split(',')]
        }

        with open(f"job_offers/{job_id}.json", "w") as f:
            json.dump(job_data, f, indent=2)

        st.success(f"📌 Offre créée avec succès ! ID de l'offre : `{job_id}`")
        st.info(f"Partagez cet ID aux candidats pour postuler.")

# ==========================
# 2️⃣ SECTION Candidat : Postuler à une offre
# ==========================
st.header("🧑‍💼 Candidat : Postuler à une Offre")

job_id_input = st.text_input("📥 Entrez l'ID de l'offre reçue", "")
uploaded_cv = st.file_uploader("📎 Déposez votre CV (PDF, DOCX, TXT,jpg)", type=["pdf", "docx", "txt","jpg","png"])

if st.button("🚀 Postuler"):
    if not job_id_input:
        st.warning("❗ Veuillez entrer un ID d'offre.")
    elif not uploaded_cv:
        st.warning("❗ Veuillez téléverser un CV.")
    else:
        job_path = f"job_offers/{job_id_input}.json"
        if not os.path.exists(job_path):
            st.error("❌ L'offre spécifiée n'existe pas.")
        else:
            try:
                # Charger l'offre
                with open(job_path, "r") as f:
                    job = json.load(f)

                # Sauvegarder le CV
                cv_filename = uploaded_cv.name
                cv_path = os.path.join("candidates", cv_filename)
                with open(cv_path, "wb") as f:
                    f.write(uploaded_cv.read())

                # Appeler le modèle de matching
                summary_df, detail_df = main("candidates", job_path, [cv_filename])
                st.success("✅ Candidature analysée avec succès !")
                st.subheader("📊 Résultat du Matching")
                st.dataframe(summary_df)

                # Historique par candidature
                history_entry = {
                    "timestamp": datetime.now().isoformat(),
                    "job_id": job_id_input,
                    "resume_file": cv_filename,
                    "match_score": summary_df.to_dict(orient="records")[0]
                }

                hist_file = f"applications/{cv_filename}_{job_id_input}.json"
                with open(hist_file, "w") as f:
                    json.dump(history_entry, f, indent=2)

                st.success("🗃️ Résultat sauvegardé pour cette candidature.")

            except Exception as e:
                st.error(f"💥 Erreur pendant le traitement : {e}")

# ==========================
# 3️⃣ Visualiser les Offres / Candidatures
# ==========================
st.header("📜 Historique")

if st.checkbox("📂 Voir les offres d'emploi créées"):
    job_files = os.listdir("job_offers")
    if not job_files:
        st.info("Aucune offre enregistrée.")
    else:
        for job_file in job_files[-5:]:
            with open(f"job_offers/{job_file}", "r") as f:
                job = json.load(f)
            st.markdown(f"### 🧾 Offre : `{job['job_id']}`")
            st.json(job)

if st.checkbox("📁 Voir les dernières candidatures traitées"):
    app_files = os.listdir("applications")
    if not app_files:
        st.info("Aucune candidature enregistrée.")
    else:
        for app_file in app_files[-5:]:
            with open(f"applications/{app_file}", "r") as f:
                app = json.load(f)
            st.markdown(f"### 📝 Candidature pour offre `{app['job_id']}`")
            st.json(app)

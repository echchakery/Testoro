#espace_hr:

import streamlit as st
import os
import json
import uuid
from datetime import datetime
import pandas as pd
import streamlit as st
# espace_hr.py

import streamlit as st
import os
import json
import uuid
from datetime import datetime
import pandas as pd

def render():
    st.title("📄 Accueil - Espace RH")

    st.title("👔 Espace RH : Gérer les Offres d’Emploi")
    os.makedirs("job_offers", exist_ok=True)
    os.makedirs("applications", exist_ok=True)

    # === Voir toutes les offres RH ===
    st.subheader("📋 Offres existantes")

    job_files = sorted(os.listdir("job_offers"))

    if not job_files:
        st.info("Aucune offre n’a encore été créée.")
    else:
        for file in job_files:
            with open(f"job_offers/{file}","r", encoding="utf-8") as f:
                job = json.load(f)

            st.markdown(f"### 🧾 Offre : {job['job_title']}")

            # === Voir les candidatures reçues ===
            st.markdown("**📂 Candidatures reçues :**")
            matched = []

            for app_file in os.listdir("applications"):
                if app_file.endswith(".json"):
                    with open(f"applications/{app_file}", "r", encoding="utf-8") as af:
                        app = json.load(af)
                        if app.get("job_id") == job["job_id"]:
                            matched.append(app)

            if matched:
                st.success(f"{len(matched)} candidature(s) trouvée(s)")

                data_for_df = []
                for m in matched:
                    score = m.get("match_score", {})
                    details = m.get("match_details", score)

                    # Handle skills list
                    matched_skills_list = details.get("matched_skills_list") or details.get("matched_skills") or []
                    if isinstance(matched_skills_list, str):
                        try:
                            matched_skills_list = json.loads(matched_skills_list)
                        except Exception:
                            matched_skills_list = [s.strip() for s in matched_skills_list.split(",") if s.strip()]

                    # Handle education list
                    matched_education_list = details.get("matched_education_list") or details.get("matched_education") or []
                    if isinstance(matched_education_list, str):
                        matched_education_list = [matched_education_list]

                    # Handle languages list
                    languages_list = details.get("matched_languages") or details.get("languages") or []
                    if isinstance(languages_list, str):
                        languages_list = [l.strip() for l in languages_list.split(",") if l.strip()]

                    data_for_df.append({
                        "CV": m.get("resume_file", "N/A"),
                        "Résumé": score.get("summary", "Résumé indisponible"),
                        "Score global": score.get("final", "N/A"),
                    })

                # ✅ Create DataFrame
                df = pd.DataFrame(data_for_df)

                # 1) Convertir la colonne "Score global" en numérique
                df["Score global"] = pd.to_numeric(df["Score global"], errors="coerce")

                # 2) Trier par Score global décroissant
                df = df.sort_values(by="Score global", ascending=False, ignore_index=True)

                # 3) Renommer les colonnes
                df = df.rename(columns={
                    "CV": "📄 CV",
                    "Résumé": "📝 Résumé",
                    "Score global": "🔢 Score Total",
                })

                # 4) Afficher
                st.dataframe(df, use_container_width=True)
            else:
                st.info("Aucune candidature reçue pour cette offre.")

    # === Créer une nouvelle offre ===
    st.subheader("➕ Créer une nouvelle offre")

    with st.form("create_job_offer"):
        job_title = st.text_input("📌 Titre de l'offre", "Exemple : Analyste Finance & Vente")
        description = st.text_area("📝 Description du poste", height=150)
        languages = st.text_input("🌐 Langues requises", "fr,en")
        degree = st.selectbox("🎓 Diplôme requis", ['phd', 'master', 'bachelor','licence','deug','deust','Doctorat','Ingénieur'])
        fields = st.text_input("📚 Domaines", "finance, sales")
        expfields = st.text_input("EXP job", "data scientist, manager")
        min_exp = st.slider("🔢 Expérience minimum", 0, 20, 2)
        max_exp = st.slider("🔢 Expérience maximum", 0, 30, 5)
        skills = st.text_input("🛠️ Compétences", "excel, crm, negotiation")

        submit = st.form_submit_button("✅ Enregistrer l'offre")

        if submit and description and job_title:
            job_id = str(uuid.uuid4())
            job_data = {
                "job_id": job_id,
                "job_title": job_title,
                "description": description,
                "required_languages": [x.strip().lower() for x in languages.split(',')],
                "required_degree": degree,
                "preferred_fields": [x.strip().lower() for x in fields.split(',')],
                "experience_fields": [x.strip().lower() for x in expfields.split(',')],
                "min_experience_years": min_exp,
                "max_experience_years": max_exp,
                "required_skills": [x.strip().lower() for x in skills.split(',')],
                "created_at": datetime.now().isoformat()
            }

            with open(f"job_offers/{job_id}.json", "w", encoding="utf-8") as f:
                json.dump(job_data, f, indent=2, ensure_ascii=False)

            st.success(f"✅ Offre '{job_title}' créée avec succès")

"""
def render():
    st.title("📄 Accueil - Espace RH")


st.title("👔 Espace RH : Gérer les Offres d’Emploi")
os.makedirs("job_offers", exist_ok=True)
os.makedirs("applications", exist_ok=True)


# === Voir toutes les offres RH ===
st.subheader("📋 Offres existantes")

job_files = sorted(os.listdir("job_offers"))

if not job_files:
    st.info("Aucune offre n’a encore été créée.")
else:
    for file in job_files:
        with open(f"job_offers/{file}","r", encoding="utf-8") as f:
            job = json.load(f)

        st.markdown(f"### 🧾 Offre : {job['job_title']}")
        #st.text(f"🆔 ID: {job['job_id']}")
        #st.text(f"{job['description'][:]}...")
        #st.markdown(f
        #- 🎓 Diplôme requis : {job.get('required_degree', 'N/A')}
        #- 🌐 Langues : {', '.join(job.get('required_languages', []))}
        #- 📚 Domaines : {', '.join(job.get('preferred_fields', []))}
        #- 🛠️ Compétences : {', '.join(job.get('required_skills', []))}
        #- 🔢 Expérience : {job.get('min_experience_years', 'N/A')} à {job.get('max_experience_years', 'N/A')} ans
        #- 📚 EXP jobs : {', '.join(job.get('experience_fields', []))}
        #)

        # === Voir les candidatures reçues ===
        st.markdown("**📂 Candidatures reçues :**")
        matched = []

        for app_file in os.listdir("applications"):
            if app_file.endswith(".json"):
                with open(f"applications/{app_file}", "r", encoding="utf-8") as af:
                    app = json.load(af)
                    if app.get("job_id") == job["job_id"]:
                        matched.append(app)

        if matched:
           st.success(f"{len(matched)} candidature(s) trouvée(s)")

           data_for_df = []
           for m in matched:
               score = m.get("match_score", {})
               details = m.get("match_details", score)  # fallback to match_score if no match_details

    # Handle skills list
               matched_skills_list = details.get("matched_skills_list") or details.get("matched_skills") or []
               if isinstance(matched_skills_list, str):
                  try:
                      matched_skills_list = json.loads(matched_skills_list)
                  except Exception:
                      matched_skills_list = [s.strip() for s in matched_skills_list.split(",") if s.strip()]

      # Handle education list
               matched_education_list = details.get("matched_education_list") or details.get("matched_education") or []
               if isinstance(matched_education_list, str):
                     matched_education_list = [matched_education_list]

    # Handle languages list
               languages_list = details.get("matched_languages") or details.get("languages") or []
               if isinstance(languages_list, str):
                    languages_list = [l.strip() for l in languages_list.split(",") if l.strip()]

               data_for_df.append({
                    "CV": m.get("resume_file", "N/A"),
                    "Résumé": score.get("summary", "Résumé indisponible"),
                    #"Classification": score.get("classification", "Non classé"),
                    "Score global": score.get("final", "N/A"),
                    #"Score sémantique": score.get("semantic", "N/A"),
                    #"Compétences (score)": score.get("skills", "N/A"),
                    #"Compétences (list)": matched_skills_list,
                    #"Expérience (score)": score.get("experience", "N/A"),
                    #"Éducation (score)": score.get("education", "N/A"),
                    #"Éducation (list)": matched_education_list,
                    #"Langues (score)": score.get("language", "N/A"),
                    #"Langues (list)": languages_list,
                    
    
                    # ✅ New: add LLaMA score & explanation
                    #"LLaMA (score)": score.get("llama", "N/A"),
                    #"LLaMA (explication)": score.get("llama_expl", "N/A"),
    
                    #"Date de candidature": m.get("timestamp", "N/A"),
                    #"Date de candidature": m.get("timestamp", "N/A")
                })


    # ✅ Create DataFrame
           df = pd.DataFrame(data_for_df)

    # 1) Convertir la colonne "Score global" en numérique
           df["Score global"] = pd.to_numeric(df["Score global"], errors="coerce")

    # 2) Trier par Score global décroissant
           df = df.sort_values(by="Score global", ascending=False, ignore_index=True)

    # 3) Renommer les colonnes
           df = df.rename(columns={
              "CV": "📄 CV",
              "Résumé": "📝 Résumé",
              #"Classification": "🧠 Classification",
              "Score global": "🔢 Score Total",
              #"Score sémantique": "🔍 Score Sémantique",
              #"Compétences (score)": "✅ Compétences (score)",
              #"Compétences (list)": "✅ Compétences (list)",
              #"Expérience (score)": "💼 Expérience (score)",
              #"Éducation (score)": "🎓 Éducation (score)",
              #"Éducation (list)": "🎓 Éducation (list)",
              #"Langues (score)": "🌐 Langues (score)",
              #"Langues (list)": "🌐 Langues (list)",
              #"LLaMA (score)": "🤖 LLaMA (score)",
              #"LLaMA (explication)": "📝 LLaMA (explication)",
              #"Date de candidature": "📅 Date de candidature"
           })

    # 4) Afficher
           st.dataframe(df, use_container_width=True)
        else:
         st.info("Aucune candidature reçue pour cette offre.")
# === Créer une nouvelle offre ===
st.subheader("➕ Créer une nouvelle offre")

with st.form("create_job_offer"):
    job_title = st.text_input("📌 Titre de l'offre", "Exemple : Analyste Finance & Vente")
    description = st.text_area("📝 Description du poste", height=150)
    languages = st.text_input("🌐 Langues requises", "fr,en")
    degree = st.selectbox("🎓 Diplôme requis", ['phd', 'master', 'bachelor','licence','deug','deust','Doctorat','Ingénieur'])
    fields = st.text_input("📚 Domaines", "finance, sales")
    expfields = st.text_input("EXP job", "data scientist, manager")
    min_exp = st.slider("🔢 Expérience minimum", 0, 20, 2)
    max_exp = st.slider("🔢 Expérience maximum", 0, 30, 5)
    skills = st.text_input("🛠️ Compétences", "excel, crm, negotiation")

    submit = st.form_submit_button("✅ Enregistrer l'offre")

    if submit and description and job_title:
        job_id = str(uuid.uuid4())
        job_data = {
            "job_id": job_id,
            "job_title": job_title,
            "description": description,
            "required_languages": [x.strip().lower() for x in languages.split(',')],
            "required_degree": degree,
            "preferred_fields": [x.strip().lower() for x in fields.split(',')],
            "experience_fields": [x.strip().lower() for x in expfields.split(',')],
            "min_experience_years": min_exp,
            "max_experience_years": max_exp,
            "required_skills": [x.strip().lower() for x in skills.split(',')],
            "created_at": datetime.now().isoformat()
        }

        with open(f"job_offers/{job_id}.json", "w", encoding="utf-8") as f:
           json.dump(job_data, f, indent=2, ensure_ascii=False)


        st.success(f"✅ Offre '{job_title}' créée avec succès")
        #st.success(f"✅ Offre '{job_title}' créée avec succès — ID : {job_id}")
print("the prob solving")



"""

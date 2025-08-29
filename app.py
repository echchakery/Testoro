# app.py
"""
import streamlit as st

st.set_page_config(page_title="📄 Resume Matcher", layout="wide")
st.title("📄 Resume Matcher App")
st.markdown("Bienvenue dans l'application RH et Recrutement par Matching de CV.")
st.markdown("➡️ Utilisez la barre latérale pour naviguer entre les pages.")
"""
from importlib import import_module
import streamlit as st

st.set_page_config(page_title="📄 Resume Matcher", layout="wide")

# ---------- session state init ----------
if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False
if "username" not in st.session_state:
    st.session_state["username"] = ""

USERNAME = "monuser"
PASSWORD = "monpassword"

def safe_rerun():
    rerun = getattr(st, "rerun", None) or getattr(st, "experimental_rerun", None)
    if callable(rerun):
        try:
            rerun()
        except Exception:
            pass

# ---------- login ----------
if not st.session_state["logged_in"]:
    st.markdown("## 🔐 Connexion")
    with st.form("login_form", clear_on_submit=False):
        user = st.text_input("Nom d'utilisateur")
        pwd = st.text_input("Mot de passe", type="password")
        submitted = st.form_submit_button("Se connecter")
        if submitted:
            if user == USERNAME and pwd == PASSWORD:
                st.session_state["logged_in"] = True
                st.session_state["username"] = user
                safe_rerun()
            else:
                st.error("Nom d'utilisateur ou mot de passe incorrect")
    st.stop()

# ---------- App principale (après login) ----------
st.sidebar.markdown(f"**Connecté :** {st.session_state['username']}")
if st.sidebar.button("Se déconnecter"):
    st.session_state["logged_in"] = False
    st.session_state["username"] = ""
    safe_rerun()

# menu et import dynamique des modules
page = st.sidebar.selectbox("Menu", ["Accueil", "Analyse", "Réunion"])

if page == "Accueil":
    page_module = import_module("accueil")
elif page == "Analyse":
    page_module = import_module("analyse")
elif page == "Réunion":
    page_module = import_module("reunion")
else:
    page_module = None

if page_module:
    page_module.render()

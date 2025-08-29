# login.py
import streamlit as st

def check_login():
    USERNAME = "monuser"        # change ici
    PASSWORD = "monpassword"    # change ici

    username_input = st.text_input("Nom d'utilisateur")
    password_input = st.text_input("Mot de passe", type="password")

    if username_input != USERNAME or password_input != PASSWORD:
        if username_input or password_input:
            st.error("Nom d'utilisateur ou mot de passe incorrect")
        st.stop()  # bloque tout le reste

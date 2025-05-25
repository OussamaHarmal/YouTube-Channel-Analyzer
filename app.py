import streamlit as st
import pandas as pd
import numpy as np
import pickle

st.set_page_config(page_title="Analyse YouTube", layout="centered")

st.title("Analyse de Performance d'une Chaîne YouTube")
st.markdown("Cette application prédit **les vues**, **le temps de visionnage**, et **les revenus estimés** basés sur les caractéristiques de la vidéo.")

@st.cache_resource
def charger_modeles():
    with open("best_model_views.pkl", "rb") as f_views:
        modele_views = pickle.load(f_views)
    with open("best_model_revenue.pkl", "rb") as f_revenue:
        modele_revenue = pickle.load(f_revenue)
    with open("best_model_watch_time.pkl", "rb") as f_watch:
        modele_watch_time = pickle.load(f_watch)
    return modele_views, modele_revenue, modele_watch_time

modele_views, modele_revenue, modele_watch_time = charger_modeles()

st.header("Entrer les caractéristiques de la vidéo")

col1, col2 = st.columns(2)

with col1:
    impressions = st.number_input("Impressions", min_value=0)
    video_duration = st.number_input("Durée de la vidéo (secondes)", min_value=0.0)
    avg_view_duration = st.number_input("Durée moyenne de vue (secondes)", min_value=0.0)
    
    rpm = st.number_input(" Revenu pour 1000 vues (USD)", min_value=0.0, format="%.2f")

with col2:
    shares = st.number_input(" Partages", min_value=0)

# === PRÉDICTION ===
if st.button(" Prédire la performance"):

    
    input_views = pd.DataFrame([{
        'Video Duration': video_duration,
        'Shares': shares,
        'Impressions': impressions
    }])
    vues = modele_views.predict(input_views)[0]
    ctr=(vues/impressions) * 100
    
    input_watch = pd.DataFrame([{
        'Average View Duration': avg_view_duration,
        'CTR (%)': ctr,
        'Impressions': impressions,
        'Video Duration': video_duration
    }])
    watch_time = modele_watch_time.predict(input_watch)[0]

    input_revenue = pd.DataFrame([{
        'Watch Time (hours)': watch_time,
        'Views': vues,
        'CTR (%)': ctr,
        'Revenue per 1000 Views (USD)': rpm
    }])
    revenue = modele_revenue.predict(input_revenue)[0]

    # === Résultats ===
    st.success("Prédictions effectuées avec succès !")
    st.metric("Vues Prédites", f"{vues:,.0f}")
    st.metric("Temps de Visionnage (heures)", f"{watch_time:.2f}")
    st.metric("Revenu Estimé (USD)", f"{revenue:.2f}")

st.markdown("---")
st.markdown("Développé dans le cadre d'un **PFE 2025** | Utilise : `XGBoost`, `RandomForest`, `CatBoost`, `Streamlit` |Développé par Oussama Harmal ")

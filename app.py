import streamlit as st
import pandas as pd
import numpy as np
import joblib

# 1. Configuration de la page (Doit être la première ligne)
st.set_page_config(
    page_title="ART PREDICTION INCENDIE AI",
    page_icon="🌲",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- STYLE CSS PERSONNALISÉ ---
st.markdown("""
    <style>
    .big-font {
        font-size:30px !important;
        font-weight: bold;
    }
    .stButton>button {
        width: 100%;
        background-color: #FF4B4B;
        color: white;
        height: 3em;
        font-size: 20px;
    }
    </style>
    """, unsafe_allow_html=True)

# 2. Chargement des ressources
@st.cache_resource
def load_assets():
    try:
        model_loaded = joblib.load('modele_foret.pkl')
        scaler_loaded = joblib.load('scaler.pkl')
        return model_loaded, scaler_loaded
    except FileNotFoundError:
        return None, None

model, scaler = load_assets()

# Vérification des fichiers
if model is None or scaler is None:
    st.error("🚨 Erreur : Fichiers manquants. Assurez-vous d'avoir 'modele_foret.pkl' ET 'scaler.pkl'.")
    st.stop()

# --- BARRE LATÉRALE (SIDEBAR) ---
with st.sidebar:
    st.title("⚙️ Welcom")
    st.markdown("Ajustez les conditions météorologiques ci-dessous.")
    
    st.divider()
    
    st.subheader("🌦️ Météo")
    temperature = st.slider("Température (°C)", 0, 50, 35)
    rh = st.slider("Humidité Relative (%)", 0, 100, 40)
    ws = st.slider("Vent (km/h)", 0, 50, 15)
    rain = st.number_input("Pluie (mm)", 0.0, 50.0, 0.0, step=0.1)
    
    st.divider()
    
    st.subheader("🔥 Indices FWI")
    with st.expander("Voir les indices techniques (Avancé)", expanded=False):
        ffmc = st.slider("FFMC", 0.0, 100.0, 85.0)
        dmc = st.slider("DMC", 0.0, 100.0, 25.0)
        dc = st.slider("DC", 0.0, 200.0, 60.0)
        isi = st.slider("ISI", 0.0, 30.0, 8.0)
        bui = st.slider("BUI", 0.0, 100.0, 30.0)
        fwi = st.slider("FWI (Global)", 0.0, 50.0, 15.0)

# --- PAGE PRINCIPALE ---

# En-tête
st.title("ART PREDICTION INCENDIE AI")
st.markdown("### Système intelligent de prédiction des incendies de forêt")
st.markdown("Ce système analyse les conditions météorologiques en temps réel pour estimer le risque d'incendie.")

st.divider()

# Section Résultat (Vide par défaut)
col_res1, col_res2 = st.columns([2, 1])

with col_res1:
    st.info("👈 Modifiez les paramètres dans le menu de gauche puis cliquez sur 'Analyser'.")

# Bouton d'action
if st.sidebar.button("🔍 Analyser le Risque", type="primary"):
    
    # Préparation des données
    input_data = [temperature, rh, ws, rain, ffmc, dmc, dc, isi, bui, fwi]
    features = np.array([input_data])
    
    # Scaling et Prédiction
    features_scaled = scaler.transform(features)
    prediction = model.predict(features_scaled)
    probability = model.predict_proba(features_scaled)[0][1]
    
    # --- AFFICHAGE DU RÉSULTAT ---
    
    # Nettoyage de la colonne principale pour afficher le résultat
    with col_res1:
        st.write("") # Espace vide
        if prediction[0] == 1:
            # Design Rouge (Danger)
            st.error("### ⚠️ ALERTE : RISQUE D'INCENDIE ÉLEVÉ")
            st.markdown(f"""
                <div style="padding: 20px; background-color: #ffebeb; border-radius: 10px; border: 1px solid #ff4b4b;">
                    <h3 style="color: #bf0000; margin:0;">Analyse critique</h3>
                    <p style="color: #333;">Les conditions actuelles sont favorables au déclenchement d'un incendie.</p>
                </div>
            """, unsafe_allow_html=True)
        else:
            # Design Vert (Sûr)
            st.success("### ✅ CONDITIONS SÛRES")
            st.markdown(f"""
                <div style="padding: 20px; background-color: #e8f5e9; border-radius: 10px; border: 1px solid #4caf50;">
                    <h3 style="color: #2e7d32; margin:0;">Situation Normale</h3>
                    <p style="color: #333;">Les conditions météorologiques ne présentent pas de risque immédiat.</p>
                </div>
            """, unsafe_allow_html=True)

    # Affichage des métriques (Jauges)
    with col_res2:
        st.write("### Statistiques")
        st.metric(label="Probabilité de Feu", value=f"{probability:.1%}", delta=f"{probability:.1%}")
        st.metric(label="Température", value=f"{temperature}°C")
        st.metric(label="Indice FWI", value=f"{fwi}")
        
        # Barre de progression personnalisée
        st.write("Niveau de menace :")
        if probability > 0.5:
            st.progress(probability, text="CRITIQUE")
        else:
            st.progress(probability, text="FAIBLE")
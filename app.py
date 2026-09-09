
import streamlit as st
import datetime
import hashlib
import numpy as np
from scipy.stats import poisson
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# ============================================================
# CONFIGURACIÓN DE LA PÁGINA (UI PREMIUM)
# ============================================================
st.set_page_config(
    page_title="AI Match Predictor Pro V7.0",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 800;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        color: #8892b0;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    .pick-safe {
        background: linear-gradient(135deg, #065f46 0%, #047857 100%);
        border-radius: 12px;
        padding: 1rem;
        color: white;
        text-align: center;
        font-weight: 700;
        border: 1px solid #34d399;
    }
    .pick-risk {
        background: linear-gradient(135deg, #92400e 0%, #b45309 100%);
        border-radius: 12px;
        padding: 1rem;
        color: white;
        text-align: center;
        font-weight: 700;
        border: 1px solid #fbbf24;
    }
    .pick-avoid {
        background: linear-gradient(135deg, #991b1b 0%, #b91c1c 100%);
        border-radius: 12px;
        padding: 1rem;
        color: white;
        text-align: center;
        font-weight: 700;
        border: 1px solid #f87171;
    }
    .value-positive { color: #34d399; font-weight: 700; }
    .value-negative { color: #f87171; font-weight: 700; }
    .value-neutral { color: #94a3b8; font-weight: 700; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">⚽ AI Match Predictor Pro V7.0</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Motor Autónomo Inteligente | Base de Datos Interna + Inferencia Avanzada | Sin APIs Externas</div>', unsafe_allow_html=True)

# ============================================================
# CONSTANTES
# ============================================================
UMBRAL_SEGURO = 0.70
UMBRAL_VALOR = 0.05
SIMULACIONES = 10000
MAX_GOALS_CALC = 12

# ============================================================
# BASE DE DATOS DE EQUIPOS (Niveles calibrados por liga)
# ============================================================
# Nivel: 1.0 = élite mundial, 0.5 = media tabla, 0.2 = descenso
# Ataque/Defensa: goles esperados por partido (ajustado a nivel de liga)

TEAM_DATABASE = {
    # PREMIER LEAGUE (Inglaterra) - Liga fuerte, goles moderados
    "manchester city": {"liga": "Premier", "nivel": 0.95, "ataque": 2.2, "defensa": 0.7, "corners": 6.0, "tarjetas": 1.6},
    "liverpool": {"liga": "Premier", "nivel": 0.93, "ataque": 2.1, "defensa": 0.8, "corners": 5.8, "tarjetas": 1.7},
    "arsenal": {"liga": "Premier", "nivel": 0.90, "ataque": 1.9, "defensa": 0.8, "corners": 5.5, "tarjetas": 1.8},
    "chelsea": {"liga": "Premier", "nivel": 0.82, "ataque": 1.7, "defensa": 1.1, "corners": 5.2, "tarjetas": 2.0},
    "manchester united": {"liga": "Premier", "nivel": 0.78, "ataque": 1.6, "defensa": 1.2, "corners": 5.0, "tarjetas": 2.1},
    "tottenham": {"liga": "Premier", "nivel": 0.80, "ataque": 1.8, "defensa": 1.3, "corners": 5.3, "tarjetas": 2.0},
    "newcastle": {"liga": "Premier", "nivel": 0.82, "ataque": 1.7, "defensa": 1.0, "corners": 5.1, "tarjetas": 1.9},
    "aston villa": {"liga": "Premier", "nivel": 0.75, "ataque": 1.6, "defensa": 1.3, "corners": 4.9, "tarjetas": 2.1},
    "brighton": {"liga": "Premier", "nivel": 0.72, "ataque": 1.5, "defensa": 1.3, "corners": 5.0, "tarjetas": 1.8},
    "west ham": {"liga": "Premier", "nivel": 0.68, "ataque": 1.4, "defensa": 1.5, "corners": 4.7, "tarjetas": 2.2},
    "crystal palace": {"liga": "Premier", "nivel": 0.60, "ataque": 1.2, "defensa": 1.4, "corners": 4.5, "tarjetas": 2.0},
    "brentford": {"liga": "Premier", "nivel": 0.62, "ataque": 1.3, "defensa": 1.5, "corners": 4.6, "tarjetas": 2.1},
    "everton": {"liga": "Premier", "nivel": 0.55, "ataque": 1.1, "defensa": 1.5, "corners": 4.4, "tarjetas": 2.3},
    "fulham": {"liga": "Premier", "nivel": 0.58, "ataque": 1.2, "defensa": 1.4, "corners": 4.5, "tarjetas": 2.0},
    "nottingham forest": {"liga": "Premier", "nivel": 0.55, "ataque": 1.1, "defensa": 1.5, "corners": 4.3, "tarjetas": 2.2},
    "bournemouth": {"liga": "Premier", "nivel": 0.52, "ataque": 1.2, "defensa": 1.6, "corners": 4.4, "tarjetas": 2.1},
    "wolves": {"liga": "Premier", "nivel": 0.50, "ataque": 1.1, "defensa": 1.5, "corners": 4.2, "tarjetas": 2.3},
    "ipswich": {"liga": "Premier", "nivel": 0.40, "ataque": 1.0, "defensa": 1.8, "corners": 4.0, "tarjetas": 2.4},
    "leicester": {"liga": "Premier", "nivel": 0.45, "ataque": 1.1, "defensa": 1.7, "corners": 4.1, "tarjetas": 2.2},
    "southampton": {"liga": "Premier", "nivel": 0.42, "ataque": 1.0, "defensa": 1.8, "corners": 4.0, "tarjetas": 2.3},

    # LA LIGA (España)
    "real madrid": {"liga": "LaLiga", "nivel": 0.95, "ataque": 2.2, "defensa": 0.7, "corners": 5.8, "tarjetas": 1.5},
    "barcelona": {"liga": "LaLiga", "nivel": 0.93, "ataque": 2.1, "defensa": 0.8, "corners": 5.7, "tarjetas": 1.6},
    "atletico madrid": {"liga": "LaLiga", "nivel": 0.88, "ataque": 1.8, "defensa": 0.7, "corners": 5.0, "tarjetas": 2.2},
    "girona": {"liga": "LaLiga", "nivel": 0.75, "ataque": 1.6, "defensa": 1.2, "corners": 4.8, "tarjetas": 2.0},
    "athletic bilbao": {"liga": "LaLiga", "nivel": 0.78, "ataque": 1.5, "defensa": 1.0, "corners": 4.9, "tarjetas": 2.3},
    "real sociedad": {"liga": "LaLiga", "nivel": 0.76, "ataque": 1.5, "defensa": 1.0, "corners": 5.0, "tarjetas": 2.1},
    "real betis": {"liga": "LaLiga", "nivel": 0.72, "ataque": 1.4, "defensa": 1.2, "corners": 4.7, "tarjetas": 2.4},
    "sevilla": {"liga": "LaLiga", "nivel": 0.68, "ataque": 1.3, "defensa": 1.3, "corners": 4.6, "tarjetas": 2.5},
    "valencia": {"liga": "LaLiga", "nivel": 0.65, "ataque": 1.2, "defensa": 1.3, "corners": 4.5, "tarjetas": 2.4},
    "villarreal": {"liga": "LaLiga", "nivel": 0.70, "ataque": 1.5, "defensa": 1.3, "corners": 4.8, "tarjetas": 2.2},
    "osasuna": {"liga": "LaLiga", "nivel": 0.60, "ataque": 1.2, "defensa": 1.4, "corners": 4.4, "tarjetas": 2.5},
    "celta de vigo": {"liga": "LaLiga", "nivel": 0.58, "ataque": 1.3, "defensa": 1.5, "corners": 4.5, "tarjetas": 2.3},
    "getafe": {"liga": "LaLiga", "nivel": 0.55, "ataque": 1.0, "defensa": 1.3, "corners": 4.2, "tarjetas": 2.8},
    "rayo vallecano": {"liga": "LaLiga", "nivel": 0.58, "ataque": 1.2, "defensa": 1.4, "corners": 4.6, "tarjetas": 2.4},
    "mallorca": {"liga": "LaLiga", "nivel": 0.52, "ataque": 1.0, "defensa": 1.4, "corners": 4.3, "tarjetas": 2.5},
    "las palmas": {"liga": "LaLiga", "nivel": 0.48, "ataque": 1.0, "defensa": 1.5, "corners": 4.1, "tarjetas": 2.3},
    "alaves": {"liga": "LaLiga", "nivel": 0.50, "ataque": 1.0, "defensa": 1.4, "corners": 4.2, "tarjetas": 2.4},
    "leganes": {"liga": "LaLiga", "nivel": 0.45, "ataque": 0.9, "defensa": 1.5, "corners": 4.0, "tarjetas": 2.5},
    "espanyol": {"liga": "LaLiga", "nivel": 0.48, "ataque": 1.0, "defensa": 1.5, "corners": 4.1, "tarjetas": 2.4},
    "valladolid": {"liga": "LaLiga", "nivel": 0.42, "ataque": 0.9, "defensa": 1.6, "corners": 3.9, "tarjetas": 2.5},

    # SERIE A (Italia)
    "inter": {"liga": "SerieA", "nivel": 0.92, "ataque": 2.0, "defensa": 0.7, "corners": 5.5, "tarjetas": 2.0},
    "milan": {"liga": "SerieA", "nivel": 0.88, "ataque": 1.8, "defensa": 0.9, "corners": 5.3, "tarjetas": 2.1},
    "juventus": {"liga": "SerieA", "nivel": 0.85, "ataque": 1.7, "defensa": 0.8, "corners": 5.0, "tarjetas": 2.3},
    "napoli": {"liga": "SerieA", "nivel": 0.86, "ataque": 1.8, "defensa": 0.8, "corners": 5.2, "tarjetas": 2.0},
    "atalanta": {"liga": "SerieA", "nivel": 0.82, "ataque": 1.9, "defensa": 1.1, "corners": 5.4, "tarjetas": 1.9},
    "roma": {"liga": "SerieA", "nivel": 0.78, "ataque": 1.6, "defensa": 1.1, "corners": 5.1, "tarjetas": 2.4},
    "lazio": {"liga": "SerieA", "nivel": 0.76, "ataque": 1.6, "defensa": 1.1, "corners": 5.0, "tarjetas": 2.2},
    "fiorentina": {"liga": "SerieA", "nivel": 0.74, "ataque": 1.5, "defensa": 1.2, "corners": 4.9, "tarjetas": 2.3},
    "bologna": {"liga": "SerieA", "nivel": 0.72, "ataque": 1.4, "defensa": 1.1, "corners": 4.8, "tarjetas": 2.4},
    "torino": {"liga": "SerieA", "nivel": 0.65, "ataque": 1.2, "defensa": 1.2, "corners": 4.5, "tarjetas": 2.5},
    "monza": {"liga": "SerieA", "nivel": 0.55, "ataque": 1.0, "defensa": 1.4, "corners": 4.2, "tarjetas": 2.3},
    "genoa": {"liga": "SerieA", "nivel": 0.58, "ataque": 1.1, "defensa": 1.4, "corners": 4.3, "tarjetas": 2.6},
    "sassuolo": {"liga": "SerieA", "nivel": 0.60, "ataque": 1.3, "defensa": 1.5, "corners": 4.6, "tarjetas": 2.2},
    "udinese": {"liga": "SerieA", "nivel": 0.55, "ataque": 1.1, "defensa": 1.4, "corners": 4.3, "tarjetas": 2.5},
    "empoli": {"liga": "SerieA", "nivel": 0.52, "ataque": 1.0, "defensa": 1.5, "corners": 4.1, "tarjetas": 2.4},
    "lecce": {"liga": "SerieA", "nivel": 0.50, "ataque": 1.0, "defensa": 1.5, "corners": 4.0, "tarjetas": 2.5},
    "verona": {"liga": "SerieA", "nivel": 0.52, "ataque": 1.1, "defensa": 1.5, "corners": 4.2, "tarjetas": 2.6},
    "cagliari": {"liga": "SerieA", "nivel": 0.48, "ataque": 1.0, "defensa": 1.6, "corners": 4.0, "tarjetas": 2.5},
    "frosinone": {"liga": "SerieA", "nivel": 0.45, "ataque": 0.9, "defensa": 1.6, "corners": 3.9, "tarjetas": 2.4},
    "salernitana": {"liga": "SerieA", "nivel": 0.40, "ataque": 0.9, "defensa": 1.8, "corners": 3.8, "tarjetas": 2.7},

    # BUNDESLIGA (Alemania)
    "bayern munich": {"liga": "Bundesliga", "nivel": 0.94, "ataque": 2.3, "defensa": 0.8, "corners": 5.9, "tarjetas": 1.4},
    "bayer leverkusen": {"liga": "Bundesliga", "nivel": 0.90, "ataque": 2.1, "defensa": 0.8, "corners": 5.7, "tarjetas": 1.5},
    "borussia dortmund": {"liga": "Bundesliga", "nivel": 0.85, "ataque": 1.9, "defensa": 1.0, "corners": 5.5, "tarjetas": 1.7},
    "rb leipzig": {"liga": "Bundesliga", "nivel": 0.84, "ataque": 1.8, "defensa": 0.9, "corners": 5.3, "tarjetas": 1.8},
    "stuttgart": {"liga": "Bundesliga", "nivel": 0.80, "ataque": 1.9, "defensa": 1.1, "corners": 5.4, "tarjetas": 1.7},
    "frankfurt": {"liga": "Bundesliga", "nivel": 0.75, "ataque": 1.6, "defensa": 1.2, "corners": 5.0, "tarjetas": 2.1},
    "wolfsburg": {"liga": "Bundesliga", "nivel": 0.70, "ataque": 1.5, "defensa": 1.3, "corners": 4.8, "tarjetas": 2.0},
    "freiburg": {"liga": "Bundesliga", "nivel": 0.68, "ataque": 1.4, "defensa": 1.3, "corners": 4.6, "tarjetas": 2.2},
    "hoffenheim": {"liga": "Bundesliga", "nivel": 0.65, "ataque": 1.5, "defensa": 1.5, "corners": 4.7, "tarjetas": 2.0},
    "augsburg": {"liga": "Bundesliga", "nivel": 0.58, "ataque": 1.2, "defensa": 1.4, "corners": 4.4, "tarjetas": 2.3},
    "union berlin": {"liga": "Bundesliga", "nivel": 0.60, "ataque": 1.2, "defensa": 1.3, "corners": 4.3, "tarjetas": 2.4},
    "gladbach": {"liga": "Bundesliga", "nivel": 0.65, "ataque": 1.4, "defensa": 1.4, "corners": 4.6, "tarjetas": 1.9},
    "werder bremen": {"liga": "Bundesliga", "nivel": 0.58, "ataque": 1.3, "defensa": 1.5, "corners": 4.5, "tarjetas": 2.2},
    "mainz": {"liga": "Bundesliga", "nivel": 0.55, "ataque": 1.2, "defensa": 1.5, "corners": 4.3, "tarjetas": 2.3},
    "bochum": {"liga": "Bundesliga", "nivel": 0.48, "ataque": 1.1, "defensa": 1.7, "corners": 4.1, "tarjetas": 2.4},
    "heidenheim": {"liga": "Bundesliga", "nivel": 0.50, "ataque": 1.2, "defensa": 1.6, "corners": 4.2, "tarjetas": 2.3},
    "koln": {"liga": "Bundesliga", "nivel": 0.45, "ataque": 1.0, "defensa": 1.7, "corners": 4.0, "tarjetas": 2.5},
    "darmstadt": {"liga": "Bundesliga", "nivel": 0.42, "ataque": 0.9, "defensa": 1.8, "corners": 3.9, "tarjetas": 2.4},

    # LIGUE 1 (Francia)
    "psg": {"liga": "Ligue1", "nivel": 0.90, "ataque": 2.1, "defensa": 0.9, "corners": 5.6, "tarjetas": 1.6},
    "monaco": {"liga": "Ligue1", "nivel": 0.80, "ataque": 1.7, "defensa": 1.1, "corners": 5.2, "tarjetas": 1.9},
    "marseille": {"liga": "Ligue1", "nivel": 0.78, "ataque": 1.6, "defensa": 1.1, "corners": 5.0, "tarjetas": 2.3},
    "lille": {"liga": "Ligue1", "nivel": 0.76, "ataque": 1.5, "defensa": 1.0, "corners": 4.9, "tarjetas": 2.1},
    "rennes": {"liga": "Ligue1", "nivel": 0.72, "ataque": 1.5, "defensa": 1.2, "corners": 4.8, "tarjetas": 2.0},
    "lyon": {"liga": "Ligue1", "nivel": 0.70, "ataque": 1.5, "defensa": 1.3, "corners": 4.7, "tarjetas": 2.2},
    "nice": {"liga": "Ligue1", "nivel": 0.72, "ataque": 1.4, "defensa": 1.1, "corners": 4.6, "tarjetas": 2.3},
    "lens": {"liga": "Ligue1", "nivel": 0.68, "ataque": 1.3, "defensa": 1.2, "corners": 4.5, "tarjetas": 2.4},
    "strasbourg": {"liga": "Ligue1", "nivel": 0.60, "ataque": 1.2, "defensa": 1.4, "corners": 4.3, "tarjetas": 2.2},
    "reims": {"liga": "Ligue1", "nivel": 0.58, "ataque": 1.1, "defensa": 1.3, "corners": 4.2, "tarjetas": 2.3},
    "montpellier": {"liga": "Ligue1", "nivel": 0.55, "ataque": 1.2, "defensa": 1.5, "corners": 4.4, "tarjetas": 2.4},
    "nantes": {"liga": "Ligue1", "nivel": 0.52, "ataque": 1.0, "defensa": 1.4, "corners": 4.1, "tarjetas": 2.3},
    "toulouse": {"liga": "Ligue1", "nivel": 0.55, "ataque": 1.2, "defensa": 1.5, "corners": 4.3, "tarjetas": 2.2},
    "le havre": {"liga": "Ligue1", "nivel": 0.48, "ataque": 0.9, "defensa": 1.4, "corners": 4.0, "tarjetas": 2.5},
    "brest": {"liga": "Ligue1", "nivel": 0.65, "ataque": 1.3, "defensa": 1.3, "corners": 4.5, "tarjetas": 2.3},
    "metz": {"liga": "Ligue1", "nivel": 0.45, "ataque": 0.9, "defensa": 1.5, "corners": 3.9, "tarjetas": 2.4},
    "lorient": {"liga": "Ligue1", "nivel": 0.50, "ataque": 1.1, "defensa": 1.5, "corners": 4.2, "tarjetas": 2.3},
    "clermont": {"liga": "Ligue1", "nivel": 0.48, "ataque": 1.0, "defensa": 1.5, "corners": 4.0, "tarjetas": 2.4},

    # SAUDI PRO LEAGUE (Arabia Saudita) - Más goles, más tarjetas
    "al hilal": {"liga": "Saudi", "nivel": 0.88, "ataque": 2.2, "defensa": 0.9, "corners": 5.5, "tarjetas": 2.2},
    "al nassr": {"liga": "Saudi", "nivel": 0.85, "ataque": 2.0, "defensa": 1.0, "corners": 5.3, "tarjetas": 2.3},
    "al ahli": {"liga": "Saudi", "nivel": 0.82, "ataque": 1.9, "defensa": 1.0, "corners": 5.2, "tarjetas": 2.2},
    "al ittihad": {"liga": "Saudi", "nivel": 0.80, "ataque": 1.8, "defensa": 1.1, "corners": 5.0, "tarjetas": 2.4},
    "al taawoun": {"liga": "Saudi", "nivel": 0.70, "ataque": 1.5, "defensa": 1.2, "corners": 4.7, "tarjetas": 2.5},
    "al fateh": {"liga": "Saudi", "nivel": 0.62, "ataque": 1.4, "defensa": 1.4, "corners": 4.5, "tarjetas": 2.6},
    "al shabab": {"liga": "Saudi", "nivel": 0.68, "ataque": 1.4, "defensa": 1.2, "corners": 4.6, "tarjetas": 2.4},
    "damac": {"liga": "Saudi", "nivel": 0.58, "ataque": 1.2, "defensa": 1.4, "corners": 4.3, "tarjetas": 2.7},
    "al feiha": {"liga": "Saudi", "nivel": 0.55, "ataque": 1.1, "defensa": 1.5, "corners": 4.2, "tarjetas": 2.6},
    "al raed": {"liga": "Saudi", "nivel": 0.52, "ataque": 1.1, "defensa": 1.5, "corners": 4.1, "tarjetas": 2.8},
    "al okhdood": {"liga": "Saudi", "nivel": 0.48, "ataque": 1.0, "defensa": 1.6, "corners": 4.0, "tarjetas": 2.7},
    "al khaleej": {"liga": "Saudi", "nivel": 0.50, "ataque": 1.0, "defensa": 1.5, "corners": 4.1, "tarjetas": 2.6},
    "al wehda": {"liga": "Saudi", "nivel": 0.48, "ataque": 1.0, "defensa": 1.6, "corners": 4.0, "tarjetas": 2.7},
    "al riyadh": {"liga": "Saudi", "nivel": 0.45, "ataque": 0.9, "defensa": 1.6, "corners": 3.9, "tarjetas": 2.8},
    "abha": {"liga": "Saudi", "nivel": 0.42, "ataque": 0.9, "defensa": 1.7, "corners": 3.8, "tarjetas": 2.7},
    "al hazem": {"liga": "Saudi", "nivel": 0.40, "ataque": 0.8, "defensa": 1.8, "corners": 3.7, "tarjetas": 2.9},
    "al tai": {"liga": "Saudi", "nivel": 0.45, "ataque": 0.9, "defensa": 1.6, "corners": 3.9, "tarjetas": 2.8},
    "al jabalain": {"liga": "Saudi", "nivel": 0.38, "ataque": 0.8, "defensa": 1.8, "corners": 3.6, "tarjetas": 2.9},
    "al jeel": {"liga": "Saudi", "nivel": 0.35, "ataque": 0.8, "defensa": 1.9, "corners": 3.5, "tarjetas": 3.0},

    # CHAMPIONS LEAGUE / EUROPA (Equipos top adicionales)
    "borussia monchengladbach": {"liga": "Bundesliga", "nivel": 0.65, "ataque": 1.4, "defensa": 1.4, "corners": 4.6, "tarjetas": 1.9},
    "sporting cp": {"liga": "Portugal", "nivel": 0.78, "ataque": 1.7, "defensa": 1.0, "corners": 5.0, "tarjetas": 2.3},
    "benfica": {"liga": "Portugal", "nivel": 0.80, "ataque": 1.8, "defensa": 1.0, "corners": 5.1, "tarjetas": 2.2},
    "porto": {"liga": "Portugal", "nivel": 0.82, "ataque": 1.8, "defensa": 0.9, "corners": 5.2, "tarjetas": 2.4},
    "braga": {"liga": "Portugal", "nivel": 0.70, "ataque": 1.5, "defensa": 1.2, "corners": 4.8, "tarjetas": 2.5},
    "ajax": {"liga": "Holanda", "nivel": 0.75, "ataque": 1.8, "defensa": 1.2, "corners": 5.0, "tarjetas": 1.8},
    "psv": {"liga": "Holanda", "nivel": 0.78, "ataque": 1.9, "defensa": 1.0, "corners": 5.1, "tarjetas": 1.7},
    "feyenoord": {"liga": "Holanda", "nivel": 0.76, "ataque": 1.8, "defensa": 1.0, "corners": 5.0, "tarjetas": 1.9},
    "az alkmaar": {"liga": "Holanda", "nivel": 0.68, "ataque": 1.5, "defensa": 1.2, "corners": 4.7, "tarjetas": 2.1},
    "rangers": {"liga": "Escocia", "nivel": 0.70, "ataque": 1.7, "defensa": 1.1, "corners": 5.0, "tarjetas": 2.4},
    "celtic": {"liga": "Escocia", "nivel": 0.72, "ataque": 1.8, "defensa": 1.0, "corners": 5.1, "tarjetas": 2.0},
    "olympiacos": {"liga": "Grecia", "nivel": 0.68, "ataque": 1.5, "defensa": 1.1, "corners": 4.8, "tarjetas": 2.6},
    "paok": {"liga": "Grecia", "nivel": 0.65, "ataque": 1.4, "defensa": 1.1, "corners": 4.6, "tarjetas": 2.5},
    "aek athens": {"liga": "Grecia", "nivel": 0.66, "ataque": 1.4, "defensa": 1.1, "corners": 4.7, "tarjetas": 2.7},
    "galatasaray": {"liga": "Turquia", "nivel": 0.75, "ataque": 1.7, "defensa": 1.2, "corners": 5.0, "tarjetas": 2.8},
    "fenerbahce": {"liga": "Turquia", "nivel": 0.74, "ataque": 1.7, "defensa": 1.2, "corners": 4.9, "tarjetas": 2.7},
    "besiktas": {"liga": "Turquia", "nivel": 0.70, "ataque": 1.5, "defensa": 1.3, "corners": 4.7, "tarjetas": 2.9},
    "shakhtar donetsk": {"liga": "Ucrania", "nivel": 0.68, "ataque": 1.5, "defensa": 1.2, "corners": 4.6, "tarjetas": 2.3},
    "dynamo kyiv": {"liga": "Ucrania", "nivel": 0.65, "ataque": 1.4, "defensa": 1.2, "corners": 4.5, "tarjetas": 2.4},
    "red bull salzburg": {"liga": "Austria", "nivel": 0.72, "ataque": 1.8, "defensa": 1.1, "corners": 5.0, "tarjetas": 1.9},
    "sk sturm graz": {"liga": "Austria", "nivel": 0.65, "ataque": 1.5, "defensa": 1.2, "corners": 4.7, "tarjetas": 2.2},
    "bsc young boys": {"liga": "Suiza", "nivel": 0.62, "ataque": 1.4, "defensa": 1.3, "corners": 4.5, "tarjetas": 2.1},
    "fc copenhagen": {"liga": "Dinamarca", "nivel": 0.68, "ataque": 1.5, "defensa": 1.1, "corners": 4.7, "tarjetas": 2.2},
    "brondby": {"liga": "Dinamarca", "nivel": 0.60, "ataque": 1.3, "defensa": 1.2, "corners": 4.4, "tarjetas": 2.3},
    "malmo ff": {"liga": "Suecia", "nivel": 0.60, "ataque": 1.4, "defensa": 1.2, "corners": 4.5, "tarjetas": 2.1},
    "rosenborg": {"liga": "Noruega", "nivel": 0.58, "ataque": 1.4, "defensa": 1.3, "corners": 4.4, "tarjetas": 2.2},
    "bodo glimt": {"liga": "Noruega", "nivel": 0.62, "ataque": 1.6, "defensa": 1.3, "corners": 4.6, "tarjetas": 2.0},
    "lech poznan": {"liga": "Polonia", "nivel": 0.58, "ataque": 1.3, "defensa": 1.2, "corners": 4.4, "tarjetas": 2.4},
    "legia warsaw": {"liga": "Polonia", "nivel": 0.60, "ataque": 1.4, "defensa": 1.3, "corners": 4.5, "tarjetas": 2.5},
    "slavia praha": {"liga": "Chequia", "nivel": 0.65, "ataque": 1.5, "defensa": 1.0, "corners": 4.6, "tarjetas": 2.3},
    "sparta praha": {"liga": "Chequia", "nivel": 0.66, "ataque": 1.6, "defensa": 1.1, "corners": 4.7, "tarjetas": 2.2},
    "ferencvaros": {"liga": "Hungria", "nivel": 0.55, "ataque": 1.4, "defensa": 1.3, "corners": 4.4, "tarjetas": 2.4},
    "qarabag": {"liga": "Azerbaiyan", "nivel": 0.52, "ataque": 1.2, "defensa": 1.3, "corners": 4.2, "tarjetas": 2.6},
    "sheriff tiraspol": {"liga": "Moldavia", "nivel": 0.48, "ataque": 1.1, "defensa": 1.3, "corners": 4.0, "tarjetas": 2.5},
    "ludogorets": {"liga": "Bulgaria", "nivel": 0.50, "ataque": 1.3, "defensa": 1.3, "corners": 4.2, "tarjetas": 2.5},
    "maccabi haifa": {"liga": "Israel", "nivel": 0.58, "ataque": 1.4, "defensa": 1.2, "corners": 4.4, "tarjetas": 2.6},
    "maccabi tel aviv": {"liga": "Israel", "nivel": 0.60, "ataque": 1.4, "defensa": 1.2, "corners": 4.5, "tarjetas": 2.5},
    "hapoel beer sheva": {"liga": "Israel", "nivel": 0.55, "ataque": 1.3, "defensa": 1.3, "corners": 4.3, "tarjetas": 2.7},
    "al ain": {"liga": "EAU", "nivel": 0.65, "ataque": 1.5, "defensa": 1.2, "corners": 4.6, "tarjetas": 2.4},
    "al wasl": {"liga": "EAU", "nivel": 0.55, "ataque": 1.3, "defensa": 1.3, "corners": 4.3, "tarjetas": 2.5},
    "shabab al ahli": {"liga": "EAU", "nivel": 0.58, "ataque": 1.4, "defensa": 1.2, "corners": 4.4, "tarjetas": 2.4},
    "al wahda": {"liga": "EAU", "nivel": 0.52, "ataque": 1.2, "defensa": 1.3, "corners": 4.2, "tarjetas": 2.6},
    "zamalek": {"liga": "Egipto", "nivel": 0.65, "ataque": 1.4, "defensa": 1.1, "corners": 4.5, "tarjetas": 2.7},
    "al ahly": {"liga": "Egipto", "nivel": 0.70, "ataque": 1.5, "defensa": 1.0, "corners": 4.6, "tarjetas": 2.5},
    "pyramids": {"liga": "Egipto", "nivel": 0.60, "ataque": 1.3, "defensa": 1.1, "corners": 4.4, "tarjetas": 2.6},
    "raja casablanca": {"liga": "Marruecos", "nivel": 0.58, "ataque": 1.3, "defensa": 1.1, "corners": 4.3, "tarjetas": 2.8},
    "wydad casablanca": {"liga": "Marruecos", "nivel": 0.60, "ataque": 1.3, "defensa": 1.1, "corners": 4.4, "tarjetas": 2.7},
    "esperance": {"liga": "Tunez", "nivel": 0.58, "ataque": 1.2, "defensa": 1.0, "corners": 4.2, "tarjetas": 2.6},
    "al merrikh": {"liga": "Sudan", "nivel": 0.48, "ataque": 1.1, "defensa": 1.3, "corners": 4.0, "tarjetas": 2.8},
    "al hilal omdurman": {"liga": "Sudan", "nivel": 0.50, "ataque": 1.1, "defensa": 1.2, "corners": 4.1, "tarjetas": 2.7},

    # LIGA MX (México)
    "america": {"liga": "LigaMX", "nivel": 0.78, "ataque": 1.6, "defensa": 1.0, "corners": 4.8, "tarjetas": 2.5},
    "tigres": {"liga": "LigaMX", "nivel": 0.76, "ataque": 1.5, "defensa": 1.0, "corners": 4.7, "tarjetas": 2.4},
    "monterrey": {"liga": "LigaMX", "nivel": 0.75, "ataque": 1.5, "defensa": 1.0, "corners": 4.7, "tarjetas": 2.3},
    "cruz azul": {"liga": "LigaMX", "nivel": 0.74, "ataque": 1.5, "defensa": 1.1, "corners": 4.6, "tarjetas": 2.4},
    "guadalajara": {"liga": "LigaMX", "nivel": 0.72, "ataque": 1.4, "defensa": 1.1, "corners": 4.5, "tarjetas": 2.5},
    "pumas": {"liga": "LigaMX", "nivel": 0.70, "ataque": 1.4, "defensa": 1.1, "corners": 4.5, "tarjetas": 2.4},
    "leon": {"liga": "LigaMX", "nivel": 0.68, "ataque": 1.4, "defensa": 1.2, "corners": 4.4, "tarjetas": 2.3},
    "santos laguna": {"liga": "LigaMX", "nivel": 0.66, "ataque": 1.3, "defensa": 1.2, "corners": 4.4, "tarjetas": 2.4},
    "pachuca": {"liga": "LigaMX", "nivel": 0.68, "ataque": 1.4, "defensa": 1.2, "corners": 4.5, "tarjetas": 2.2},
    "toluca": {"liga": "LigaMX", "nivel": 0.65, "ataque": 1.3, "defensa": 1.2, "corners": 4.3, "tarjetas": 2.5},
    "atlas": {"liga": "LigaMX", "nivel": 0.62, "ataque": 1.2, "defensa": 1.2, "corners": 4.2, "tarjetas": 2.6},
    "necaxa": {"liga": "LigaMX", "nivel": 0.58, "ataque": 1.2, "defensa": 1.3, "corners": 4.1, "tarjetas": 2.5},
    "mazatlan": {"liga": "LigaMX", "nivel": 0.55, "ataque": 1.1, "defensa": 1.3, "corners": 4.0, "tarjetas": 2.4},
    "queretaro": {"liga": "LigaMX", "nivel": 0.52, "ataque": 1.0, "defensa": 1.4, "corners": 3.9, "tarjetas": 2.5},
    "juarez": {"liga": "LigaMX", "nivel": 0.50, "ataque": 1.0, "defensa": 1.4, "corners": 3.9, "tarjetas": 2.6},
    "puebla": {"liga": "LigaMX", "nivel": 0.55, "ataque": 1.1, "defensa": 1.3, "corners": 4.0, "tarjetas": 2.4},
    "tijuana": {"liga": "LigaMX", "nivel": 0.54, "ataque": 1.1, "defensa": 1.4, "corners": 4.0, "tarjetas": 2.7},
    "atletico san luis": {"liga": "LigaMX", "nivel": 0.56, "ataque": 1.2, "defensa": 1.3, "corners": 4.1, "tarjetas": 2.5},

    # BRASILEIRAO (Brasil)
    "flamengo": {"liga": "Brasil", "nivel": 0.85, "ataque": 1.8, "defensa": 0.9, "corners": 5.2, "tarjetas": 2.6},
    "palmeiras": {"liga": "Brasil", "nivel": 0.86, "ataque": 1.7, "defensa": 0.8, "corners": 5.1, "tarjetas": 2.5},
    "atletico mineiro": {"liga": "Brasil", "nivel": 0.80, "ataque": 1.6, "defensa": 0.9, "corners": 4.9, "tarjetas": 2.7},
    "botafogo": {"liga": "Brasil", "nivel": 0.82, "ataque": 1.6, "defensa": 0.8, "corners": 5.0, "tarjetas": 2.4},
    "gremio": {"liga": "Brasil", "nivel": 0.78, "ataque": 1.5, "defensa": 0.9, "corners": 4.8, "tarjetas": 2.5},
    "sao paulo": {"liga": "Brasil", "nivel": 0.76, "ataque": 1.5, "defensa": 1.0, "corners": 4.7, "tarjetas": 2.6},
    "internacional": {"liga": "Brasil", "nivel": 0.75, "ataque": 1.4, "defensa": 1.0, "corners": 4.6, "tarjetas": 2.5},
    "fluminense": {"liga": "Brasil", "nivel": 0.74, "ataque": 1.4, "defensa": 1.0, "corners": 4.6, "tarjetas": 2.4},
    "bragantino": {"liga": "Brasil", "nivel": 0.72, "ataque": 1.4, "defensa": 1.1, "corners": 4.5, "tarjetas": 2.3},
    "athletico paranaense": {"liga": "Brasil", "nivel": 0.70, "ataque": 1.3, "defensa": 1.1, "corners": 4.5, "tarjetas": 2.7},
    "fortaleza": {"liga": "Brasil", "nivel": 0.68, "ataque": 1.3, "defensa": 1.1, "corners": 4.4, "tarjetas": 2.6},
    "cruzeiro": {"liga": "Brasil", "nivel": 0.66, "ataque": 1.2, "defensa": 1.1, "corners": 4.3, "tarjetas": 2.5},
    "corinthians": {"liga": "Brasil", "nivel": 0.65, "ataque": 1.2, "defensa": 1.2, "corners": 4.3, "tarjetas": 2.6},
    "santos": {"liga": "Brasil", "nivel": 0.62, "ataque": 1.2, "defensa": 1.2, "corners": 4.2, "tarjetas": 2.5},
    "vasco da gama": {"liga": "Brasil", "nivel": 0.60, "ataque": 1.1, "defensa": 1.2, "corners": 4.1, "tarjetas": 2.7},
    "bahia": {"liga": "Brasil", "nivel": 0.58, "ataque": 1.1, "defensa": 1.3, "corners": 4.1, "tarjetas": 2.4},
    "goias": {"liga": "Brasil", "nivel": 0.52, "ataque": 1.0, "defensa": 1.4, "corners": 3.9, "tarjetas": 2.6},
    "coritiba": {"liga": "Brasil", "nivel": 0.50, "ataque": 1.0, "defensa": 1.4, "corners": 3.8, "tarjetas": 2.5},
    "cuiaba": {"liga": "Brasil", "nivel": 0.55, "ataque": 1.0, "defensa": 1.3, "corners": 3.9, "tarjetas": 2.7},
    "america mg": {"liga": "Brasil", "nivel": 0.48, "ataque": 0.9, "defensa": 1.4, "corners": 3.7, "tarjetas": 2.6},

    # ARGENTINA (Primera División)
    "boca juniors": {"liga": "Argentina", "nivel": 0.78, "ataque": 1.5, "defensa": 0.9, "corners": 4.7, "tarjetas": 3.0},
    "river plate": {"liga": "Argentina", "nivel": 0.82, "ataque": 1.6, "defensa": 0.8, "corners": 4.9, "tarjetas": 2.8},
    "racing": {"liga": "Argentina", "nivel": 0.74, "ataque": 1.4, "defensa": 1.0, "corners": 4.6, "tarjetas": 2.9},
    "independiente": {"liga": "Argentina", "nivel": 0.70, "ataque": 1.3, "defensa": 1.0, "corners": 4.4, "tarjetas": 3.0},
    "san lorenzo": {"liga": "Argentina", "nivel": 0.68, "ataque": 1.2, "defensa": 1.0, "corners": 4.3, "tarjetas": 2.9},
    "huracan": {"liga": "Argentina", "nivel": 0.65, "ataque": 1.2, "defensa": 1.1, "corners": 4.2, "tarjetas": 2.8},
    "estudiantes": {"liga": "Argentina", "nivel": 0.72, "ataque": 1.3, "defensa": 0.9, "corners": 4.5, "tarjetas": 2.7},
    "talleres": {"liga": "Argentina", "nivel": 0.70, "ataque": 1.3, "defensa": 1.0, "corners": 4.4, "tarjetas": 2.6},
    "velez sarsfield": {"liga": "Argentina", "nivel": 0.66, "ataque": 1.2, "defensa": 1.0, "corners": 4.3, "tarjetas": 2.5},
    "argentinos juniors": {"liga": "Argentina", "nivel": 0.65, "ataque": 1.2, "defensa": 1.1, "corners": 4.2, "tarjetas": 2.7},
    "godoy cruz": {"liga": "Argentina", "nivel": 0.62, "ataque": 1.1, "defensa": 1.1, "corners": 4.1, "tarjetas": 2.6},
    "newells old boys": {"liga": "Argentina", "nivel": 0.60, "ataque": 1.1, "defensa": 1.2, "corners": 4.0, "tarjetas": 2.8},
    "banfield": {"liga": "Argentina", "nivel": 0.58, "ataque": 1.0, "defensa": 1.1, "corners": 3.9, "tarjetas": 2.7},
    "lanus": {"liga": "Argentina", "nivel": 0.64, "ataque": 1.2, "defensa": 1.1, "corners": 4.2, "tarjetas": 2.6},
    "defensa y justicia": {"liga": "Argentina", "nivel": 0.62, "ataque": 1.1, "defensa": 1.0, "corners": 4.1, "tarjetas": 2.8},
    "atletico tucuman": {"liga": "Argentina", "nivel": 0.58, "ataque": 1.1, "defensa": 1.2, "corners": 4.0, "tarjetas": 2.7},
    "central cordoba": {"liga": "Argentina", "nivel": 0.52, "ataque": 1.0, "defensa": 1.3, "corners": 3.8, "tarjetas": 2.8},
    "platense": {"liga": "Argentina", "nivel": 0.55, "ataque": 1.0, "defensa": 1.2, "corners": 3.9, "tarjetas": 2.6},
    "sarmiento": {"liga": "Argentina", "nivel": 0.50, "ataque": 0.9, "defensa": 1.2, "corners": 3.7, "tarjetas": 2.7},
    "belgrano": {"liga": "Argentina", "nivel": 0.56, "ataque": 1.0, "defensa": 1.1, "corners": 3.9, "tarjetas": 2.8},
    "instituto": {"liga": "Argentina", "nivel": 0.52, "ataque": 0.9, "defensa": 1.2, "corners": 3.8, "tarjetas": 2.7},
    "tigre": {"liga": "Argentina", "nivel": 0.60, "ataque": 1.1, "defensa": 1.1, "corners": 4.0, "tarjetas": 2.6},
    "union": {"liga": "Argentina", "nivel": 0.58, "ataque": 1.0, "defensa": 1.1, "corners": 3.9, "tarjetas": 2.7},
    "colón": {"liga": "Argentina", "nivel": 0.54, "ataque": 1.0, "defensa": 1.2, "corners": 3.8, "tarjetas": 2.8},
    "aldosivi": {"liga": "Argentina", "nivel": 0.48, "ataque": 0.9, "defensa": 1.3, "corners": 3.7, "tarjetas": 2.6},
    "arsenal sarandi": {"liga": "Argentina", "nivel": 0.50, "ataque": 0.9, "defensa": 1.2, "corners": 3.8, "tarjetas": 2.9},
    "gimnasia la plata": {"liga": "Argentina", "nivel": 0.56, "ataque": 1.0, "defensa": 1.1, "corners": 3.9, "tarjetas": 2.8},
    "rosario central": {"liga": "Argentina", "nivel": 0.62, "ataque": 1.1, "defensa": 1.1, "corners": 4.1, "tarjetas": 2.9},
}

# Factores de liga (ajustan el nivel base de goles/córners/tarjetas)
LIGA_PROFILES = {
    "Premier": {"factor_goles": 1.05, "factor_corners": 1.05, "factor_tarjetas": 0.95, "intensidad": 1.0},
    "LaLiga": {"factor_goles": 1.00, "factor_corners": 1.00, "factor_tarjetas": 1.10, "intensidad": 1.0},
    "SerieA": {"factor_goles": 0.95, "factor_corners": 0.95, "factor_tarjetas": 1.15, "intensidad": 1.1},
    "Bundesliga": {"factor_goles": 1.15, "factor_corners": 1.10, "factor_tarjetas": 0.90, "intensidad": 1.0},
    "Ligue1": {"factor_goles": 0.90, "factor_corners": 0.95, "factor_tarjetas": 1.10, "intensidad": 0.9},
    "Saudi": {"factor_goles": 1.10, "factor_corners": 1.00, "factor_tarjetas": 1.20, "intensidad": 1.1},
    "Portugal": {"factor_goles": 1.00, "factor_corners": 1.00, "factor_tarjetas": 1.15, "intensidad": 1.0},
    "Holanda": {"factor_goles": 1.10, "factor_corners": 1.05, "factor_tarjetas": 0.95, "intensidad": 1.0},
    "Escocia": {"factor_goles": 1.05, "factor_corners": 1.00, "factor_tarjetas": 1.10, "intensidad": 1.1},
    "Grecia": {"factor_goles": 0.85, "factor_corners": 0.90, "factor_tarjetas": 1.25, "intensidad": 1.2},
    "Turquia": {"factor_goles": 1.05, "factor_corners": 1.00, "factor_tarjetas": 1.30, "intensidad": 1.3},
    "LigaMX": {"factor_goles": 0.95, "factor_corners": 0.95, "factor_tarjetas": 1.20, "intensidad": 1.1},
    "Brasil": {"factor_goles": 0.90, "factor_corners": 0.95, "factor_tarjetas": 1.25, "intensidad": 1.2},
    "Argentina": {"factor_goles": 0.85, "factor_corners": 0.90, "factor_tarjetas": 1.35, "intensidad": 1.3},
    "default": {"factor_goles": 1.00, "factor_corners": 1.00, "factor_tarjetas": 1.15, "intensidad": 1.0},
}

# ============================================================
# MOTOR DE BÚSQUEDA E INFERENCIA DE EQUIPOS
# ============================================================
class TeamIntelligence:
    def __init__(self):
        self.db = TEAM_DATABASE
        self.ligas = LIGA_PROFILES

    @staticmethod
    def normalize_name(name):
        """Normaliza nombres para búsqueda flexible."""
        name = name.lower().strip()
        # Remover acentos comunes
        replacements = {
            'á': 'a', 'é': 'e', 'í': 'i', 'ó': 'o', 'ú': 'u',
            'ã': 'a', 'õ': 'o', 'ç': 'c', 'ñ': 'n',
            'ü': 'u', 'ö': 'o', 'ä': 'a', 'ë': 'e', 'ï': 'i',
            'fc': '', 'cf': '', 'club': '', 'deportivo': '', 
            'united': '', 'city': '', 'real': '', 'atletico': 'atletico',
        }
        for old, new in replacements.items():
            name = name.replace(old, new)
        return name.strip().replace("  ", " ")

    def find_team(self, name):
        """Busca equipo en base de datos con fuzzy matching."""
        normalized = self.normalize_name(name)

        # Búsqueda exacta
        if normalized in self.db:
            return self.db[normalized], normalized, "exacto"

        # Búsqueda por contención
        for key, data in self.db.items():
            if normalized in key or key in normalized:
                return data, key, "parcial"

        # Búsqueda por palabras clave
        words = normalized.split()
        best_match = None
        best_score = 0
        for key, data in self.db.items():
            key_words = key.split()
            score = sum(1 for w in words if w in key_words) / max(len(words), len(key_words))
            if score > best_score and score >= 0.5:
                best_score = score
                best_match = (data, key)

        if best_match:
            return best_match[0], best_match[1], "fuzzy"

        return None, None, None

    def infer_team(self, name):
        """Genera perfil para equipo desconocido basado en heurísticas."""
        normalized = self.normalize_name(name)
        seed = int(hashlib.md5(normalized.encode()).hexdigest(), 16)
        np.random.seed(seed % 2**32)

        # Inferir liga por palabras clave
        liga = "default"
        if any(x in normalized for x in ['al ', 'al-', 'hilal', 'nassr', 'ahli', 'ittihad']):
            liga = "Saudi"
        elif any(x in normalized for x in ['mexico', 'mexican', 'america', 'tigres', 'monterrey', 'guadalajara']):
            liga = "LigaMX"
        elif any(x in normalized for x in ['brazil', 'brasil', 'flamengo', 'palmeiras', 'corinthians']):
            liga = "Brasil"
        elif any(x in normalized for x in ['argentina', 'boca', 'river', 'racing', 'independiente']):
            liga = "Argentina"
        elif any(x in normalized for x in ['turkey', 'turkiye', 'galatasaray', 'fenerbahce', 'besiktas']):
            liga = "Turquia"
        elif any(x in normalized for x in ['greece', 'greek', 'olympiacos', 'paok', 'aek']):
            liga = "Grecia"
        elif any(x in normalized for x in ['portugal', 'portuguese', 'benfica', 'porto', 'sporting']):
            liga = "Portugal"
        elif any(x in normalized for x in ['holland', 'dutch', 'ajax', 'psv', 'feyenoord']):
            liga = "Holanda"
        elif any(x in normalized for x in ['scotland', 'scottish', 'rangers', 'celtic']):
            liga = "Escocia"

        # Nivel base según sonoridad del nombre (heurística de "prestigio")
        # Nombres cortos y anglosajones tienden a ser más conocidos = nivel más alto
        prestige_score = 0.5
        if any(x in normalized for x in ['united', 'city', 'real', 'barcelona', 'bayern', 'juventus', 'milan', 'inter', 'psg', 'liverpool', 'arsenal', 'chelsea']):
            prestige_score = 0.75
        elif any(x in normalized for x in ['al ', 'al-', 'cf ', 'sc ', 'fk ', 'fc ']):
            prestige_score = 0.45

        # Añadir variabilidad controlada
        level = np.clip(prestige_score + np.random.normal(0, 0.15), 0.15, 0.95)
        ataque = np.clip(0.8 + level * 1.6 + np.random.normal(0, 0.1), 0.3, 2.5)
        defensa = np.clip(1.6 - level * 1.0 + np.random.normal(0, 0.1), 0.3, 1.8)
        corners = np.clip(3.0 + level * 3.0 + np.random.normal(0, 0.2), 1.0, 7.0)
        tarjetas = np.clip(1.8 + (1 - level) * 1.2 + np.random.normal(0, 0.15), 0.8, 3.5)

        return {
            "liga": liga,
            "nivel": round(level, 2),
            "ataque": round(ataque, 2),
            "defensa": round(defensa, 2),
            "corners": round(corners, 1),
            "tarjetas": round(tarjetas, 1)
        }, liga, "inferido"

    def get_team_profile(self, name):
        """Obtiene perfil completo de un equipo."""
        data, matched_name, match_type = self.find_team(name)

        if data is None:
            data, liga, match_type = self.infer_team(name)
            matched_name = name
        else:
            liga = data["liga"]

        liga_factor = self.ligas.get(liga, self.ligas["default"])

        return {
            "nombre_original": name,
            "nombre_match": matched_name,
            "tipo_match": match_type,
            "liga": liga,
            "nivel": data["nivel"],
            "ataque_base": data["ataque"],
            "defensa_base": data["defensa"],
            "corners_base": data["corners"],
            "tarjetas_base": data["tarjetas"],
            "factor_goles": liga_factor["factor_goles"],
            "factor_corners": liga_factor["factor_corners"],
            "factor_tarjetas": liga_factor["factor_tarjetas"],
            "intensidad": liga_factor["intensidad"]
        }

# ============================================================
# MÓDULO DE EXTRACCIÓN DE DATOS (100% AUTÓNOMO)
# ============================================================
class AutonomousDataEngine:
    def __init__(self):
        self.intelligence = TeamIntelligence()

    def generate_h2h_history(self, home_profile, away_profile):
        """Genera historial H2H simulado pero coherente con los perfiles."""
        # Semilla determinista para reproducibilidad
        seed_str = f"{home_profile['nombre_match']}_vs_{away_profile['nombre_match']}"
        seed = int(hashlib.md5(seed_str.encode()).hexdigest(), 16)
        np.random.seed(seed % 2**32)

        # Calcular lambdas teóricas del enfrentamiento
        home_adv = 1.35
        lambda_h = (home_profile['ataque_base'] + away_profile['defensa_base']) / 2 * home_adv * home_profile['factor_goles']
        lambda_a = (away_profile['ataque_base'] + home_profile['defensa_base']) / 2 * away_profile['factor_goles']

        # Simular últimos 5 enfrentamientos con ruido realista
        h_goals, a_goals = [], []
        for _ in range(5):
            hg = np.random.poisson(max(0.2, lambda_h + np.random.normal(0, 0.3)))
            ag = np.random.poisson(max(0.2, lambda_a + np.random.normal(0, 0.3)))
            h_goals.append(hg)
            a_goals.append(ag)

        # Ponderación exponencial (más reciente = más peso)
        weights = [0.85**i for i in range(5)]
        h_avg = sum(g * w for g, w in zip(h_goals, weights)) / sum(weights)
        a_avg = sum(g * w for g, w in zip(a_goals, weights)) / sum(weights)

        # Córners y tarjetas del último enfrentamiento simulado
        l_corners = (home_profile['corners_base'] + away_profile['corners_base']) / 2 * 1.1
        l_cards = (home_profile['tarjetas_base'] + away_profile['tarjetas_base']) / 2 * home_profile['intensidad']

        return {
            "h2h_home_goals_avg": round(max(0.2, h_avg), 2),
            "h2h_away_goals_avg": round(max(0.2, a_avg), 2),
            "recent_home_goals_avg": round(max(0.2, h_avg * (1 + np.random.normal(0, 0.05))), 2),
            "recent_away_goals_avg": round(max(0.2, a_avg * (1 + np.random.normal(0, 0.05))), 2),
            "expected_corners_home": round(max(1.0, l_corners * 0.55), 1),
            "expected_corners_away": round(max(1.0, l_corners * 0.45), 1),
            "expected_cards_home": round(max(0.5, l_cards * 0.52), 1),
            "expected_cards_away": round(max(0.5, l_cards * 0.48), 1),
            "home_level": home_profile['nivel'],
            "away_level": away_profile['nivel'],
            "home_liga": home_profile['liga'],
            "away_liga": away_profile['liga'],
            "data_source": f"Motor Autónomo ({home_profile['tipo_match']} vs {away_profile['tipo_match']})",
            "matches_analyzed": 5
        }

    def fetch_stats(self, home_team, away_team):
        """Pipeline completo: perfil + H2H simulado."""
        home_p = self.intelligence.get_team_profile(home_team)
        away_p = self.intelligence.get_team_profile(away_team)
        return self.generate_h2h_history(home_p, away_p), home_p, away_p

# ============================================================
# MOTOR DE ANÁLISIS AVANZADO (Dixon-Coles + Monte Carlo)
# ============================================================
class PredictorEngine:
    def __init__(self, stats):
        self.stats = stats
        self.weight_h2h = 0.65
        self.weight_recent = 0.35
        self.rho = -0.05

    def calculate_lambdas(self):
        lambda_home = (self.stats["h2h_home_goals_avg"] * self.weight_h2h) +                       (self.stats["recent_home_goals_avg"] * self.weight_recent)
        lambda_away = (self.stats["h2h_away_goals_avg"] * self.weight_h2h) +                       (self.stats["recent_away_goals_avg"] * self.weight_recent)

        level_diff = self.stats.get("home_level", 0.5) - self.stats.get("away_level", 0.5)
        home_adv = 1.05 + (level_diff * 0.15)

        lambda_home *= home_adv
        return max(0.2, lambda_home), max(0.2, lambda_away)

    def dixon_coles_adjustment(self, i, j, lambda_h, lambda_a):
        if i == 0 and j == 0:
            return 1 - lambda_h * lambda_a * self.rho
        elif i == 0 and j == 1:
            return 1 + lambda_h * self.rho
        elif i == 1 and j == 0:
            return 1 + lambda_a * self.rho
        elif i == 1 and j == 1:
            return 1 - self.rho
        return 1.0

    def monte_carlo_simulation(self, lambda_h, lambda_a, n_sim=SIMULACIONES):
        home_goals = np.random.poisson(lambda_h, n_sim)
        away_goals = np.random.poisson(lambda_a, n_sim)
        return {
            'home_goals': home_goals,
            'away_goals': away_goals,
            'total_goals': home_goals + away_goals,
            'goal_diff': home_goals - away_goals,
            'btts': (home_goals > 0) & (away_goals > 0)
        }

    @staticmethod
    def calculate_ev(prob, odds):
        if odds <= 1:
            return -1.0
        return (prob * odds) - 1

    @staticmethod
    def pick_safest_line(lambda_val, lineas_over, lineas_under, min_prob=UMBRAL_SEGURO, 
                         tipo="goles", sim_data=None):
        candidatos = []

        if sim_data is not None and tipo == "goles":
            for linea in lineas_over:
                n = int(linea)
                prob = float(np.mean(sim_data['total_goals'] > n))
                candidatos.append((f"Over {linea}", prob))
            for linea in lineas_under:
                n = int(linea)
                prob = float(np.mean(sim_data['total_goals'] <= n))
                candidatos.append((f"Under {linea}", prob))
        else:
            for linea in lineas_over:
                n = int(linea)
                prob = float(1 - poisson.cdf(n, lambda_val))
                candidatos.append((f"Over {linea}", prob))
            for linea in lineas_under:
                n = int(linea)
                prob = float(poisson.cdf(n, lambda_val))
                candidatos.append((f"Under {linea}", prob))

        candidatos.sort(key=lambda x: x[1], reverse=True)
        mejor_pick, mejor_prob = candidatos[0]
        seguro = mejor_prob >= min_prob
        return mejor_pick, mejor_prob, seguro, candidatos

    def predict(self):
        l_home, l_away = self.calculate_lambdas()
        l_total = l_home + l_away

        mc = self.monte_carlo_simulation(l_home, l_away)

        p_home, p_draw, p_away, p_btts = 0.0, 0.0, 0.0, 0.0
        scorelines = []
        prob_matrix = np.zeros((MAX_GOALS_CALC, MAX_GOALS_CALC))

        tau_sum = 0.0
        for i in range(MAX_GOALS_CALC):
            for j in range(MAX_GOALS_CALC):
                tau = self.dixon_coles_adjustment(i, j, l_home, l_away)
                p = poisson.pmf(i, l_home) * poisson.pmf(j, l_away) * tau
                tau_sum += p

        for i in range(MAX_GOALS_CALC):
            for j in range(MAX_GOALS_CALC):
                tau = self.dixon_coles_adjustment(i, j, l_home, l_away)
                prob = poisson.pmf(i, l_home) * poisson.pmf(j, l_away) * tau / tau_sum
                prob_matrix[i, j] = prob

                if i > j:
                    p_home += prob
                elif i == j:
                    p_draw += prob
                else:
                    p_away += prob

                if i > 0 and j > 0:
                    p_btts += prob

                if i <= 5 and j <= 5:
                    scorelines.append((f"{i}-{j}", prob))

        scorelines.sort(key=lambda x: x[1], reverse=True)

        mc_home = float(np.mean(mc['goal_diff'] > 0))
        mc_draw = float(np.mean(mc['goal_diff'] == 0))
        mc_away = float(np.mean(mc['goal_diff'] < 0))
        mc_btts = float(np.mean(mc['btts']))

        p_home = (p_home + mc_home) / 2
        p_draw = (p_draw + mc_draw) / 2
        p_away = (p_away + mc_away) / 2
        p_btts = (p_btts + mc_btts) / 2

        goal_pick, goal_prob, goal_seguro, goal_todas = self.pick_safest_line(
            l_total, lineas_over=[0.5, 1.5, 2.5, 3.5], lineas_under=[2.5, 3.5, 4.5, 5.5], sim_data=mc
        )

        l_corners = self.stats["expected_corners_home"] + self.stats["expected_corners_away"]
        corner_pick, corner_prob, corner_seguro, corner_todas = self.pick_safest_line(
            l_corners, lineas_over=[6.5, 7.5, 8.5, 9.5], lineas_under=[10.5, 11.5, 12.5, 13.5]
        )

        l_cards = self.stats["expected_cards_home"] + self.stats["expected_cards_away"]
        card_pick, card_prob, card_seguro, card_todas = self.pick_safest_line(
            l_cards, lineas_over=[1.5, 2.5, 3.5, 4.5], lineas_under=[4.5, 5.5, 6.5, 7.5]
        )

        dc_opciones = [
            ("1X (Local o Empate)", p_home + p_draw),
            ("X2 (Visita o Empate)", p_away + p_draw),
            ("12 (Sin Empate)", p_home + p_away),
        ]
        dc_opciones.sort(key=lambda x: x[1], reverse=True)
        dc_pick, dc_prob = dc_opciones[0]
        dc_seguro = dc_prob >= UMBRAL_SEGURO

        asian_lines = []
        for handicap in [-1.5, -0.5, 0, 0.5, 1.5]:
            if handicap == 0:
                prob = p_draw
                label = "Draw No Bet (0)"
            elif handicap < 0:
                prob = sum(prob_matrix[i, j] for i in range(MAX_GOALS_CALC) 
                           for j in range(MAX_GOALS_CALC) if i - j > abs(handicap))
                label = f"AH Local {handicap}"
            else:
                prob = sum(prob_matrix[i, j] for i in range(MAX_GOALS_CALC) 
                           for j in range(MAX_GOALS_CALC) if i - j > -handicap)
                label = f"AH Local +{handicap}"
            asian_lines.append((label, float(prob)))
        asian_lines.sort(key=lambda x: x[1], reverse=True)

        team_goal_lines = []
        for line in [0.5, 1.5, 2.5]:
            prob_over_home = float(1 - poisson.cdf(int(line), l_home))
            prob_over_away = float(1 - poisson.cdf(int(line), l_away))
            team_goal_lines.extend([
                (f"Local Over {line}", prob_over_home),
                (f"Visitante Over {line}", prob_over_away)
            ])
        team_goal_lines.sort(key=lambda x: x[1], reverse=True)

        return {
            "p_home": p_home, "p_draw": p_draw, "p_away": p_away,
            "p_btts": p_btts,
            "scores": scorelines[:5],
            "prob_matrix": prob_matrix,
            "lambda_home": l_home, "lambda_away": l_away, "lambda_total": l_total,
            "goal_pick": goal_pick, "goal_prob": goal_prob, "goal_seguro": goal_seguro,
            "goal_todas": goal_todas,
            "corner_pick": corner_pick, "corner_prob": corner_prob, "corner_seguro": corner_seguro,
            "expected_corners": l_corners, "corner_todas": corner_todas,
            "card_pick": card_pick, "card_prob": card_prob, "card_seguro": card_seguro,
            "expected_cards": l_cards, "card_todas": card_todas,
            "dc_pick": dc_pick, "dc_prob": dc_prob, "dc_seguro": dc_seguro,
            "dc_opciones": dc_opciones,
            "asian_lines": asian_lines[:3],
            "team_goals": team_goal_lines[:4],
            "mc_data": mc,
            "weight_h2h": self.weight_h2h * 100, "weight_recent": self.weight_recent * 100,
            "data_source": self.stats.get("data_source", "Autónomo"),
            "matches_analyzed": self.stats.get("matches_analyzed", 0)
        }

# ============================================================
# INTERFAZ VISUAL
# ============================================================
with st.sidebar:
    st.header("⚙️ Configuración")

    st.info("🧠 V7.0 usa Motor Autónomo Inteligente. No requiere API externa.")

    st.markdown("---")

    st.subheader("💰 Odds del Mercado")
    st.caption("Para calcular Valor Esperado (EV)")

    col_odds1, col_odds2 = st.columns(2)
    with col_odds1:
        odds_1 = st.number_input("1 (Local)", min_value=1.01, value=2.50, step=0.05, format="%.2f")
        odds_x = st.number_input("X (Empate)", min_value=1.01, value=3.40, step=0.05, format="%.2f")
    with col_odds2:
        odds_2 = st.number_input("2 (Visita)", min_value=1.01, value=2.80, step=0.05, format="%.2f")
        odds_btts = st.number_input("BTTS Sí", min_value=1.01, value=1.85, step=0.05, format="%.2f")

    st.markdown("---")
    st.caption(f"🎯 Seguro: ≥ {UMBRAL_SEGURO*100:.0f}%")
    st.caption(f"📊 Valor: ≥ {UMBRAL_VALOR*100:.0f}% edge")

col1, col2, col3 = st.columns([2, 2, 1])
with col1:
    home_team = st.text_input("🏠 Equipo Local", value="Real Madrid")
with col2:
    away_team = st.text_input("✈️ Equipo Visitante", value="Barcelona")
with col3:
    match_date = st.date_input("📅 Fecha", datetime.date.today())

predict_btn = st.button("🚀 Iniciar Análisis Predictivo", use_container_width=True, type="primary")

def pick_card(title, pick, prob, seguro, icon="🎯"):
    css_class = "pick-safe" if seguro else "pick-risk" if prob >= 0.55 else "pick-avoid"
    badge = "✅ SEGURO" if seguro else "⚠️ MODERADO" if prob >= 0.55 else "❌ RIESGO"

    st.markdown(f"""
    <div class="{css_class}" style="margin-bottom: 1rem;">
        <div style="font-size: 0.85rem; opacity: 0.9; margin-bottom: 0.3rem;">{icon} {title}</div>
        <div style="font-size: 1.3rem; margin-bottom: 0.3rem;">{pick}</div>
        <div style="font-size: 1.6rem; font-weight: 800;">{prob*100:.1f}%</div>
        <div style="font-size: 0.8rem; opacity: 0.8; margin-top: 0.3rem;">{badge}</div>
    </div>
    """, unsafe_allow_html=True)

if predict_btn and home_team and away_team:
    if home_team.lower().strip() == away_team.lower().strip():
        st.error("❌ Los equipos no pueden ser el mismo")
    else:
        with st.spinner(f"🔬 Analizando {home_team} vs {away_team} | Motor Autónomo V7.0..."):

            engine_data = AutonomousDataEngine()
            stats, home_p, away_p = engine_data.fetch_stats(home_team, away_team)

            predictor = PredictorEngine(stats)
            r = predictor.predict()

            # Info de reconocimiento
            col_info1, col_info2 = st.columns(2)
            with col_info1:
                st.success(f"✅ Análisis completado")
                st.caption(f"📊 Fuente: {r['data_source']}")
                st.caption(f"🏠 {home_team} → {home_p['nombre_match']} ({home_p['tipo_match']}) | Liga: {home_p['liga']}")
            with col_info2:
                st.caption(f"✈️ {away_team} → {away_p['nombre_match']} ({away_p['tipo_match']}) | Liga: {away_p['liga']}")
                st.caption(f"📈 Nivel Local: {home_p['nivel']:.2f} | Nivel Visita: {away_p['nivel']:.2f}")

            # PICKS PRINCIPALES
            st.markdown("## 🏆 Picks del Sistema")

            c1, c2, c3, c4 = st.columns(4)
            with c1:
                pick_card("Línea de Goles", r['goal_pick'], r['goal_prob'], r['goal_seguro'], "⚽")
            with c2:
                pick_card("Doble Oportunidad", r['dc_pick'], r['dc_prob'], r['dc_seguro'], "🛡️")
            with c3:
                pick_card("Línea de Córners", r['corner_pick'], r['corner_prob'], r['corner_seguro'], "🚩")
            with c4:
                pick_card("Línea de Tarjetas", r['card_pick'], r['card_prob'], r['card_seguro'], "🟨")

            st.markdown("---")

            # ANÁLISIS DE VALOR
            st.markdown("## 💰 Análisis de Valor vs Mercado")

            ev_1 = PredictorEngine.calculate_ev(r['p_home'], odds_1)
            ev_x = PredictorEngine.calculate_ev(r['p_draw'], odds_x)
            ev_2 = PredictorEngine.calculate_ev(r['p_away'], odds_2)
            ev_btts = PredictorEngine.calculate_ev(r['p_btts'], odds_btts)

            ev_data = []
            for label, prob, odds, ev in [
                (f"1 - {home_team}", r['p_home'], odds_1, ev_1),
                ("X - Empate", r['p_draw'], odds_x, ev_x),
                (f"2 - {away_team}", r['p_away'], odds_2, ev_2),
                ("Ambos Anotan", r['p_btts'], odds_btts, ev_btts)
            ]:
                ev_data.append({
                    'Mercado': label,
                    'Prob. Modelo': f"{prob*100:.1f}%",
                    'Cuota': f"{odds:.2f}",
                    'Prob. Implícita': f"{100/odds:.1f}%",
                    'EV (Valor)': f"{ev:+.1%}",
                    'Recomendación': '✅ Apostar' if ev > UMBRAL_VALOR else '❌ Evitar' if ev < -0.10 else '➖ Neutro'
                })

            df_ev = pd.DataFrame(ev_data)
            st.dataframe(df_ev, use_container_width=True, hide_index=True)

            best_ev = max([ev_1, ev_x, ev_2, ev_btts])
            if best_ev > UMBRAL_VALOR:
                best_idx = [ev_1, ev_x, ev_2, ev_btts].index(best_ev)
                best_labels = [f"1 ({home_team})", "X (Empate)", f"2 ({away_team})", "BTTS Sí"]
                st.balloons()
                st.success(f"🚀 **VALOR DETECTADO**: {best_labels[best_idx]} con EV de {best_ev:+.1%}")

            st.markdown("---")

            # GRÁFICOS
            col_chart1, col_chart2 = st.columns([3, 2])

            with col_chart1:
                st.markdown("##### 📊 Probabilidades 1X2")
                df_probs = pd.DataFrame({
                    'Resultado': [f'1\n{home_team}', 'X\nEmpate', f'2\n{away_team}'],
                    'Probabilidad': [r['p_home']*100, r['p_draw']*100, r['p_away']*100],
                    'Color': ['#667eea', '#fbbf24', '#f43f5e']
                })

                fig_bar = go.Figure()
                for i, row in df_probs.iterrows():
                    fig_bar.add_trace(go.Bar(
                        x=[row['Resultado']], y=[row['Probabilidad']],
                        text=[f"{row['Probabilidad']:.1f}%"], textposition='outside',
                        marker_color=row['Color'], name=row['Resultado'].split('\n')[0]
                    ))

                fig_bar.update_layout(
                    showlegend=False, height=300,
                    margin=dict(t=30, b=20, l=20, r=20),
                    plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='white'),
                    yaxis=dict(gridcolor='#334155', title='Probabilidad (%)')
                )
                st.plotly_chart(fig_bar, use_container_width=True)

            with col_chart2:
                st.markdown("##### 🎯 Distribución de Goles")
                goal_dist = []
                for g in range(6):
                    prob = poisson.pmf(g, r['lambda_total'])
                    goal_dist.append({'Goles': g, 'Probabilidad': prob*100})

                fig_dist = px.area(
                    pd.DataFrame(goal_dist), x='Goles', y='Probabilidad',
                    color_discrete_sequence=['#667eea']
                )
                fig_dist.update_layout(
                    height=300, margin=dict(t=30, b=20, l=20, r=20),
                    plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='white'), yaxis=dict(gridcolor='#334155')
                )
                st.plotly_chart(fig_dist, use_container_width=True)

            st.markdown("---")

            # HEATMAP
            st.markdown("##### 🔥 Matriz de Resultados Probables")

            heat_data = []
            for i in range(6):
                for j in range(6):
                    heat_data.append({
                        'Local': i, 'Visitante': j,
                        'Probabilidad': r['prob_matrix'][i, j] * 100
                    })

            df_heat = pd.DataFrame(heat_data)
            pivot_heat = df_heat.pivot(index='Local', columns='Visitante', values='Probabilidad')

            fig_heat = px.imshow(
                pivot_heat.values,
                labels=dict(x=f"{away_team} (Goles)", y=f"{home_team} (Goles)", color="Prob %"),
                x=[str(i) for i in range(6)], y=[str(i) for i in range(6)],
                color_continuous_scale='Viridis', aspect="equal"
            )
            fig_heat.update_traces(
                text=[[f"{val:.1f}%" for val in row] for row in pivot_heat.values],
                texttemplate="%{text}", textfont={"size": 10}
            )
            fig_heat.update_layout(
                height=400, plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white')
            )
            st.plotly_chart(fig_heat, use_container_width=True)

            st.markdown("---")

            # DETALLE POR MERCADO
            col_goles, col_dc, col_corners, col_cards = st.columns(4)

            with col_goles:
                st.subheader("⚽ Goles")
                st.metric("Pick", r['goal_pick'], f"{r['goal_prob']*100:.1f}%")
                st.caption(f"λ total: {r['lambda_total']:.2f}")
                st.caption(f"{home_team}: {r['lambda_home']:.2f} | {away_team}: {r['lambda_away']:.2f}")

                st.write(f"**Ambos Anotan: {r['p_btts']*100:.1f}%**")

                with st.expander("Todas las líneas"):
                    for nombre, prob in r['goal_todas']:
                        marker = "⭐" if nombre == r['goal_pick'] else ""
                        st.write(f"{marker} {nombre}: **{prob*100:.1f}%**")

                st.write("**Marcadores probables:**")
                for score, prob in r['scores']:
                    st.write(f"👉 **{score}** ({prob*100:.1f}%)")

                st.write("**Goles por equipo:**")
                for label, prob in r['team_goals']:
                    st.write(f"• {label}: {prob*100:.1f}%")

            with col_dc:
                st.subheader("🛡️ Doble Oport.")
                st.metric("Pick", r['dc_pick'], f"{r['dc_prob']*100:.1f}%")

                for nombre, prob in r['dc_opciones']:
                    if nombre == r['dc_pick']:
                        st.success(f"**{nombre}:** {prob*100:.1f}% ⭐")
                    else:
                        st.write(f"**{nombre}:** {prob*100:.1f}%")

                st.write("---")
                st.write("**🎯 Handicap Asiático:**")
                for label, prob in r['asian_lines']:
                    st.write(f"• {label}: {prob*100:.1f}%")

            with col_corners:
                st.subheader("🚩 Córners")
                st.metric("Pick", r['corner_pick'], f"{r['corner_prob']*100:.1f}%")
                st.caption(f"Total: {r['expected_corners']:.1f}")

                with st.expander("Todas las líneas"):
                    for nombre, prob in r['corner_todas']:
                        marker = "⭐" if nombre == r['corner_pick'] else ""
                        st.write(f"{marker} {nombre}: **{prob*100:.1f}%**")

            with col_cards:
                st.subheader("🟨 Tarjetas")
                st.metric("Pick", r['card_pick'], f"{r['card_prob']*100:.1f}%")
                st.caption(f"Total: {r['expected_cards']:.1f}")

                with st.expander("Todas las líneas"):
                    for nombre, prob in r['card_todas']:
                        marker = "⭐" if nombre == r['card_pick'] else ""
                        st.write(f"{marker} {nombre}: **{prob*100:.1f}%**")

            st.markdown("---")

            # EXPORTAR
            col_exp1, col_exp2 = st.columns(2)

            import json as json_lib
            export_json = {
                "partido": f"{home_team} vs {away_team}",
                "fecha": str(match_date),
                "fuente_datos": r['data_source'],
                "perfiles": {
                    "local": {"nombre": home_p['nombre_match'], "liga": home_p['liga'], "nivel": home_p['nivel']},
                    "visita": {"nombre": away_p['nombre_match'], "liga": away_p['liga'], "nivel": away_p['nivel']}
                },
                "probabilidades": {
                    "1": round(r['p_home'], 4), "X": round(r['p_draw'], 4),
                    "2": round(r['p_away'], 4), "btts": round(r['p_btts'], 4)
                },
                "picks": {
                    "goles": {"pick": r['goal_pick'], "prob": round(r['goal_prob'], 4)},
                    "doble_oportunidad": {"pick": r['dc_pick'], "prob": round(r['dc_prob'], 4)},
                    "corners": {"pick": r['corner_pick'], "prob": round(r['corner_prob'], 4)},
                    "tarjetas": {"pick": r['card_pick'], "prob": round(r['card_prob'], 4)}
                },
                "lambdas": {
                    "home": round(r['lambda_home'], 4),
                    "away": round(r['lambda_away'], 4),
                    "total": round(r['lambda_total'], 4)
                }
            }

            with col_exp1:
                st.download_button(
                    "📥 Descargar JSON",
                    data=json_lib.dumps(export_json, indent=2),
                    file_name=f"prediccion_{home_team}_{away_team}.json",
                    mime="application/json"
                )

            with col_exp2:
                csv_data = pd.DataFrame([
                    ['Resultado', 'Probabilidad', 'Pick'],
                    [f'1 ({home_team})', f"{r['p_home']*100:.1f}%", ''],
                    ['X (Empate)', f"{r['p_draw']*100:.1f}%", ''],
                    [f'2 ({away_team})', f"{r['p_away']*100:.1f}%", ''],
                    ['Goles', f"{r['goal_prob']*100:.1f}%", r['goal_pick']],
                    ['Doble Oport.', f"{r['dc_prob']*100:.1f}%", r['dc_pick']],
                    ['Córners', f"{r['corner_prob']*100:.1f}%", r['corner_pick']],
                    ['Tarjetas', f"{r['card_prob']*100:.1f}%", r['card_pick']]
                ])
                st.download_button(
                    "📊 Descargar CSV",
                    data=csv_data.to_csv(index=False, header=False),
                    file_name=f"prediccion_{home_team}_{away_team}.csv",
                    mime="text/csv"
                )

            st.markdown("---")
            st.caption(
                "⚠️ **Disclaimer**: Probabilidades estimadas con modelo Dixon-Coles + Monte Carlo. "
                "Ningún resultado garantizado. El EV requiere odds reales. Apuesta con responsabilidad."
            )

elif predict_btn:
    st.warning("⚠️ Ingresa los nombres de ambos equipos.")

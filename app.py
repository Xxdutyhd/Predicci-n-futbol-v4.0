
import streamlit as st
import requests
import datetime
import hashlib
import numpy as np
from scipy.stats import poisson, skellam
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ============================================================
# CONFIGURACIÓN DE LA PÁGINA (UI PREMIUM)
# ============================================================
st.set_page_config(
    page_title="AI Match Predictor Pro V6.0",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS Premium
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
    .metric-card {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border-radius: 16px;
        padding: 1.5rem;
        border: 1px solid #334155;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
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
    .confidence-bar {
        height: 8px;
        border-radius: 4px;
        background: #1e293b;
        overflow: hidden;
    }
    .confidence-fill {
        height: 100%;
        border-radius: 4px;
        transition: width 0.5s ease;
    }
    .value-positive { color: #34d399; font-weight: 700; }
    .value-negative { color: #f87171; font-weight: 700; }
    .value-neutral { color: #94a3b8; font-weight: 700; }
    div[data-testid="stExpander"] {
        background: #0f172a;
        border-radius: 12px;
        border: 1px solid #1e293b;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">⚽ AI Match Predictor Pro V6.0</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Motor Dixon-Coles + Monte Carlo | Análisis de Valor | Selección Inteligente de Líneas</div>', unsafe_allow_html=True)

# ============================================================
# CONSTANTES Y CONFIGURACIÓN
# ============================================================
UMBRAL_SEGURO = 0.70
UMBRAL_VALOR = 0.05  # 5% de edge mínimo para recomendar valor
SIMULACIONES = 10000
MAX_GOALS_CALC = 12  # Aumentado para mayor precisión

# ============================================================
# 1. MÓDULO DE EXTRACCIÓN DE DATOS (API / DINÁMICO)
# ============================================================
class APIFootballFetcher:
    def __init__(self, api_key=""):
        self.api_key = api_key
        self.headers = {
            "x-rapidapi-key": self.api_key,
            "x-rapidapi-host": "api-football-v1.p.rapidapi.com"
        }
        self.base_url = "https://api-football-v1.p.rapidapi.com/v3"

    @st.cache_data(ttl=3600, show_spinner=False)
    def get_team_id(_self, team_name):
        if not _self.api_key:
            return None
        url = f"{_self.base_url}/teams"
        try:
            res = requests.get(url, headers=_self.headers, params={"search": team_name}, timeout=10).json()
            return res['response'][0]['team']['id'] if res.get('response') else None
        except Exception:
            return None

    def _generate_dynamic_stats(self, home_team, away_team):
        """
        Genera estadísticas simuladas pero con variabilidad controlada.
        A diferencia del hash fijo anterior, esto crea perfiles realistas
        basados en 'perfiles de equipo' simulados.
        """
        seed = int(hashlib.md5(f"{home_team}_{away_team}".encode()).hexdigest(), 16)
        np.random.seed(seed % 2**32)

        # Perfil de ataque/defensa basado en "nivel del equipo" simulado
        def team_profile(name):
            name_hash = int(hashlib.md5(name.lower().encode()).hexdigest(), 16)
            np.random.seed(name_hash % 2**32)

            # Nivel del equipo (0-1, donde 1 es elite)
            level = np.random.beta(2, 2)  # Distribución más realista

            # Ataque: goles esperados por partido
            attack = 0.8 + level * 1.8 + np.random.normal(0, 0.15)
            # Defensa: goles concedidos esperados
            defense = 1.6 - level * 0.9 + np.random.normal(0, 0.15)

            # Córners
            corner_attack = 3.0 + level * 3.0 + np.random.normal(0, 0.3)
            corner_defense = 4.0 - level * 1.5 + np.random.normal(0, 0.3)

            # Tarjetas (equipos defensivos = más tarjetas)
            card_tendency = 1.8 + (1 - level) * 1.2 + np.random.normal(0, 0.2)

            return {
                'attack': max(0.3, attack),
                'defense': max(0.3, defense),
                'corner_attack': max(1.0, corner_attack),
                'corner_defense': max(1.0, corner_defense),
                'card_tendency': max(0.8, card_tendency),
                'level': level
            }

        home = team_profile(home_team)
        away = team_profile(away_team)

        # Factor de localía realista
        home_advantage = 1.35

        # Goles esperados
        h2h_home = (home['attack'] + away['defense']) / 2 * home_advantage
        h2h_away = (away['attack'] + home['defense']) / 2

        # Forma reciente simulada (con peso decreciente)
        recent_home = h2h_home * (1 + np.random.normal(0, 0.1))
        recent_away = h2h_away * (1 + np.random.normal(0, 0.1))

        # Córners esperados
        exp_corners_home = (home['corner_attack'] + away['corner_defense']) / 2 * 1.1
        exp_corners_away = (away['corner_attack'] + home['corner_defense']) / 2

        # Tarjetas esperadas (más tarjetas en partidos equilibrados y de bajo nivel)
        intensity = 1.0 + abs(home['level'] - away['level']) * 0.3
        exp_cards_home = home['card_tendency'] * intensity
        exp_cards_away = away['card_tendency'] * intensity

        return {
            "h2h_home_goals_avg": round(max(0.2, h2h_home), 2),
            "h2h_away_goals_avg": round(max(0.2, h2h_away), 2),
            "recent_home_goals_avg": round(max(0.2, recent_home), 2),
            "recent_away_goals_avg": round(max(0.2, recent_away), 2),
            "expected_corners_home": round(max(1.0, exp_corners_home), 1),
            "expected_corners_away": round(max(1.0, exp_corners_away), 1),
            "expected_cards_home": round(max(0.5, exp_cards_home), 1),
            "expected_cards_away": round(max(0.5, exp_cards_away), 1),
            "home_level": round(home['level'], 2),
            "away_level": round(away['level'], 2)
        }

    @st.cache_data(ttl=1800, show_spinner=False)
    def get_fixture_statistics(_self, fixture_id, defaults):
        try:
            url = f"{_self.base_url}/fixtures/statistics"
            res = requests.get(url, headers=_self.headers, params={"fixture": fixture_id}, timeout=10).json()
            if res.get('response') and len(res['response']) == 2:
                stats_home = {s['type']: s['value'] for s in res['response'][0]['statistics']}
                stats_away = {s['type']: s['value'] for s in res['response'][1]['statistics']}

                h_corn = float(stats_home.get('Corner Kicks', defaults[0]) or defaults[0])
                a_corn = float(stats_away.get('Corner Kicks', defaults[1]) or defaults[1])
                h_cards = float(stats_home.get('Yellow Cards', 2.0) or 2.0) + float(stats_home.get('Red Cards', 0) or 0)
                a_cards = float(stats_away.get('Yellow Cards', 2.0) or 2.0) + float(stats_away.get('Red Cards', 0) or 0)

                return h_corn, a_corn, h_cards, a_cards
        except Exception:
            pass
        return None

    @st.cache_data(ttl=1800, show_spinner=False)
    def fetch_h2h_and_stats(_self, home_team, away_team, home_id=None, away_id=None):
        if _self.api_key and home_id and away_id:
            try:
                url = f"{_self.base_url}/fixtures/headtohead"
                res = requests.get(url, headers=_self.headers,
                                   params={"h2h": f"{home_id}-{away_id}", "last": "10"}, timeout=10).json()
                if res.get('response'):
                    fixtures = res['response']

                    # Ponderación exponencial: partidos más recientes valen más
                    h_goals, a_goals, total_weight = 0, 0, 0
                    for idx, f in enumerate(fixtures):
                        weight = 0.85 ** idx  # Decaimiento exponencial
                        h_goals += (f['goals']['home'] or 0) * weight
                        a_goals += (f['goals']['away'] or 0) * weight
                        total_weight += weight

                    count = len(fixtures)

                    # Estadísticas del último enfrentamiento
                    last_fixture_id = fixtures[0]['fixture']['id']
                    real_stats = _self.get_fixture_statistics(last_fixture_id, (4.5, 4.5))

                    if real_stats:
                        h_corn, a_corn, h_cards, a_cards = real_stats
                    else:
                        h_corn, a_corn = 5.2, 4.3
                        h_cards, a_cards = 2.2, 2.4

                    return {
                        "h2h_home_goals_avg": round(h_goals / total_weight, 2),
                        "h2h_away_goals_avg": round(a_goals / total_weight, 2),
                        "recent_home_goals_avg": round(h_goals / total_weight, 2),
                        "recent_away_goals_avg": round(a_goals / total_weight, 2),
                        "expected_corners_home": round(h_corn, 1),
                        "expected_corners_away": round(a_corn, 1),
                        "expected_cards_home": round(h_cards, 1),
                        "expected_cards_away": round(a_cards, 1),
                        "data_source": "API Real (H2H ponderado)",
                        "matches_analyzed": count
                    }
            except Exception as e:
                st.warning(f"Error API: {str(e)}. Usando motor dinámico.")

        stats = _self._generate_dynamic_stats(home_team, away_team)
        stats["data_source"] = "Motor Predictivo (Simulado)"
        stats["matches_analyzed"] = 0
        return stats

# ============================================================
# 2. MOTOR DE ANÁLISIS AVANZADO
# ============================================================
class PredictorEngine:
    """
    Motor híbrido: Dixon-Coles para goles + Monte Carlo para mercados complejos
    """
    def __init__(self, stats):
        self.stats = stats
        self.weight_h2h = 0.65
        self.weight_recent = 0.35
        self.rho = -0.05  # Factor de correlación Dixon-Coles (ajuste para empates de bajo scoring)

    def calculate_lambdas(self):
        """Calcula lambdas con ponderación y factor de localía dinámico."""
        lambda_home = (self.stats["h2h_home_goals_avg"] * self.weight_h2h) +                       (self.stats["recent_home_goals_avg"] * self.weight_recent)
        lambda_away = (self.stats["h2h_away_goals_avg"] * self.weight_h2h) +                       (self.stats["recent_away_goals_avg"] * self.weight_recent)

        # Factor de localía basado en diferencia de nivel
        level_diff = self.stats.get("home_level", 0.5) - self.stats.get("away_level", 0.5)
        home_adv = 1.05 + (level_diff * 0.15)  # 1.05 a 1.20 según nivel

        lambda_home *= home_adv
        return max(0.2, lambda_home), max(0.2, lambda_away)

    def dixon_coles_adjustment(self, i, j, lambda_h, lambda_a):
        """
        Ajuste de Dixon-Coles para empates de bajo scoring.
        Corrige la subestimación de empates 0-0 y 1-1 en Poisson básico.
        """
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
        """
        Simulación Monte Carlo para mercados complejos donde Poisson
        analítico es difícil de calcular directamente.
        """
        home_goals = np.random.poisson(lambda_h, n_sim)
        away_goals = np.random.poisson(lambda_a, n_sim)

        total_goals = home_goals + away_goals
        goal_diff = home_goals - away_goals
        btts = (home_goals > 0) & (away_goals > 0)

        return {
            'home_goals': home_goals,
            'away_goals': away_goals,
            'total_goals': total_goals,
            'goal_diff': goal_diff,
            'btts': btts
        }

    @staticmethod
    def calculate_ev(prob, odds):
        """Calcula Expected Value: (prob * odds) - 1"""
        if odds <= 1:
            return -1.0
        return (prob * odds) - 1

    @staticmethod
    def pick_safest_line(lambda_val, lineas_over, lineas_under, min_prob=UMBRAL_SEGURO, 
                         tipo="goles", sim_data=None):
        """
        Selección inteligente de línea con análisis de valor.
        """
        candidatos = []

        if sim_data is not None and tipo == "goles":
            # Usar simulación Monte Carlo para mayor precisión
            for linea in lineas_over:
                n = int(linea)
                prob = np.mean(sim_data['total_goals'] > n)
                candidatos.append((f"Over {linea}", prob))
            for linea in lineas_under:
                n = int(linea)
                prob = np.mean(sim_data['total_goals'] <= n)
                candidatos.append((f"Under {linea}", prob))
        else:
            # Poisson analítico
            for linea in lineas_over:
                n = int(linea)
                prob = 1 - poisson.cdf(n, lambda_val)
                candidatos.append((f"Over {linea}", prob))
            for linea in lineas_under:
                n = int(linea)
                prob = poisson.cdf(n, lambda_val)
                candidatos.append((f"Under {linea}", prob))

        candidatos.sort(key=lambda x: x[1], reverse=True)
        mejor_pick, mejor_prob = candidatos[0]
        seguro = mejor_prob >= min_prob
        return mejor_pick, mejor_prob, seguro, candidatos

    def predict(self):
        l_home, l_away = self.calculate_lambdas()
        l_total = l_home + l_away

        # Simulación Monte Carlo
        mc = self.monte_carlo_simulation(l_home, l_away)

        # Probabilidades exactas con Dixon-Coles
        p_home, p_draw, p_away, p_btts = 0.0, 0.0, 0.0, 0.0
        scorelines = []
        prob_matrix = np.zeros((MAX_GOALS_CALC, MAX_GOALS_CALC))

        # Constante de normalización para Dixon-Coles
        tau_sum = 0
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

        # Verificar coherencia con Monte Carlo
        mc_home = np.mean(mc['goal_diff'] > 0)
        mc_draw = np.mean(mc['goal_diff'] == 0)
        mc_away = np.mean(mc['goal_diff'] < 0)
        mc_btts = np.mean(mc['btts'])

        # Promediar ambos métodos para robustez
        p_home = (p_home + mc_home) / 2
        p_draw = (p_draw + mc_draw) / 2
        p_away = (p_away + mc_away) / 2
        p_btts = (p_btts + mc_btts) / 2

        # ---------- GOLES: línea segura con Monte Carlo ----------
        goal_pick, goal_prob, goal_seguro, goal_todas = self.pick_safest_line(
            l_total,
            lineas_over=[0.5, 1.5, 2.5, 3.5],
            lineas_under=[2.5, 3.5, 4.5, 5.5],
            sim_data=mc
        )

        # ---------- CÓRNERS: modelo independiente ----------
        l_corners = self.stats["expected_corners_home"] + self.stats["expected_corners_away"]
        corner_pick, corner_prob, corner_seguro, corner_todas = self.pick_safest_line(
            l_corners,
            lineas_over=[6.5, 7.5, 8.5, 9.5],
            lineas_under=[10.5, 11.5, 12.5, 13.5]
        )

        # ---------- TARJETAS: modelo independiente ----------
        l_cards = self.stats["expected_cards_home"] + self.stats["expected_cards_away"]
        card_pick, card_prob, card_seguro, card_todas = self.pick_safest_line(
            l_cards,
            lineas_over=[1.5, 2.5, 3.5, 4.5],
            lineas_under=[4.5, 5.5, 6.5, 7.5]
        )

        # ---------- DOBLE OPORTUNIDAD ----------
        dc_opciones = [
            ("1X (Local o Empate)", p_home + p_draw),
            ("X2 (Visita o Empate)", p_away + p_draw),
            ("12 (Sin Empate)", p_home + p_away),
        ]
        dc_opciones.sort(key=lambda x: x[1], reverse=True)
        dc_pick, dc_prob = dc_opciones[0]
        dc_seguro = dc_prob >= UMBRAL_SEGURO

        # ---------- HANDICAP ASIÁTICO ----------
        asian_lines = []
        for handicap in [-1.5, -0.5, 0, 0.5, 1.5]:
            if handicap == 0:
                prob = p_draw
                label = "Draw No Bet (0)"
            elif handicap < 0:
                # Local debe ganar por más de |handicap|
                prob = np.sum(prob_matrix[i, j] for i in range(MAX_GOALS_CALC) 
                             for j in range(MAX_GOALS_CALC) if i - j > abs(handicap))
                label = f"AH Local {handicap}"
            else:
                # Local puede perder por menos de handicap
                prob = np.sum(prob_matrix[i, j] for i in range(MAX_GOALS_CALC) 
                             for j in range(MAX_GOALS_CALC) if i - j > -handicap)
                label = f"AH Local +{handicap}"
            asian_lines.append((label, prob))
        asian_lines.sort(key=lambda x: x[1], reverse=True)

        # ---------- MÁS/MENOS GOLES POR EQUIPO ----------
        team_goal_lines = []
        for line in [0.5, 1.5, 2.5]:
            prob_over_home = 1 - poisson.cdf(int(line), l_home)
            prob_over_away = 1 - poisson.cdf(int(line), l_away)
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
            "data_source": self.stats.get("data_source", "Desconocido"),
            "matches_analyzed": self.stats.get("matches_analyzed", 0)
        }

# ============================================================
# 3. INTERFAZ VISUAL PREMIUM
# ============================================================
with st.sidebar:
    st.header("⚙️ Configuración")

    api_key_input = st.text_input(
        "API-Football Key (Opcional)", 
        type="password",
        help="RapidAPI Key para datos reales. Sin ella se usa el motor predictivo."
    )

    if not api_key_input:
        st.info("💡 Modo: Motor Predictivo Inteligente")
    else:
        st.success("🔑 API Key activa")

    st.markdown("---")

    # Input de odds para análisis de valor
    st.subheader("💰 Análisis de Valor")
    st.caption("Ingresa odds del mercado para calcular EV")

    col_odds1, col_odds2 = st.columns(2)
    with col_odds1:
        odds_1 = st.number_input("1 (Local)", min_value=1.01, value=2.50, step=0.05, format="%.2f")
        odds_x = st.number_input("X (Empate)", min_value=1.01, value=3.40, step=0.05, format="%.2f")
    with col_odds2:
        odds_2 = st.number_input("2 (Visita)", min_value=1.01, value=2.80, step=0.05, format="%.2f")
        odds_btts = st.number_input("BTTS Sí", min_value=1.01, value=1.85, step=0.05, format="%.2f")

    st.markdown("---")
    st.caption(f"🎯 Umbral seguro: ≥ {UMBRAL_SEGURO*100:.0f}%")
    st.caption(f"📊 Umbral valor: ≥ {UMBRAL_VALOR*100:.0f}% edge")

# Layout principal
col1, col2, col3 = st.columns([2, 2, 1])
with col1:
    home_team = st.text_input("🏠 Equipo Local", value="Real Madrid")
with col2:
    away_team = st.text_input("✈️ Equipo Visitante", value="Barcelona")
with col3:
    match_date = st.date_input("📅 Fecha", datetime.date.today())

predict_btn = st.button("🚀 Iniciar Análisis Predictivo Avanzado", use_container_width=True, type="primary")

# ============================================================
# 4. RENDERIZADO DE RESULTADOS
# ============================================================
def render_confidence_bar(prob, height="8px"):
    color = "#34d399" if prob >= 0.70 else "#fbbf24" if prob >= 0.55 else "#f87171"
    return f"""
    <div style="height: {height}; border-radius: 4px; background: #1e293b; overflow: hidden;">
        <div style="height: 100%; width: {prob*100:.1f}%; background: {color}; border-radius: 4px; transition: width 0.5s;"></div>
    </div>
    """

def value_badge(ev):
    if ev > 0.10:
        return "<span class='value-positive'>🔥 VALOR ALTO</span>"
    elif ev > 0.05:
        return "<span class='value-positive'>✅ VALOR</span>"
    elif ev > -0.05:
        return "<span class='value-neutral'>➖ Neutro</span>"
    else:
        return "<span class='value-negative'>❌ Sin valor</span>"

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
        with st.spinner(f"🔬 Analizando {home_team} vs {away_team} | Motor Dixon-Coles + Monte Carlo..."):

            fetcher = APIFootballFetcher(api_key_input)
            h_id = fetcher.get_team_id(home_team)
            a_id = fetcher.get_team_id(away_team)
            stats = fetcher.fetch_h2h_and_stats(home_team, away_team, h_id, a_id)

            engine = PredictorEngine(stats)
            r = engine.predict()

            # Banner de éxito
            source_color = "#34d399" if "API" in r["data_source"] else "#fbbf24"
            st.success(f"✅ Análisis completado | Fuente: {r['data_source']}")

            # ============================================================
            # PICKS PRINCIPALES
            # ============================================================
            st.markdown("## 🏆 Picks Automáticos del Sistema")

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

            # ============================================================
            # ANÁLISIS DE VALOR (VALUE BETTING)
            # ============================================================
            st.markdown("## 💰 Análisis de Valor vs Mercado")

            ev_1 = engine.calculate_ev(r['p_home'], odds_1)
            ev_x = engine.calculate_ev(r['p_draw'], odds_x)
            ev_2 = engine.calculate_ev(r['p_away'], odds_2)
            ev_btts = engine.calculate_ev(r['p_btts'], odds_btts)

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

            # Alerta de valor
            best_ev = max([ev_1, ev_x, ev_2, ev_btts])
            if best_ev > UMBRAL_VALOR:
                best_idx = [ev_1, ev_x, ev_2, ev_btts].index(best_ev)
                best_labels = [f"1 ({home_team})", "X (Empate)", f"2 ({away_team})", "BTTS Sí"]
                st.balloons()
                st.success(f"🚀 **OPORTUNIDAD DETECTADA**: {best_labels[best_idx]} tiene un valor esperado de {best_ev:+.1%}. ¡Edge significativo sobre el mercado!")

            st.markdown("---")

            # ============================================================
            # GRÁFICO 1X2 + DISTRIBUCIÓN
            # ============================================================
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
                        x=[row['Resultado']],
                        y=[row['Probabilidad']],
                        text=[f"{row['Probabilidad']:.1f}%"],
                        textposition='outside',
                        marker_color=row['Color'],
                        name=row['Resultado'].split('\n')[0]
                    ))

                fig_bar.update_layout(
                    showlegend=False,
                    height=300,
                    margin=dict(t=30, b=20, l=20, r=20),
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
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
                    pd.DataFrame(goal_dist), 
                    x='Goles', y='Probabilidad',
                    color_discrete_sequence=['#667eea']
                )
                fig_dist.update_layout(
                    height=300,
                    margin=dict(t=30, b=20, l=20, r=20),
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='white'),
                    yaxis=dict(gridcolor='#334155')
                )
                st.plotly_chart(fig_dist, use_container_width=True)

            st.markdown("---")

            # ============================================================
            # HEATMAP DE RESULTADOS
            # ============================================================
            st.markdown("##### 🔥 Matriz de Resultados Probables (Dixon-Coles)")

            heat_data = []
            for i in range(6):
                for j in range(6):
                    heat_data.append({
                        'Local': i,
                        'Visitante': j,
                        'Probabilidad': r['prob_matrix'][i, j] * 100
                    })

            df_heat = pd.DataFrame(heat_data)
            pivot_heat = df_heat.pivot(index='Local', columns='Visitante', values='Probabilidad')

            fig_heat = px.imshow(
                pivot_heat.values,
                labels=dict(x=f"{away_team} (Goles)", y=f"{home_team} (Goles)", color="Prob %"),
                x=[str(i) for i in range(6)],
                y=[str(i) for i in range(6)],
                color_continuous_scale='Viridis',
                aspect="equal"
            )
            fig_heat.update_traces(
                text=[[f"{val:.1f}%" for val in row] for row in pivot_heat.values],
                texttemplate="%{text}",
                textfont={"size": 10}
            )
            fig_heat.update_layout(
                height=400,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white')
            )
            st.plotly_chart(fig_heat, use_container_width=True)

            st.markdown("---")

            # ============================================================
            # DETALLE POR MERCADO
            # ============================================================
            col_goles, col_dc, col_corners, col_cards = st.columns(4)

            with col_goles:
                st.subheader("⚽ Goles")
                st.metric("Pick Principal", r['goal_pick'], f"{r['goal_prob']*100:.1f}%")
                st.caption(f"λ total: {r['lambda_total']:.2f}")
                st.caption(f"{home_team}: {r['lambda_home']:.2f} | {away_team}: {r['lambda_away']:.2f}")

                st.write("**Ambos Anotan:**")
                st.progress(r['p_btts'], text=f"{r['p_btts']*100:.1f}%")

                with st.expander("Todas las líneas"):
                    for nombre, prob in r['goal_todas']:
                        marker = "⭐" if nombre == r['goal_pick'] else ""
                        st.write(f"{marker} {nombre}: **{prob*100:.1f}%**")

                st.write("**Marcadores más probables:**")
                for score, prob in r['scores']:
                    st.write(f"👉 **{score}** ({prob*100:.1f}%)")

                st.write("**Goles por equipo:**")
                for label, prob in r['team_goals']:
                    st.write(f"• {label}: {prob*100:.1f}%")

            with col_dc:
                st.subheader("🛡️ Doble Oport.")
                st.metric("Pick Principal", r['dc_pick'], f"{r['dc_prob']*100:.1f}%")

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
                st.metric("Pick Principal", r['corner_pick'], f"{r['corner_prob']*100:.1f}%")
                st.caption(f"Total esperados: {r['expected_corners']:.1f}")

                with st.expander("Todas las líneas"):
                    for nombre, prob in r['corner_todas']:
                        marker = "⭐" if nombre == r['corner_pick'] else ""
                        st.write(f"{marker} {nombre}: **{prob*100:.1f}%**")

            with col_cards:
                st.subheader("🟨 Tarjetas")
                st.metric("Pick Principal", r['card_pick'], f"{r['card_prob']*100:.1f}%")
                st.caption(f"Total esperadas: {r['expected_cards']:.1f}")

                with st.expander("Todas las líneas"):
                    for nombre, prob in r['card_todas']:
                        marker = "⭐" if nombre == r['card_pick'] else ""
                        st.write(f"{marker} {nombre}: **{prob*100:.1f}%**")

            st.markdown("---")

            # ============================================================
            # EXPORTAR Y METADATOS
            # ============================================================
            col_exp1, col_exp2 = st.columns(2)

            export_data = {
                "partido": f"{home_team} vs {away_team}",
                "fecha": str(match_date),
                "fuente_datos": r['data_source'],
                "partidos_analizados": r['matches_analyzed'],
                "probabilidades": {
                    "1": round(r['p_home'], 4),
                    "X": round(r['p_draw'], 4),
                    "2": round(r['p_away'], 4),
                    "btts": round(r['p_btts'], 4)
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
                    data=pd.json_normalize(export_data).to_json(indent=2),
                    file_name=f"prediccion_{home_team}_{away_team}.json",
                    mime="application/json"
                )

            with col_exp2:
                csv_data = pd.DataFrame([
                    ['Resultado', 'Probabilidad', 'Pick'],
                    [f'1 ({home_team})', f"{r["p_home"]*100:.1f}%", ''],
                    ['X (Empate)', f"{r["p_draw"]*100:.1f}%", ''],
                    [f'2 ({away_team})', f"{r["p_away"]*100:.1f}%", ''],
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

            # Footer
            st.markdown("---")
            st.caption(
                "⚠️ **Disclaimer**: Las probabilidades son estimaciones estadísticas usando modelo Dixon-Coles "
                "y simulación Monte Carlo. Ningún resultado está garantizado. "
                "El análisis de valor (EV) requiere odds reales del mercado para ser preciso. "
                "Apuesta con responsabilidad."
            )

elif predict_btn:
    st.warning("⚠️ Ingresa los nombres de ambos equipos.")

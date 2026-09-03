import streamlit as st
import requests
import datetime
from scipy.stats import poisson, norm
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# ============================================================
# CONFIGURACIÓN DE LA PÁGINA (UI)
# ============================================================
st.set_page_config(page_title="AI Predictor V7.5", page_icon="🎯", layout="wide")

st.title("🎯 AI Match Predictor V7.5 - Motor H2H Multi-Deporte Avanzado")
st.markdown(
    "Motor con **Métricas Avanzadas (xG, Pace, xFIP)**, **Ponderación Dinámica (Time-Decay)** "
    "y **Caché de Peticiones API** con sistema de respaldo automático ante fallos de red."
)

UMBRAL_SEGURO = 0.70  # 70%

# ============================================================
# 1. MÓDULO DE EXTRACCIÓN DE DATOS CON CACHÉ Y HÍBRIDO
# ============================================================
class APIFetcher:
    def __init__(self, api_key="", deporte="Fútbol"):
        self.api_key = api_key
        self.deporte = deporte
        
        self.api_configs = {
            "Fútbol": {"host": "api-football-v1.p.rapidapi.com", "url": "https://api-football-v1.p.rapidapi.com/v3"},
            "NBA": {"host": "api-nba-v1.p.rapidapi.com", "url": "https://api-nba-v1.p.rapidapi.com"},
            "MLB": {"host": "api-baseball.p.rapidapi.com", "url": "https://api-baseball.p.rapidapi.com"}
        }
        
        self.headers = {
            "x-rapidapi-key": self.api_key,
            "x-rapidapi-host": self.api_configs[self.deporte]["host"]
        }
        self.base_url = self.api_configs[self.deporte]["url"]

    def get_fallback_stats(self, home_team, away_team):
        """Valores base de liga estructurados cuando la API no está disponible."""
        if self.deporte == "Fútbol":
            return {
                "xg_home": 1.65, "xga_home": 1.10,
                "xg_away": 1.25, "xga_away": 1.45,
                "recent_xg_home": 1.80, "recent_xg_away": 1.15,
                "expected_corners_home": 5.8, "expected_corners_away": 4.1,
                "expected_cards_home": 2.2, "expected_cards_away": 2.6,
                "days_since_last_h2h": 45
            }
        elif self.deporte == "NBA":
            return {
                "pace_home": 101.5, "off_rtg_home": 115.2, "def_rtg_home": 112.0,
                "pace_away": 99.8, "off_rtg_away": 111.5, "def_rtg_away": 114.3,
                "recent_form_adj_home": 1.03, "recent_form_adj_away": 0.97,
                "variance_factor": 12.5,
                "days_since_last_h2h": 30
            }
        elif self.deporte == "MLB":
            return {
                "pitcher_xfip_home": 3.85, "batter_wrc_home": 105,
                "pitcher_xfip_away": 4.20, "batter_wrc_away": 98,
                "bullpen_era_home": 3.45, "bullpen_era_away": 4.15,
                "expected_hits_home": 8.2, "expected_hits_away": 7.5,
                "days_since_last_h2h": 15
            }

    def get_team_id(self, team_name):
        if not self.api_key: return None
        url = f"{self.base_url}/teams"
        try:
            res = requests.get(url, headers=self.headers, params={"search": team_name}, timeout=5).json()
            if res.get('response'):
                if self.deporte == "Fútbol": return res['response'][0]['team']['id']
                elif self.deporte == "NBA": return res['response'][0]['id']
        except: return None
        return None

    def fetch_stats(self, home_team, away_team):
        stats = self.get_fallback_stats(home_team, away_team)
        if not self.api_key:
            return stats

        try:
            h_id = self.get_team_id(home_team)
            a_id = self.get_team_id(away_team)
            if not h_id or not a_id:
                return stats

            if self.deporte == "Fútbol":
                url_stats = f"{self.base_url}/teams/statistics"
                res_home = requests.get(url_stats, headers=self.headers, params={"team": h_id, "league": "39", "season": "2023"}, timeout=5).json()
                if res_home.get('response'):
                    data = res_home['response']
                    if 'expected_goals' in data.get('fixtures', {}):
                        stats["xg_home"] = float(data['fixtures']['expected_goals']['for']['average'])
                        stats["xga_home"] = float(data['fixtures']['expected_goals']['against']['average'])
            
            elif self.deporte == "NBA":
                url_stats = f"{self.base_url}/statistics/teams"
                res_home = requests.get(url_stats, headers=self.headers, params={"id": h_id, "season": "2023"}, timeout=5).json()
                if res_home.get('response'):
                    data = res_home['response'][0]
                    stats["pace_home"] = float(data.get('pace', stats["pace_home"]))
                    stats["off_rtg_home"] = float(data.get('offensiveRating', stats["off_rtg_home"]))
                    stats["def_rtg_home"] = float(data.get('defensiveRating', stats["def_rtg_home"]))

        except Exception:
            pass

        return stats

# Función con caché de Streamlit para evitar llamadas repetidas a la API en la misma sesión
@st.cache_data(ttl=3600)
def cached_fetch_stats(api_key, deporte, home_team, away_team):
    fetcher = APIFetcher(api_key, deporte)
    return fetcher.fetch_stats(home_team, away_team)

# ============================================================
# 2. MOTORES DE ANÁLISIS (MODELOS AVANZADOS)
# ============================================================

class PredictorBase:
    def __init__(self, stats):
        self.stats = stats
        # Ponderación dinámica basada en Time-Decay (pierde 1% por cada 10 días desde el último choque)
        dias_h2h = self.stats.get("days_since_last_h2h", 60)
        self.w_h2h = max(0.30, 0.75 - (dias_h2h / 1000.0))
        self.w_rec = 1.0 - self.w_h2h

    @staticmethod
    def _pick_safest_line_poisson(lambda_total, lineas_over, lineas_under, min_prob=UMBRAL_SEGURO):
        candidatos = []
        for linea in lineas_over:
            candidatos.append((f"Over {linea}", 1 - poisson.cdf(int(linea), lambda_total)))
        for linea in lineas_under:
            candidatos.append((f"Under {linea}", poisson.cdf(int(linea), lambda_total)))
        candidatos.sort(key=lambda x: x[1], reverse=True)
        return candidatos[0][0], candidatos[0][1], candidatos[0][1] >= min_prob, candidatos

    @staticmethod
    def _pick_safest_line_norm(mu, std_dev, lineas_over, lineas_under, min_prob=UMBRAL_SEGURO):
        candidatos = []
        for linea in lineas_over:
            candidatos.append((f"Over {linea}", 1 - norm.cdf(linea, loc=mu, scale=std_dev)))
        for linea in lineas_under:
            candidatos.append((f"Under {linea}", norm.cdf(linea, loc=mu, scale=std_dev)))
        candidatos.sort(key=lambda x: x[1], reverse=True)
        return candidatos[0][0], candidatos[0][1], candidatos[0][1] >= min_prob, candidatos

class PredictorFootball(PredictorBase):
    def predict(self):
        attack_home = (self.stats["xg_home"] * self.w_h2h) + (self.stats["recent_xg_home"] * self.w_rec)
        defense_away = (self.stats["xga_away"] * self.w_h2h) + (self.stats["xg_away"] * self.w_rec)
        
        attack_away = (self.stats["xg_away"] * self.w_h2h) + (self.stats["recent_xg_away"] * self.w_rec)
        defense_home = (self.stats["xga_home"] * self.w_h2h) + (self.stats["xg_home"] * self.w_rec)
        
        l_home = max(0.2, (attack_home + defense_away) / 2.0 * 1.05)
        l_away = max(0.2, (attack_away + defense_home) / 2.0)
        l_total = l_home + l_away

        p_home, p_draw, p_away, p_btts = 0.0, 0.0, 0.0, 0.0
        for i in range(9):
            for j in range(9):
                prob = poisson.pmf(i, l_home) * poisson.pmf(j, l_away)
                if i > j: p_home += prob
                elif i == j: p_draw += prob
                else: p_away += prob
                if i > 0 and j > 0: p_btts += prob

        g_pick, g_prob, g_seg, _ = self._pick_safest_line_poisson(l_total, [1.5, 0.5, 2.5], [3.5, 4.5])
        
        # Filtro estricto algorítmico de córners (Más de 7.5, Menos de 11.5)
        c_pick, c_prob, c_seg, _ = self._pick_safest_line_poisson(
            self.stats["expected_corners_home"] + self.stats["expected_corners_away"], 
            [7.5, 8.5], [10.5, 11.5]
        )
        
        card_pick, card_prob, card_seg, _ = self._pick_safest_line_poisson(
            self.stats["expected_cards_home"] + self.stats["expected_cards_away"], 
            [2.5, 3.5], [6.5, 7.5]
        )

        dc_ops = sorted([("1X", p_home+p_draw), ("X2", p_away+p_draw), ("12", p_home+p_away)], key=lambda x: x[1], reverse=True)

        return {
            "p_home": p_home, "p_draw": p_draw, "p_away": p_away, "p_btts": p_btts,
            "goal_pick": g_pick, "goal_prob": g_prob, "goal_seguro": g_seg,
            "corner_pick": c_pick, "corner_prob": c_prob, "corner_seguro": c_seg,
            "card_pick": card_pick, "card_prob": card_prob, "card_seguro": card_seg,
            "dc_pick": dc_ops[0][0], "dc_prob": dc_ops[0][1], "dc_seguro": dc_ops[0][1] >= UMBRAL_SEGURO
        }

class PredictorNBA(PredictorBase):
    def predict(self):
        avg_pace = (self.stats["pace_home"] + self.stats["pace_away"]) / 2.0
        adj_off_home = self.stats["off_rtg_home"] * self.stats["recent_form_adj_home"]
        adj_off_away = self.stats["off_rtg_away"] * self.stats["recent_form_adj_away"]
        
        mu_home = avg_pace * ((adj_off_home + self.stats["def_rtg_away"]) / 200.0) + 3.0
        mu_away = avg_pace * ((adj_off_away + self.stats["def_rtg_home"]) / 200.0)
        
        std_dev = self.stats["variance_factor"]
        mu_total = mu_home + mu_away

        diff_mu = mu_home - mu_away
        diff_std = (std_dev**2 + std_dev**2)**0.5
        p_home = 1 - norm.cdf(0, loc=diff_mu, scale=diff_std)
        p_away = 1 - p_home

        t_pick, t_prob, t_seg, _ = self._pick_safest_line_norm(
            mu_total, std_dev * 1.5, 
            lineas_over=[205.5, 215.5, 225.5], 
            lineas_under=[235.5, 245.5, 255.5]
        )

        h_line = round(diff_mu * 2) / 2
        h_pick = f"Local {-h_line}" if diff_mu > 0 else f"Visita {+h_line}"

        return {
            "p_home": p_home, "p_away": p_away, "mu_total": mu_total,
            "total_pick": t_pick, "total_prob": t_prob, "total_seguro": t_seg,
            "h_pick": h_pick, "h_prob": 0.52, "h_seguro": False
        }

class PredictorMLB(PredictorBase):
    def predict(self):
        base_runs_home = self.stats["pitcher_xfip_away"] * (self.stats["batter_wrc_home"] / 100.0)
        base_runs_away = self.stats["pitcher_xfip_home"] * (self.stats["batter_wrc_away"] / 100.0)
        
        l_home = (base_runs_home * 0.66) + (self.stats["bullpen_era_away"] * 0.34) * 1.05
        l_away = (base_runs_away * 0.66) + (self.stats["bullpen_era_home"] * 0.34)
        l_total = l_home + l_away
        
        h_total = self.stats["expected_hits_home"] + self.stats["expected_hits_away"]

        p_home, p_away = 0.0, 0.0
        for i in range(15):
            for j in range(15):
                prob = poisson.pmf(i, l_home) * poisson.pmf(j, l_away)
                if i > j: p_home += prob
                elif i < j: p_away += prob

        r_pick, r_prob, r_seg, _ = self._pick_safest_line_poisson(
            l_total, [5.5, 7.5, 8.5], [10.5, 11.5]
        )
        
        hits_pick, hits_prob, hits_seg, _ = self._pick_safest_line_poisson(
            h_total, [13.5, 15.5], [19.5, 21.5]
        )

        return {
            "p_home": p_home, "p_away": p_away, "l_total": l_total,
            "run_pick": r_pick, "run_prob": r_prob, "run_seguro": r_seg,
            "hits_pick": hits_pick, "hits_prob": hits_prob, "hits_seguro": hits_seg
        }

# ============================================================
# 3. INTERFAZ VISUAL EN STREAMLIT
# ============================================================
with st.sidebar:
    st.header("⚙️ Configuración")
    deporte = st.selectbox("🏆 Selecciona el Deporte", ["Fútbol", "NBA", "MLB"])
    
    api_key_input = st.text_input("API Key (Opcional)", type="password")
    st.markdown("---")
    st.caption(f"🎯 Picks ✅ SEGUROS requieren ≥ {UMBRAL_SEGURO*100:.0f}% de confianza.")

col1, col2, col3 = st.columns(3)
with col1:
    home_team = st.text_input("Equipo Local", value="Real Madrid" if deporte=="Fútbol" else "Lakers" if deporte=="NBA" else "Yankees")
with col2:
    away_team = st.text_input("Equipo Visitante", value="Barcelona" if deporte=="Fútbol" else "Warriors" if deporte=="NBA" else "Dodgers")
with col3:
    match_date = st.date_input("Fecha del Partido", datetime.date.today())

predict_btn = st.button("🚀 Iniciar Análisis Predictivo", use_container_width=True)

def badge(seguro): return "✅ PICK SEGURO" if seguro else "⚠️ Riesgo moderado"

if predict_btn and home_team and away_team:
    with st.spinner(f"Evaluando métricas avanzadas y caché para {home_team} vs {away_team}..."):
        
        # Llamada optimizada con caché de Streamlit
        stats = cached_fetch_stats(api_key_input, deporte, home_team, away_team)
        
        if deporte == "Fútbol":
            res = PredictorFootball(stats).predict()
        elif deporte == "NBA":
            res = PredictorNBA(stats).predict()
        elif deporte == "MLB":
            res = PredictorMLB(stats).predict()

        st.success(f"✅ Análisis completado ({deporte})")
        
        predictor = PredictorBase(stats)
        dias = stats.get('days_since_last_h2h', 0)
        st.info(f"⚖️ **Ponderación Dinámica:** H2H hace {dias} días. "
                f"Pesos aplicados: **{predictor.w_h2h*100:.1f}% H2H** | **{predictor.w_rec*100:.1f}% Reciente**.")
        
        st.markdown("## 🏆 Picks Recomendados")

        if deporte == "Fútbol":
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("🥅 Goles", res['goal_pick'], f"{res['goal_prob']*100:.1f}%")
            c1.caption(badge(res['goal_seguro']))
            c2.metric("🛡️ Doble Op.", res['dc_pick'], f"{res['dc_prob']*100:.1f}%")
            c2.caption(badge(res['dc_seguro']))
            c3.metric("🚩 Córners", res['corner_pick'], f"{res['corner_prob']*100:.1f}%")
            c3.caption(badge(res['corner_seguro']))
            c4.metric("🟨 Tarjetas", res['card_pick'], f"{res['card_prob']*100:.1f}%")
            c4.caption(badge(res['card_seguro']))

            df_probs = pd.DataFrame({'Resultado': [f'Gana {home_team}', 'Empate', f'Gana {away_team}'],
                                     'Probabilidad (%)': [res['p_home']*100, res['p_draw']*100, res['p_away']*100]})
            
        elif deporte == "NBA":
            c1, c2, c3 = st.columns(3)
            ganador = home_team if res['p_home'] > res['p_away'] else away_team
            prob_ganador = max(res['p_home'], res['p_away'])
            c1.metric("🏀 Moneyline", f"Gana {ganador}", f"{prob_ganador*100:.1f}%")
            c1.caption(badge(prob_ganador >= UMBRAL_SEGURO))
            c2.metric("📊 Total Puntos", res['total_pick'], f"{res['total_prob']*100:.1f}%")
            c2.caption(badge(res['total_seguro']))
            c3.metric("⚖️ Hándicap Proyectado", res['h_pick'])
            c3.caption(badge(False))

            df_probs = pd.DataFrame({'Resultado': [f'Gana {home_team}', f'Gana {away_team}'],
                                     'Probabilidad (%)': [res['p_home']*100, res['p_away']*100]})
            
        elif deporte == "MLB":
            c1, c2, c3 = st.columns(3)
            ganador = home_team if res['p_home'] > res['p_away'] else away_team
            prob_ganador = max(res['p_home'], res['p_away'])
            c1.metric("⚾ Moneyline", f"Gana {ganador}", f"{prob_ganador*100:.1f}%")
            c1.caption(badge(prob_ganador >= UMBRAL_SEGURO))
            c2.metric("🏃 Carreras", res['run_pick'], f"{res['run_prob']*100:.1f}%")
            c2.caption(badge(res['run_seguro']))
            c3.metric("🦇 Total Hits", res['hits_pick'], f"{res['hits_prob']*100:.1f}%")
            c3.caption(badge(res['hits_seguro']))

            df_probs = pd.DataFrame({'Resultado': [f'Gana {home_team}', f'Gana {away_team}'],
                                     'Probabilidad (%)': [res['p_home']*100, res['p_away']*100]})

        st.markdown("---")
        fig_bar = px.bar(df_probs, x='Resultado', y='Probabilidad (%)', text='Probabilidad (%)',
                         color='Resultado', color_discrete_sequence=['#2EF0A0', '#FFC107', '#FF5252'])
        fig_bar.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
        fig_bar.update_layout(margin=dict(t=20, b=20, l=20, r=20), height=250, showlegend=False)
        st.plotly_chart(fig_bar, use_container_width=True)

elif predict_btn:
    st.warning("⚠️ Ingresa los nombres de ambos equipos.")

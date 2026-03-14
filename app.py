import streamlit as st
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics.pairwise import cosine_similarity
import pulp as pl
import os, tempfile, subprocess

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="CricIntel",
    page_icon="🏏",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# GLOBAL CSS  — dark premium look
# ─────────────────────────────────────────────
st.markdown("""
<style>
/* Base */
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.block-container { padding-top: 1.5rem; padding-bottom: 2rem; }

/* Sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0a0f1e 0%, #0d1b2a 100%);
    border-right: 1px solid #1e3a5f;
}
section[data-testid="stSidebar"] * { color: #e0e6ef !important; }
section[data-testid="stSidebar"] .stRadio label { font-size: 0.95rem; }

/* Title banner */
.cricintel-banner {
    background: linear-gradient(135deg, #0a0f1e 0%, #0d2137 60%, #061a30 100%);
    border: 1px solid #00d4ff33;
    border-radius: 12px;
    padding: 1.6rem 2rem;
    margin-bottom: 1.5rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
}
.cricintel-banner h1 {
    color: #00d4ff;
    font-size: 2.4rem;
    font-weight: 800;
    letter-spacing: 0.15em;
    margin: 0;
    text-shadow: 0 0 20px #00d4ff66;
}
.cricintel-banner p { color: #7ba7c4; margin: 0.3rem 0 0; font-size: 0.95rem; }

/* Metric cards */
.metric-card {
    background: linear-gradient(135deg, #0d1b2a, #0a1628);
    border: 1px solid #1e3a5f;
    border-radius: 10px;
    padding: 1rem 1.2rem;
    text-align: center;
}
.metric-card .val { font-size: 2rem; font-weight: 700; color: #00d4ff; }
.metric-card .lbl { font-size: 0.78rem; color: #7ba7c4; text-transform: uppercase; letter-spacing: 0.08em; margin-top: 0.2rem; }

/* Section headers */
.section-header {
    background: linear-gradient(90deg, #00d4ff15, transparent);
    border-left: 3px solid #00d4ff;
    padding: 0.5rem 1rem;
    border-radius: 0 8px 8px 0;
    margin: 1.5rem 0 0.8rem;
    font-size: 1.05rem;
    font-weight: 600;
    color: #c8e6f5;
    letter-spacing: 0.04em;
}

/* Role badge */
.badge {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 20px;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.06em;
}
.badge-bat  { background:#1a3a2a; color:#4ade80; border:1px solid #4ade8044; }
.badge-bowl { background:#1a1a3a; color:#818cf8; border:1px solid #818cf844; }
.badge-ar   { background:#2a2a1a; color:#fbbf24; border:1px solid #fbbf2444; }
.badge-wk   { background:#2a1a1a; color:#f87171; border:1px solid #f8717144; }

/* Player card */
.player-card {
    background: #0d1b2a;
    border: 1px solid #1e3a5f;
    border-radius: 10px;
    padding: 0.9rem 1.1rem;
    margin-bottom: 0.6rem;
    transition: border-color 0.2s;
}
.player-card:hover { border-color: #00d4ff66; }
.player-card .pname { font-size: 1rem; font-weight: 600; color: #e0e6ef; }
.player-card .pstat { font-size: 0.8rem; color: #7ba7c4; margin-top: 0.2rem; }

/* Explain card */
.explain-card {
    background: #080f1a;
    border: 1px solid #1e3a5f;
    border-radius: 8px;
    padding: 0.8rem 1rem;
    margin-bottom: 0.5rem;
}
.explain-card .ename { font-size: 0.95rem; font-weight: 600; color: #00d4ff; }
.explain-card .escores { font-size: 0.78rem; color: #7ba7c4; margin-top: 0.15rem; }
.explain-pos { color: #4ade80; font-size: 0.82rem; }
.explain-neg { color: #f87171; font-size: 0.82rem; }

/* Divider */
.cricdiv { border: none; border-top: 1px solid #1e3a5f; margin: 1.2rem 0; }

/* Download button */
.stDownloadButton button {
    background: linear-gradient(135deg, #00d4ff22, #0066aa22) !important;
    border: 1px solid #00d4ff55 !important;
    color: #00d4ff !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
}
.stDownloadButton button:hover {
    background: linear-gradient(135deg, #00d4ff33, #0066aa33) !important;
    border-color: #00d4ff !important;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    background: #0a0f1e;
    border-radius: 8px;
    padding: 4px;
}
.stTabs [data-baseweb="tab"] {
    color: #7ba7c4 !important;
    font-weight: 500;
}
.stTabs [aria-selected="true"] {
    background: #0d2137 !important;
    color: #00d4ff !important;
    border-radius: 6px;
}

/* Info/warning/success */
.stAlert { border-radius: 8px; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def nz(s, default=0.0):
    if s is None:
        return pd.Series(dtype=float)
    return s.fillna(default)

def norm01(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce").fillna(0).astype(float)
    mn, mx = s.min(), s.max()
    if mx <= mn:
        return pd.Series(np.full(len(s), 0.5), index=s.index)
    return (s - mn) / (mx - mn)

def clamp01(x):
    return np.clip(x, 0, 1)

def to_csv_bytes(df_):
    return df_.to_csv(index=False).encode("utf-8")

def stable_noise(id_series: pd.Series, mod=97) -> np.ndarray:
    s = pd.to_numeric(id_series, errors="coerce").fillna(0).astype(int)
    return (s % mod) / float(mod)

def safe_numeric(df, cols, default=0.0):
    for c in cols:
        if c not in df.columns:
            df[c] = default
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(default)
    return df

def role_color(role):
    return {"BAT": "#4ade80", "BOWL": "#818cf8", "AR": "#fbbf24", "WK": "#f87171"}.get(role, "#7ba7c4")

def role_badge(role):
    cls = {"BAT": "badge-bat", "BOWL": "badge-bowl", "AR": "badge-ar", "WK": "badge-wk"}.get(role, "badge-bat")
    return f'<span class="badge {cls}">{role}</span>'

def section(title, icon="▸"):
    st.markdown(f'<div class="section-header">{icon} {title}</div>', unsafe_allow_html=True)

def metric_row(items: list):
    """items = list of (label, value, delta=None)"""
    cols = st.columns(len(items))
    for col, (lbl, val, *rest) in zip(cols, items):
        delta = rest[0] if rest else None
        col.metric(lbl, val, delta)

def cric_divider():
    st.markdown('<hr class="cricdiv">', unsafe_allow_html=True)


# ─────────────────────────────────────────────
# OPTIMISERS
# ─────────────────────────────────────────────
def optimize_squad_soft(df, budget_limit, max_players, min_role,
                        price_col, extra_min_flags=None, extra_max_flags=None,
                        lock_players=None, max_single_price=None,
                        penalty_weight=2.5, price_concentration_penalty=0.0):
    d = df.copy()
    players_list = d["player"].tolist()
    if not players_list:
        return pd.DataFrame(), {"status": "empty"}

    x = pl.LpVariable.dicts("pick", players_list, lowBound=0, upBound=1, cat="Binary")
    prob = pl.LpProblem("SquadSoft", pl.LpMaximize)

    slack = {}
    if extra_min_flags:
        for col, mn in extra_min_flags.items():
            if col in d.columns and int(mn) > 0:
                slack[col] = pl.LpVariable(f"slack_{col}", lowBound=0, cat="Continuous")

    norm_price = norm01(d[price_col]).values

    prob += (
        pl.lpSum(d.loc[d.player == p, "objective_score"].values[0] * x[p] for p in players_list)
        - float(penalty_weight) * pl.lpSum(slack[c] for c in slack)
        - float(price_concentration_penalty) * pl.lpSum(norm_price[i] * x[p] for i, p in enumerate(players_list))
    )

    prob += pl.lpSum(d.loc[d.player == p, price_col].values[0] * x[p] for p in players_list) <= float(budget_limit)
    prob += pl.lpSum(x[p] for p in players_list) <= int(max_players)

    def role_count(code):
        return pl.lpSum(x[p] for p in players_list if d.loc[d.player == p, "role"].values[0] == code)

    for r, mn in min_role.items():
        prob += role_count(r) >= int(mn)

    if extra_min_flags:
        for col, mn in extra_min_flags.items():
            if col in d.columns and int(mn) > 0:
                cnt = pl.lpSum(x[p] for p in players_list if int(d.loc[d.player == p, col].values[0]) == 1)
                prob += cnt + slack[col] >= int(mn)

    if extra_max_flags:
        for col, mx in extra_max_flags.items():
            if col in d.columns and mx is not None:
                prob += pl.lpSum(x[p] for p in players_list if int(d.loc[d.player == p, col].values[0]) == 1) <= int(mx)

    if max_single_price is not None:
        for p in players_list:
            prob += d.loc[d.player == p, price_col].values[0] * x[p] <= float(max_single_price)

    if lock_players:
        for lp in lock_players:
            if lp in players_list:
                prob += x[lp] == 1

    prob.solve(pl.PULP_CBC_CMD(msg=False))

    picked = [p for p in players_list if x[p].value() == 1]
    squad = d[d.player.isin(picked)].copy()

    violations = {}
    for col in slack:
        try:
            violations[col] = float(slack[col].value())
        except Exception:
            violations[col] = None

    metrics = {
        "status": "ok" if len(squad) else "no_solution",
        "count": int(len(squad)),
        "spend": float(squad[price_col].sum()) if len(squad) else 0.0,
        "objective": float(squad["objective_score"].sum()) if len(squad) else 0.0,
        "violations": violations
    }
    for r in ["BAT", "BOWL", "AR", "WK"]:
        metrics[r] = int((squad["role"] == r).sum()) if len(squad) else 0
    return squad, metrics


def pick_best_xi(squad, xi_size, xi_min_role, max_overseas_xi, enforce_left_in_top4):
    if len(squad) == 0:
        return pd.DataFrame(), {"status": "empty"}

    d = squad.copy()
    players_list = d["player"].tolist()

    x = pl.LpVariable.dicts("xi", players_list, lowBound=0, upBound=1, cat="Binary")
    prob = pl.LpProblem("BestXI", pl.LpMaximize)

    prob += pl.lpSum(d.loc[d.player == p, "xi_score"].values[0] * x[p] for p in players_list)
    prob += pl.lpSum(x[p] for p in players_list) == int(xi_size)

    def role_count(code):
        return pl.lpSum(x[p] for p in players_list if d.loc[d.player == p, "role"].values[0] == code)

    for r, mn in xi_min_role.items():
        prob += role_count(r) >= int(mn)

    if max_overseas_xi is not None and "is_overseas" in d.columns:
        prob += pl.lpSum(x[p] for p in players_list if int(d.loc[d.player == p, "is_overseas"].values[0]) == 1) <= int(max_overseas_xi)

    if enforce_left_in_top4 and "bat_hand" in d.columns and "is_top4_candidate" in d.columns:
        prob += pl.lpSum(
            x[p] for p in players_list
            if (int(d.loc[d.player == p, "is_top4_candidate"].values[0]) == 1 and
                str(d.loc[d.player == p, "bat_hand"].values[0]) == "L")
        ) >= 1

    prob.solve(pl.PULP_CBC_CMD(msg=False))

    picked = [p for p in players_list if x[p].value() == 1]
    xi = d[d.player.isin(picked)].copy()
    metrics = {"status": "ok" if len(xi) else "no_solution", "count": int(len(xi)),
               "xi_score": float(xi["xi_score"].sum()) if len(xi) else 0.0}
    return xi, metrics


# ─────────────────────────────────────────────
# CORE DATA PIPELINE
# ─────────────────────────────────────────────
def build_base_df(players: pd.DataFrame, perf: pd.DataFrame, contracts: pd.DataFrame | None = None):
    # ── merge ──────────────────────────────────────────────────────────────
    df = players.merge(perf, on="player_id", how="left", suffixes=("", "_perf"))

    # FIX: resolve age conflict — prefer players.csv age
    if "age" not in df.columns:
        if "age_perf" in df.columns:
            df["age"] = df["age_perf"]
        else:
            df["age"] = 25  # safe fallback

    if contracts is not None:
        df = df.merge(contracts, on="player_id", how="left")

    # ── required column validation ─────────────────────────────────────────
    req_players = {"player", "player_id", "role", "age"}
    req_perf    = {"player_id", "matches", "runs", "strike_rate", "wickets",
                   "economy", "dot_ball_pct", "boundary_pct"}

    missing_p = req_players - set(df.columns)
    missing_r = req_perf    - set(df.columns)
    if missing_p:
        st.error(f"❌ players.csv missing columns: {sorted(missing_p)}")
        st.stop()
    if missing_r:
        st.error(f"❌ performance.csv missing columns: {sorted(missing_r)}")
        st.stop()

    # ── numeric safety ─────────────────────────────────────────────────────
    df = safe_numeric(df, ["matches","runs","strike_rate","wickets","economy",
                           "dot_ball_pct","boundary_pct","age"], 0.0)

    # ── optional phase cols ────────────────────────────────────────────────
    for col in ["pp_sr","middle_sr","death_sr","pp_runs","death_runs",
                "pp_eco","middle_eco","death_eco","pp_wkts","death_wkts",
                "injury_risk","availability_risk"]:
        if col not in df.columns:
            df[col] = np.nan

    for col in ["is_pp_bowler","is_death_bowler","is_pp_batter","is_death_hitter",
                "is_spinner","is_pacer","is_overseas"]:
        if col not in df.columns:
            df[col] = 0

    if "bat_hand" not in df.columns:
        df["bat_hand"] = "R"
    if "bowling_arm" not in df.columns:
        df["bowling_arm"] = np.nan
    if "spin_type" not in df.columns:
        df["spin_type"] = np.nan

    # ── deterministic phase proxies ────────────────────────────────────────
    noise = stable_noise(df["player_id"])

    if df["pp_sr"].isna().all():
        df["pp_sr"]     = df["strike_rate"] + (-2 + 6*(noise - 0.5))
        df["middle_sr"] = df["strike_rate"] + (-6 + 8*(noise - 0.5))
        df["death_sr"]  = df["strike_rate"] + (8 + 10*(noise - 0.5))
        df["pp_runs"]   = (df["runs"] * (0.33 + 0.06*(noise - 0.5))).astype(float)
        df["death_runs"]= (df["runs"] * (0.23 + 0.06*(noise - 0.5))).astype(float)

    if df["pp_eco"].isna().all():
        df["pp_eco"]    = df["economy"] + (-0.25 + 0.6*(noise - 0.5))
        df["middle_eco"]= df["economy"] + (0.00  + 0.5*(noise - 0.5))
        df["death_eco"] = df["economy"] + (0.55  + 0.8*(noise - 0.5))
        df["pp_wkts"]   = (df["wickets"] * (0.33 + 0.05*(noise - 0.5))).astype(float)
        df["death_wkts"]= (df["wickets"] * (0.23 + 0.05*(noise - 0.5))).astype(float)

    # ── role inference ─────────────────────────────────────────────────────
    if "batting_role"  not in df.columns: df["batting_role"]  = "ANCHOR"
    if "bowling_role"  not in df.columns: df["bowling_role"]  = "MIDDLE"

    is_bat  = df["role"].isin(["BAT","AR","WK"])
    is_bowl = df["role"].isin(["BOWL","AR"])

    df.loc[is_bat, "batting_role"]  = "ANCHOR"
    df.loc[is_bat & (df["pp_sr"]    >= df["pp_sr"].quantile(0.70)),    "batting_role"] = "OPENER"
    df.loc[is_bat & (df["death_sr"] >= df["death_sr"].quantile(0.75)), "batting_role"] = "FINISHER"

    df.loc[is_bowl, "bowling_role"] = "MIDDLE"
    df.loc[is_bowl & ((df["pp_wkts"]    >= df["pp_wkts"].quantile(0.65))    | (df["is_pp_bowler"].astype(int)==1)),    "bowling_role"] = "PP"
    df.loc[is_bowl & ((df["death_wkts"] >= df["death_wkts"].quantile(0.70)) | (df["is_death_bowler"].astype(int)==1)), "bowling_role"] = "DEATH"

    df["is_opener"]       = (df["batting_role"]  == "OPENER").astype(int)
    df["is_finisher"]     = (df["batting_role"]  == "FINISHER").astype(int)
    df["is_death_bowler2"]= (df["bowling_role"]  == "DEATH").astype(int)
    df["is_pp_bowler2"]   = (df["bowling_role"]  == "PP").astype(int)
    df["is_top4_candidate"] = (
        df["role"].isin(["BAT","WK","AR"]) &
        ((df["is_opener"]==1) | (df["runs"] >= df["runs"].quantile(0.60)))
    ).astype(int)

    # ── bowling arm + spin type ────────────────────────────────────────────
    if df["bowling_arm"].isna().all():
        df["bowling_arm"] = np.where(noise < 0.28, "L", "R")
    df["bowling_arm"] = df["bowling_arm"].astype(str).str.upper()
    df.loc[~df["bowling_arm"].isin(["L","R"]), "bowling_arm"] = "R"

    if df["spin_type"].isna().all():
        df["spin_type"] = "NONE"
        spin_mask = df["is_spinner"].astype(int) == 1
        df.loc[spin_mask, "spin_type"] = np.where(noise[spin_mask] < 0.55, "OFF", "LEG")
    df["spin_type"] = df["spin_type"].astype(str).str.upper()

    df["is_left_arm_spinner"]  = ((df["is_spinner"].astype(int)==1) & (df["bowling_arm"]=="L")).astype(int)
    df["is_right_arm_spinner"] = ((df["is_spinner"].astype(int)==1) & (df["bowling_arm"]=="R")).astype(int)
    df["is_off_spinner"]       = ((df["is_spinner"].astype(int)==1) & (df["spin_type"]=="OFF")).astype(int)
    df["is_leg_spinner"]       = ((df["is_spinner"].astype(int)==1) & (df["spin_type"]=="LEG")).astype(int)
    df["is_left_arm_pacer"]    = ((df["is_pacer"].astype(int)==1)   & (df["bowling_arm"]=="L")).astype(int)
    df["is_right_arm_pacer"]   = ((df["is_pacer"].astype(int)==1)   & (df["bowling_arm"]=="R")).astype(int)

    # ── risk model ─────────────────────────────────────────────────────────
    if df["injury_risk"].isna().all():
        workload = norm01(df["matches"])
        df["injury_risk"] = clamp01(0.12 + 0.02*(df["age"]-df["age"].min()) + 0.20*workload + 0.10*df["is_pacer"].astype(float))

    if df["availability_risk"].isna().all():
        df["availability_risk"] = clamp01(0.08 + 0.15*df["is_overseas"].astype(float) + 0.10*norm01(df["matches"]))

    df["total_risk"] = clamp01(0.65*df["injury_risk"].astype(float) + 0.35*df["availability_risk"].astype(float))
    return df


# ─────────────────────────────────────────────
# SHARED PHASE SCORES
# ─────────────────────────────────────────────
def compute_phase_scores(df, w_pp=1.0, w_mid=1.0, w_death=1.2):
    pp_bat    = 0.6*norm01(df["pp_sr"])    + 0.4*norm01(df["pp_runs"])
    mid_bat   = norm01(df["middle_sr"])
    death_bat = 0.6*norm01(df["death_sr"]) + 0.4*norm01(df["death_runs"])

    pp_bowl    = 0.6*norm01(df["pp_wkts"])    + 0.4*(1 - norm01(df["pp_eco"]))
    mid_bowl   = 0.7*(1 - norm01(df["middle_eco"])) + 0.3*norm01(df["dot_ball_pct"])
    death_bowl = 0.6*norm01(df["death_wkts"]) + 0.4*(1 - norm01(df["death_eco"]))

    df["pp_bat_score"]    = pp_bat
    df["mid_bat_score"]   = mid_bat
    df["death_bat_score"] = death_bat
    df["pp_bowl_score"]   = pp_bowl
    df["mid_bowl_score"]  = mid_bowl
    df["death_bowl_score"]= death_bowl

    impact = []
    for r, b, w in zip(
        df["role"].fillna("BAT"),
        (w_pp*pp_bat + w_mid*mid_bat + w_death*death_bat),
        (w_pp*pp_bowl + w_mid*mid_bowl + w_death*death_bowl)
    ):
        if r == "BAT":   impact.append(b)
        elif r == "BOWL":impact.append(w)
        elif r == "AR":  impact.append(0.55*b + 0.45*w)
        elif r == "WK":  impact.append(0.95*b + 0.05)
        else:            impact.append(b)

    df["match_impact_score"] = norm01(pd.Series(impact, index=df.index))
    return df


# ─────────────────────────────────────────────
# SIMILARITY SEARCH  (FIXED)
# ─────────────────────────────────────────────
def get_similar_players(df_all: pd.DataFrame, target_name: str, top_k: int = 12):
    trow = df_all[df_all["player"] == target_name].head(1)
    if len(trow) == 0:
        return pd.DataFrame()

    feat_cols = [
        "pp_bat_score","mid_bat_score","death_bat_score",
        "pp_bowl_score","mid_bowl_score","death_bowl_score",
        "strike_rate","economy","dot_ball_pct","boundary_pct",
        "match_impact_score","total_risk"
    ]
    feat_cols = [c for c in feat_cols if c in df_all.columns]

    # FIX: exclude target from candidate pool BEFORE computing similarity
    candidates = df_all[df_all["player"] != target_name].copy()
    t_vec = trow[feat_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).values

    # FIX: scale features so no single column dominates
    scaler = StandardScaler()
    cand_feats = candidates[feat_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).values
    t_scaled   = scaler.fit_transform(np.vstack([cand_feats, t_vec]))
    cand_scaled= t_scaled[:-1]
    t_scaled   = t_scaled[-1].reshape(1, -1)

    sims = cosine_similarity(cand_scaled, t_scaled).reshape(-1)
    candidates = candidates.copy()
    candidates["similarity"] = sims

    # same-role first, then sorted by similarity
    target_role = trow["role"].values[0]
    same_role   = candidates[candidates["role"] == target_role].sort_values("similarity", ascending=False).head(top_k)
    return same_role


# ─────────────────────────────────────────────
# SCOUT MODE
# ─────────────────────────────────────────────
def run_scout_mode():
    st.markdown('<div class="section-header">📁 Upload Data</div>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        players_f    = st.file_uploader("players.csv", type="csv", key="sc_pl",
                                         help="Required cols: player_id, player, role, age, bat_hand")
    with c2:
        performance_f = st.file_uploader("performance.csv", type="csv", key="sc_pf",
                                          help="Required cols: player_id, matches, runs, strike_rate, wickets, economy, dot_ball_pct, boundary_pct")

    if not all([players_f, performance_f]):
        st.info("👆 Upload both CSVs above to unlock Scout Mode.")
        st.stop()

    players = pd.read_csv(players_f)
    perf    = pd.read_csv(performance_f)
    df      = build_base_df(players, perf)
    df      = compute_phase_scores(df)

    # ── SNAPSHOT ──────────────────────────────────────────────────────────
    section("Squad Snapshot", "📊")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Players",  len(df))
    c2.metric("Batters",        int((df["role"]=="BAT").sum()))
    c3.metric("Bowlers",        int((df["role"]=="BOWL").sum()))
    c4.metric("All-rounders",   int((df["role"]=="AR").sum()))
    c5.metric("Wicketkeepers",  int((df["role"]=="WK").sum()))

    # ── ROLE DISTRIBUTION BAR ─────────────────────────────────────────────
    cric_divider()
    section("Role & Nationality Breakdown", "🌍")
    t1, t2 = st.columns(2)

    with t1:
        role_counts = df["role"].value_counts().reset_index()
        role_counts.columns = ["Role","Count"]
        st.bar_chart(role_counts.set_index("Role"), color="#00d4ff")

    with t2:
        if "country" in df.columns:
            country_counts = df["country"].value_counts().head(8).reset_index()
            country_counts.columns = ["Country","Count"]
            st.bar_chart(country_counts.set_index("Country"), color="#818cf8")

    # ── TOP PROFILES TABLE ────────────────────────────────────────────────
    cric_divider()
    section("Top Profiles", "🏆")

    show_cols = ["player","role","age","country","bat_hand","bowling_arm","spin_type",
                 "batting_role","bowling_role","matches","runs","strike_rate",
                 "wickets","economy","dot_ball_pct","boundary_pct",
                 "match_impact_score","total_risk"]
    show_cols = [c for c in show_cols if c in df.columns]

    top_df = df[show_cols].sort_values("match_impact_score", ascending=False).head(30)

    # colour-code match_impact_score
    st.dataframe(
        top_df.style.format({"match_impact_score": "{:.3f}", "total_risk": "{:.3f}",
                     "strike_rate": "{:.1f}", "economy": "{:.2f}"}),
        use_container_width=True,
        height=420
    )

    # ── SIMILARITY SEARCH ─────────────────────────────────────────────────
    cric_divider()
    section("Similarity Search", "🔍")
    st.caption("Finds the closest stylistic matches to any player using phase profile + performance fingerprint.")

    col_sel, col_k = st.columns([3,1])
    target    = col_sel.selectbox("Select a player", df["player"].tolist(), index=0)
    top_k_sim = col_k.number_input("Top K", min_value=5, max_value=25, value=10, step=1)

    sim = get_similar_players(df, target, top_k=int(top_k_sim))

    if len(sim):
        # show target profile
        trow = df[df["player"]==target].iloc[0]
        st.markdown(f"""
        <div class="player-card">
            <div class="pname">🎯 {target} {role_badge(trow['role'])}</div>
            <div class="pstat">
                Runs: <b>{int(trow.get('runs',0))}</b> &nbsp;|&nbsp;
                SR: <b>{trow.get('strike_rate',0):.1f}</b> &nbsp;|&nbsp;
                Wkts: <b>{int(trow.get('wickets',0))}</b> &nbsp;|&nbsp;
                Eco: <b>{trow.get('economy',0):.2f}</b> &nbsp;|&nbsp;
                Impact: <b>{trow.get('match_impact_score',0):.3f}</b>
            </div>
        </div>
        """, unsafe_allow_html=True)

        sim_show = sim[["player","role","batting_role","bowling_role",
                         "similarity","match_impact_score","total_risk",
                         "runs","strike_rate","wickets","economy"]].copy()
        sim_show = sim_show.rename(columns={"similarity":"Similarity ↓"})

        st.dataframe(
            sim_show.style.format({"Similarity ↓": "{:.3f}", "match_impact_score": "{:.3f}",
                         "total_risk": "{:.3f}", "strike_rate": "{:.1f}", "economy": "{:.2f}"}),
            use_container_width=True,
            height=380
        )
    else:
        st.warning("No similar players found.")

    # ── GAP-FILL RECOMMENDER ──────────────────────────────────────────────
    cric_divider()
    section("Gap-Fill Recommender", "🧩")
    st.caption("Select your current players in a role — get the best available replacements or additions.")

    gap_options = [
        "Pacer (any)","Left-arm Pacer","Right-arm Pacer",
        "Spinner (any)","Left-arm Spinner","Right-arm Spinner",
        "Off-spinner","Leg-spinner",
        "Opener","Top-4 (anchor)","Finisher"
    ]

    g1, g2 = st.columns([2,1])
    gap_type   = g1.selectbox("Gap to fill", gap_options)
    allow_small= g2.checkbox("Allow 2-player unit", value=True)

    def gap_pool(d, gap):
        pools = {
            "Pacer (any)":        d[d["is_pacer"].astype(int)==1],
            "Left-arm Pacer":     d[d["is_left_arm_pacer"].astype(int)==1],
            "Right-arm Pacer":    d[d["is_right_arm_pacer"].astype(int)==1],
            "Spinner (any)":      d[d["is_spinner"].astype(int)==1],
            "Left-arm Spinner":   d[d["is_left_arm_spinner"].astype(int)==1],
            "Right-arm Spinner":  d[d["is_right_arm_spinner"].astype(int)==1],
            "Off-spinner":        d[d["is_off_spinner"].astype(int)==1],
            "Leg-spinner":        d[d["is_leg_spinner"].astype(int)==1],
            "Opener":             d[d["is_opener"].astype(int)==1],
            "Top-4 (anchor)":     d[d["is_top4_candidate"].astype(int)==1],
            "Finisher":           d[d["is_finisher"].astype(int)==1],
        }
        return pools.get(gap, d)

    pool_df = gap_pool(df, gap_type)

    if len(pool_df) == 0:
        st.warning("No players found for this gap type. Check your flag columns in the CSV.")
        st.stop()

    current_players = st.multiselect(
        f"Your current players ({gap_type})",
        options=pool_df["player"].tolist(),
        default=[]
    )

    n_sel = len(current_players)
    if n_sel == 0:
        st.info("Select at least 1 player to get recommendations.")
        st.stop()
    elif n_sel == 1:
        st.info("**Mode: Player-style match** — finding closest stylistic alternatives.")
    elif n_sel == 2 and not allow_small:
        st.warning("Select 3+ for stable unit match, or enable 'Allow 2-player unit'.")
        st.stop()
    elif n_sel == 2:
        st.warning("**Mode: Small unit (2 players)** — less stable, use with caution.")
    else:
        st.success(f"**Mode: Unit match ({n_sel} players)** — high confidence recommendations.")

    def feat_cols_for_gap(gap):
        bowling_feats = ["pp_bowl_score","mid_bowl_score","death_bowl_score",
                         "pp_eco","middle_eco","death_eco","pp_wkts","death_wkts",
                         "economy","dot_ball_pct","match_impact_score","total_risk"]
        batting_feats = ["pp_bat_score","mid_bat_score","death_bat_score",
                         "pp_sr","middle_sr","strike_rate","pp_runs","runs",
                         "boundary_pct","match_impact_score","total_risk"]
        death_bat_feats = ["death_bat_score","death_sr","death_runs","strike_rate",
                           "boundary_pct","match_impact_score","total_risk"]
        if gap in ["Pacer (any)","Left-arm Pacer","Right-arm Pacer",
                   "Spinner (any)","Left-arm Spinner","Right-arm Spinner",
                   "Off-spinner","Leg-spinner"]:
            return [c for c in bowling_feats if c in df.columns]
        if gap in ["Opener","Top-4 (anchor)"]:
            return [c for c in batting_feats if c in df.columns]
        if gap == "Finisher":
            return [c for c in death_bat_feats if c in df.columns]
        return [c for c in ["match_impact_score","total_risk","strike_rate","economy"] if c in df.columns]

    feat_cols = feat_cols_for_gap(gap_type)
    base = df[df["player"].isin(current_players)].copy()
    cand = pool_df[~pool_df["player"].isin(current_players)].copy()

    if len(cand) == 0:
        st.warning("No candidates available.")
        st.stop()

    # FIX: use StandardScaler for fair cosine comparison
    scaler = StandardScaler()
    all_feats = pd.concat([base[feat_cols], cand[feat_cols]]).apply(pd.to_numeric, errors="coerce").fillna(0.0)
    scaled_all = scaler.fit_transform(all_feats.values)
    base_scaled = scaled_all[:len(base)]
    cand_scaled = scaled_all[len(base):]

    centroid = base_scaled.mean(axis=0).reshape(1,-1)
    sims = cosine_similarity(cand_scaled, centroid).reshape(-1)
    cand = cand.copy()
    cand["unit_fit_score"] = sims
    cand["combined_reco"] = (
        0.70*cand["unit_fit_score"]
        + 0.30*norm01(cand["match_impact_score"])
        - 0.20*norm01(cand["total_risk"])
    )

    rec = cand.sort_values("combined_reco", ascending=False).head(10)

    # Unit profile
    unit_avg = base[feat_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).mean()

    col_up, col_rec = st.columns([1,2])
    with col_up:
        section("Unit Profile", "📐")
        unit_df = unit_avg.reset_index()
        unit_df.columns = ["Metric","Unit Avg"]
        unit_df["Unit Avg"] = unit_df["Unit Avg"].round(3)
        st.dataframe(unit_df, use_container_width=True, height=300)

    with col_rec:
        section("Top 10 Recommendations", "⭐")
        show_rec = ["player","role","age","bat_hand","bowling_arm","spin_type",
                    "batting_role","bowling_role","unit_fit_score","combined_reco",
                    "match_impact_score","total_risk"]
        show_rec = [c for c in show_rec if c in rec.columns]
        st.dataframe(
            rec[show_rec].style.format({"unit_fit_score":"{:.3f}","combined_reco":"{:.3f}",
                         "match_impact_score":"{:.3f}","total_risk":"{:.3f}"}),
            use_container_width=True,
            height=300
        )

    # Explainability
    cric_divider()
    section("Explainability — Top 5", "🔬")
    st.caption("Why each player was recommended vs your current unit.")

    for _, r in rec.head(5).iterrows():
        diffs = {}
        for c in feat_cols:
            try:
                diffs[c] = float(r[c]) - float(unit_avg.get(c, 0.0))
            except Exception:
                continue
        ranked = sorted(diffs.items(), key=lambda kv: kv[1], reverse=True)
        best  = ranked[:3]
        worst = ranked[-1:] if ranked else []

        st.markdown(f"""
        <div class="explain-card">
            <div class="ename">{r['player']} {role_badge(r['role'])}</div>
            <div class="escores">combined_reco: <b>{r.get('combined_reco',0):.3f}</b> &nbsp;|&nbsp; unit_fit: <b>{r.get('unit_fit_score',0):.3f}</b> &nbsp;|&nbsp; impact: <b>{r.get('match_impact_score',0):.3f}</b></div>
        </div>
        """, unsafe_allow_html=True)

        for k, v in best:
            st.markdown(f'<span class="explain-pos">▲ <b>{k}</b> better than unit avg ({v:+.3f})</span>', unsafe_allow_html=True)
        for k, v in worst:
            st.markdown(f'<span class="explain-neg">▼ <b>{k}</b> trade-off vs unit ({v:+.3f})</span>', unsafe_allow_html=True)
        st.write("")

    # Downloads
    cric_divider()
    d1, d2 = st.columns(2)
    d1.download_button("⬇ Download Full Scout Table", to_csv_bytes(df), "scout_table.csv", "text/csv")
    d2.download_button("⬇ Download Recommendations", to_csv_bytes(rec), "gap_fill_reco.csv", "text/csv")


# ─────────────────────────────────────────────
# AUCTION MODE
# ─────────────────────────────────────────────
def run_auction_mode():
    section("Upload Data", "📁")
    c1, c2 = st.columns(2)
    with c1:
        players_f    = st.file_uploader("players.csv",     type="csv", key="au_pl")
        performance_f= st.file_uploader("performance.csv", type="csv", key="au_pf")
    with c2:
        contracts_f  = st.file_uploader("contracts.csv",   type="csv", key="au_ct")
        budget_f     = st.file_uploader("budget.csv",      type="csv", key="au_bd")

    if not all([players_f, performance_f, contracts_f, budget_f]):
        st.info("👆 Upload all 4 CSVs to unlock Auction Mode.")
        st.stop()

    players   = pd.read_csv(players_f)
    perf      = pd.read_csv(performance_f)
    contracts = pd.read_csv(contracts_f)
    budget_row= pd.read_csv(budget_f).iloc[0]

    req_contract = {"player_id","current_salary_lakh"}
    if not req_contract.issubset(contracts.columns):
        st.error(f"contracts.csv missing: {sorted(req_contract - set(contracts.columns))}")
        st.stop()

    contracts["current_salary_lakh"] = pd.to_numeric(contracts["current_salary_lakh"], errors="coerce").fillna(0.0)
    df = build_base_df(players, perf, contracts=contracts)
    df["current_salary_lakh"] = pd.to_numeric(df.get("current_salary_lakh", 0), errors="coerce").fillna(0.0)

    # ── CONTEXT ───────────────────────────────────────────────────────────
    cric_divider()
    section("Match Context", "🏟️")
    cc1, cc2, cc3, cc4 = st.columns(4)
    pitch_type  = cc1.selectbox("Home pitch", ["Flat/True","Spin-friendly","Pace/Bounce"])
    season_goal = cc2.selectbox("Season strategy", ["Balanced","Batting dominance","Bowling dominance"])
    risk_pref   = cc3.selectbox("Risk preference", ["Balanced","Risk-averse","High-upside"])
    auction_mode= cc4.checkbox("Auction pricing", value=True)

    # ── OPPONENT ──────────────────────────────────────────────────────────
    cric_divider()
    section("Opponent Profile", "🎯")
    opp_profiles = {
        "Balanced":               {"spin":0.5,"pace":0.5,"death":0.5,"pp":0.5},
        "Spin choke team":        {"spin":0.85,"pace":0.35,"death":0.55,"pp":0.45},
        "Pace/bounce attack":     {"spin":0.35,"pace":0.85,"death":0.65,"pp":0.55},
        "Death-over specialists": {"spin":0.45,"pace":0.55,"death":0.90,"pp":0.45},
        "Powerplay smashers":     {"spin":0.45,"pace":0.55,"death":0.55,"pp":0.90},
    }
    oc1, oc2 = st.columns([2,1])
    opp_sel  = oc1.selectbox("Opponent type", list(opp_profiles.keys()))
    override = oc2.checkbox("Manual sliders", value=False)

    base_opp = opp_profiles[opp_sel]
    o1,o2,o3,o4 = st.columns(4)
    opp_spin  = o1.slider("Spin threat",       0.0,1.0,float(base_opp["spin"]),  0.05, disabled=not override)
    opp_pace  = o2.slider("Pace threat",        0.0,1.0,float(base_opp["pace"]),  0.05, disabled=not override)
    opp_death = o3.slider("Death bowl strength",0.0,1.0,float(base_opp["death"]), 0.05, disabled=not override)
    opp_pp    = o4.slider("PP aggressiveness",  0.0,1.0,float(base_opp["pp"]),    0.05, disabled=not override)

    # ── AUCTION SETTINGS ──────────────────────────────────────────────────
    cric_divider()
    section("Auction Settings", "💰")
    a1, a2, a3 = st.columns(3)
    inflation     = a1.slider("Inflation multiplier", 1.0, 2.5, 1.35, 0.05, disabled=not auction_mode)
    reserve_floor = a2.number_input("Reserve floor (₹ lakh)", value=120.0, step=10.0, disabled=not auction_mode)
    budget_lakh   = a3.number_input("Budget (₹ lakh)", value=float(budget_row.get("budget_lakh", 10000)), step=100.0)

    # ── SQUAD CONSTRAINTS ─────────────────────────────────────────────────
    cric_divider()
    section("Squad Constraints", "⚙️")
    c5,c6,c7,c8 = st.columns(4)
    max_players = c5.number_input("Max squad", value=int(budget_row.get("max_players",25)), step=1)
    min_bat     = c6.number_input("Min BAT",   value=int(budget_row.get("min_bat",8)),      step=1)
    min_bowl    = c7.number_input("Min BOWL",  value=int(budget_row.get("min_bowl",8)),     step=1)
    min_ar      = c8.number_input("Min AR",    value=int(budget_row.get("min_ar",4)),       step=1)

    c9,c10 = st.columns(2)
    min_wk            = c9.number_input("Min WK",           value=int(budget_row.get("min_wk",2)), step=1)
    max_overseas_squad = c10.number_input("Max Overseas",   value=8, step=1)
    min_role = {"BAT":min_bat,"BOWL":min_bowl,"AR":min_ar,"WK":min_wk}

    # ── BALANCE CONSTRAINTS ───────────────────────────────────────────────
    cric_divider()
    section("Balance Constraints", "⚖️")
    b1,b2,b3,b4 = st.columns(4)
    min_spinners  = b1.number_input("Min Spinners",      value=3, step=1)
    min_pacers    = b2.number_input("Min Pacers",        value=3, step=1)
    min_death_bowl= b3.number_input("Min Death Bowlers", value=2, step=1)
    min_death_hit = b4.number_input("Min Death Hitters", value=2, step=1)

    b5,b6,b7 = st.columns(3)
    min_pp_bowl  = b5.number_input("Min PP Bowlers",  value=2, step=1)
    min_openers  = b6.number_input("Min Openers",     value=2, step=1)
    min_finishers= b7.number_input("Min Finishers",   value=2, step=1)

    enforce_left = st.checkbox("Enforce left-right balance (≥1 lefty in top 4)", value=True)

    # ── SOFT CONSTRAINTS ──────────────────────────────────────────────────
    cric_divider()
    section("Solver Settings", "🔧")
    s1, s2 = st.columns(2)
    soft_mode      = s1.checkbox("Soft constraints (always returns a squad)", value=True)
    penalty_weight = s2.slider("Soft penalty (higher = stricter)", 0.5, 10.0, 2.5, 0.1, disabled=not soft_mode)

    # ── RETENTIONS ────────────────────────────────────────────────────────
    cric_divider()
    section("Retentions / RTM", "🔒")
    r1, r2 = st.columns([2,1])
    retained_players  = r1.multiselect("Retained/locked players", df["player"].tolist(), default=[])
    default_ret_cost  = r2.number_input("Default retained cost (₹ lakh)", value=600.0, step=50.0)

    retained_costs = {}
    if retained_players:
        st.write("Set retained cost per player:")
        rc_cols = st.columns(min(len(retained_players), 4))
        for i, p in enumerate(retained_players[:20]):
            retained_costs[p] = rc_cols[i % 4].number_input(p, value=float(default_ret_cost), step=50.0, key=f"rc_{p}")

    locked_set      = set(retained_players)
    retained_total  = sum(retained_costs.get(p, default_ret_cost) for p in retained_players)
    budget_after_ret= float(budget_lakh - retained_total)

    st.metric("Budget after retentions", f"₹ {budget_after_ret:,.1f} lakh",
              delta=f"-₹{retained_total:,.1f} lakh retained" if retained_total > 0 else None)

    if budget_after_ret <= 0:
        st.error("❌ Retentions exceed budget.")
        st.stop()

    # ── PRICING ───────────────────────────────────────────────────────────
    df["auction_price_lakh"] = df["current_salary_lakh"].copy()
    if auction_mode:
        df["auction_price_lakh"] = (df["current_salary_lakh"] * float(inflation)).clip(lower=float(reserve_floor))

    price_col = "auction_price_lakh" if auction_mode else "current_salary_lakh"
    df["price_used_lakh"] = df[price_col].copy()
    if retained_players:
        df["price_used_lakh"] = df["price_used_lakh"].astype(float)
        for p in retained_players:
            df.loc[df["player"]==p, "price_used_lakh"] = float(retained_costs.get(p, default_ret_cost))
    price_col = "price_used_lakh"

    # ── RISK ADJUSTMENT ───────────────────────────────────────────────────
    rp_map = {"Risk-averse":1.2, "High-upside":0.4, "Balanced":0.8}
    df["risk_adjustment"] = (1 - rp_map[risk_pref] * df["total_risk"]).clip(lower=0.25, upper=1.05)

    # ── PITCH FIT ─────────────────────────────────────────────────────────
    fit = np.ones(len(df), dtype=float)
    if pitch_type == "Spin-friendly":
        fit += 0.12*df["is_spinner"].astype(float) + 0.05*(df["batting_role"]=="ANCHOR").astype(float)
        fit -= 0.04*df["is_pacer"].astype(float)
    elif pitch_type == "Pace/Bounce":
        fit += 0.12*df["is_pacer"].astype(float) + 0.05*norm01(df["pp_sr"])
        fit -= 0.04*df["is_spinner"].astype(float)
    else:
        fit += 0.10*norm01(df["strike_rate"]) + 0.08*(df["batting_role"]=="FINISHER").astype(float)
    df["pitch_fit"] = np.clip(fit, 0.80, 1.30)

    # ── PHASE WEIGHTS ─────────────────────────────────────────────────────
    cric_divider()
    section("Phase Importance", "⏱️")
    p1,p2,p3 = st.columns(3)
    w_pp    = p1.slider("Powerplay weight",    0.0, 2.0, 1.0, 0.1)
    w_mid   = p2.slider("Middle overs weight", 0.0, 2.0, 1.0, 0.1)
    w_death = p3.slider("Death overs weight",  0.0, 2.0, 1.2, 0.1)

    df = compute_phase_scores(df, w_pp, w_mid, w_death)

    # ── OPPONENT FIT ──────────────────────────────────────────────────────
    lefty      = (df["bat_hand"].astype(str)=="L").astype(float)
    anchor     = (df["batting_role"]=="ANCHOR").astype(float)
    opener     = (df["batting_role"]=="OPENER").astype(float)
    pp_bowler  = (df["bowling_role"]=="PP").astype(float)
    pacer      = df["is_pacer"].astype(float)
    spinner    = df["is_spinner"].astype(float)

    opp_mult = np.ones(len(df), dtype=float)
    opp_mult += 0.10*opp_spin  * (0.9*lefty  + 0.6*anchor  + 0.6*spinner  + 0.2*df["mid_bat_score"])
    opp_mult += 0.10*opp_pace  * (0.9*pacer  + 0.6*opener  + 0.5*pp_bowler + 0.2*df["pp_bat_score"])
    opp_mult += 0.08*opp_death * (0.9*anchor + 0.4*df["mid_bat_score"])
    opp_mult += 0.08*opp_pp    * (0.9*pp_bowler + 0.5*(1-norm01(df["pp_eco"])))
    df["opponent_fit"] = np.clip(opp_mult, 0.85, 1.22)

    # ── FAIR SALARY REGRESSION ────────────────────────────────────────────
    cric_divider()
    section("Fair Salary Regression", "💎")
    feat_pool = ["role","age","runs","wickets","matches","strike_rate","economy",
                 "dot_ball_pct","boundary_pct","match_impact_score"]
    X = df[[c for c in feat_pool if c in df.columns]].copy()
    y = df["current_salary_lakh"].copy()

    cat_cols = ["role"] if "role" in X.columns else []
    ct    = ColumnTransformer([("ohe", OneHotEncoder(handle_unknown="ignore"), cat_cols)], remainder="passthrough")
    model = Pipeline([("prep",ct),("reg",Ridge(alpha=1.0))])
    model.fit(X, y)
    df["fair_salary_lakh"] = np.clip(model.predict(X), 0, None)
    df["value_gap"] = df["fair_salary_lakh"] - df["price_used_lakh"]

    # ── FLEX SCORE ────────────────────────────────────────────────────────
    df["phase_versatility"] = (
        0.5*((df["is_pp_batter"].astype(int)==1)|(df["is_pp_bowler"].astype(int)==1)).astype(float) +
        0.5*((df["is_death_hitter"].astype(int)==1)|(df["is_death_bowler"].astype(int)==1)).astype(float)
    )
    df["flex_score"] = norm01(0.55*(df["role"].isin(["AR","WK"]).astype(float)) + 0.45*norm01(df["phase_versatility"]))

    # ── MULTI-OBJECTIVE WEIGHTS ───────────────────────────────────────────
    cric_divider()
    section("Auction Strategy & Objective Weights", "🎚️")
    s1,s2,s3 = st.columns(3)
    strategy              = s1.selectbox("Squad style", ["Balanced","Star-heavy","Depth-heavy"])
    max_single_pct        = s2.slider("Max single-player spend (%)", 10, 60, 30, 1)
    price_conc_penalty    = s3.slider("Price concentration penalty", 0.0, 5.0, 1.0, 0.1)
    max_single_price      = float(budget_after_ret) * (float(max_single_pct)/100.0)

    base_weights = {"value":1.0,"impact":1.2,"fit":1.0,"opp":1.0,"flex":0.8,"risk":1.0}
    if strategy == "Star-heavy":
        base_weights.update({"value":0.8,"impact":1.35,"fit":1.15,"opp":1.15})
    elif strategy == "Depth-heavy":
        base_weights.update({"value":1.25,"impact":1.10})
        price_conc_penalty = max(price_conc_penalty, 1.5)

    mw1,mw2,mw3,mw4,mw5,mw6 = st.columns(6)
    w_value  = mw1.slider("Value gap",      0.0,2.0,base_weights["value"],  0.05)
    w_impact = mw2.slider("Impact",         0.0,2.0,base_weights["impact"], 0.05)
    w_fit    = mw3.slider("Pitch fit",      0.0,2.0,base_weights["fit"],    0.05)
    w_opp    = mw4.slider("Opponent fit",   0.0,2.0,base_weights["opp"],    0.05)
    w_flex   = mw5.slider("Flex",           0.0,2.0,base_weights["flex"],   0.05)
    w_risk   = mw6.slider("Risk penalty",   0.0,2.0,base_weights["risk"],   0.05)

    # objective
    df["value_gap_norm"] = norm01(df["value_gap"])
    df["impact_norm"]    = norm01(df["match_impact_score"])
    df["fit_norm"]       = norm01(df["pitch_fit"])
    df["opp_norm"]       = norm01(df["opponent_fit"])
    df["risk_norm"]      = norm01(df["total_risk"])

    df["objective_score_raw"] = (
        w_value*df["value_gap_norm"] + w_impact*df["impact_norm"] +
        w_fit*df["fit_norm"] + w_opp*df["opp_norm"] +
        w_flex*df["flex_score"] - w_risk*df["risk_norm"]
    )
    df["objective_score"] = df["objective_score_raw"] * df["risk_adjustment"]
    df["xi_score"] = (
        0.28*df["impact_norm"] + 0.20*df["fit_norm"] + 0.22*df["opp_norm"] +
        0.15*df["flex_score"]  + 0.15*df["value_gap_norm"]
    ) * df["risk_adjustment"]

    # ── TOP PLAYER TABLE ──────────────────────────────────────────────────
    cric_divider()
    section("Top Player Rankings", "📊")
    top_show = ["player","role","age","bat_hand","batting_role","bowling_role",
                "is_spinner","is_pacer","is_overseas",
                "price_used_lakh","fair_salary_lakh","value_gap",
                "match_impact_score","pitch_fit","opponent_fit",
                "flex_score","total_risk","objective_score"]
    top_show = [c for c in top_show if c in df.columns]
    st.dataframe(
        df[top_show].sort_values("objective_score", ascending=False).head(50).style.format(
            {"objective_score":"{:.3f}","value_gap":"{:.1f}",
             "match_impact_score":"{:.3f}","total_risk":"{:.3f}"}),
        use_container_width=True, height=420
    )

    # ── OPTIMISE ──────────────────────────────────────────────────────────
    cric_divider()
    section("Optimised Squad Selection", "🤖")

    extra_min_flags = {
        "is_spinner":       int(min_spinners),
        "is_pacer":         int(min_pacers),
        "is_death_bowler2": int(min_death_bowl),
        "is_death_hitter":  int(min_death_hit),
        "is_pp_bowler2":    int(min_pp_bowl),
        "is_opener":        int(min_openers),
        "is_finisher":      int(min_finishers),
    }
    extra_max_flags = {"is_overseas": int(max_overseas_squad)}

    with st.spinner("Running squad optimiser..."):
        squad, sm = optimize_squad_soft(
            df=df, budget_limit=float(budget_after_ret),
            max_players=int(max_players), min_role=min_role,
            price_col=price_col, extra_min_flags=extra_min_flags,
            extra_max_flags=extra_max_flags,
            lock_players=locked_set if locked_set else None,
            max_single_price=max_single_price,
            penalty_weight=float(penalty_weight),
            price_concentration_penalty=float(price_conc_penalty),
        )

    s1,s2,s3,s4,s5,s6 = st.columns(6)
    s1.metric("Players selected",  sm.get("count",0))
    s2.metric("Spend (₹ lakh)",    f"{sm.get('spend',0):,.1f}")
    s3.metric("Objective total",   f"{sm.get('objective',0):.2f}")
    s4.metric("BAT",               sm.get("BAT",0))
    s5.metric("BOWL",              sm.get("BOWL",0))
    s6.metric("AR+WK",             (sm.get("AR",0)+sm.get("WK",0)))

    if sm.get("violations"):
        v = {k:round(v,2) for k,v in sm["violations"].items() if v and v>0.001}
        if v:
            st.warning(f"⚠️ Soft-constraint shortfalls: {v}")

    if len(squad) == 0:
        st.error("❌ No squad returned. Check budget/constraints.")
        st.stop()

    squad_show = ["player","role","bat_hand","batting_role","bowling_role",
                  "is_spinner","is_pacer","is_overseas",
                  "price_used_lakh","fair_salary_lakh","value_gap",
                  "match_impact_score","pitch_fit","opponent_fit",
                  "flex_score","total_risk","objective_score"]
    squad_show = [c for c in squad_show if c in squad.columns]
    st.dataframe(
        squad[squad_show].sort_values("objective_score", ascending=False).style.format(
            {"objective_score":"{:.3f}","value_gap":"{:.1f}","total_risk":"{:.3f}"}),
        use_container_width=True, height=420
    )

    # ── BEST XI ───────────────────────────────────────────────────────────
    cric_divider()
    section("Best Playing XI", "🏏")
    x1,x2,x3,x4 = st.columns(4)
    xi_size     = x1.number_input("XI size",    value=11, step=1)
    xi_min_bat  = x2.number_input("XI Min BAT", value=4,  step=1)
    xi_min_bowl = x3.number_input("XI Min BOWL",value=4,  step=1)
    xi_min_wk   = x4.number_input("XI Min WK",  value=1,  step=1)
    x5,x6 = st.columns(2)
    xi_min_ar       = x5.number_input("XI Min AR",       value=1, step=1)
    max_overseas_xi = x6.number_input("XI Max Overseas", value=4, step=1)

    xi_min_role = {"BAT":xi_min_bat,"BOWL":xi_min_bowl,"AR":xi_min_ar,"WK":xi_min_wk}

    with st.spinner("Picking best XI..."):
        xi, xm = pick_best_xi(squad=squad, xi_size=int(xi_size), xi_min_role=xi_min_role,
                              max_overseas_xi=int(max_overseas_xi), enforce_left_in_top4=enforce_left)

    if len(xi) == 0:
        st.warning("⚠️ No feasible XI — relax XI constraints.")
    else:
        st.metric("XI total score", f"{xm.get('xi_score',0):.3f}")
        xi_show = ["player","role","bat_hand","batting_role","bowling_role",
                   "is_spinner","is_pacer","is_overseas",
                   "match_impact_score","pitch_fit","opponent_fit","flex_score","total_risk","xi_score"]
        xi_show = [c for c in xi_show if c in xi.columns]
        st.dataframe(
            xi[xi_show].sort_values("xi_score", ascending=False).style.format(
                {"xi_score":"{:.3f}","match_impact_score":"{:.3f}","total_risk":"{:.3f}"}),
            use_container_width=True, height=420
        )

    # ── DOWNLOADS ─────────────────────────────────────────────────────────
    cric_divider()
    d1,d2,d3 = st.columns(3)
    d1.download_button("⬇ Full Table",  to_csv_bytes(df),    "auction_full.csv",  "text/csv")
    d2.download_button("⬇ Squad",       to_csv_bytes(squad), "auction_squad.csv", "text/csv")
    d3.download_button("⬇ Best XI",     to_csv_bytes(xi) if len(xi) else b"", "auction_xi.csv", "text/csv")


# ─────────────────────────────────────────────
# HIGHLIGHTS GENERATOR
# ─────────────────────────────────────────────
def run_highlights_mode():
    section("Upload Source Video", "🎬")

    if "hl_video_path" not in st.session_state:
        st.session_state["hl_video_path"] = None

    tab_file, tab_yt = st.tabs(["📂 Upload Video", "▶️ YouTube Link"])

    with tab_file:
        vid = st.file_uploader("Upload video (mp4 / mov / mkv)", type=["mp4","mov","mkv"], key="hl_upload")
        if vid:
            tmp_dir    = tempfile.mkdtemp()
            video_path = os.path.join(tmp_dir, vid.name)
            with open(video_path,"wb") as f:
                f.write(vid.getbuffer())
            st.session_state["hl_video_path"] = video_path
            st.success("✅ Video uploaded")
            st.video(video_path)

    with tab_yt:
        yt_url = st.text_input("Paste YouTube URL", placeholder="https://www.youtube.com/watch?v=...", key="hl_yt")
        st.caption("Requires `yt-dlp` + `ffmpeg` on the server.")

        if yt_url and st.button("Fetch from YouTube", key="hl_fetch"):
            tmp_dir = tempfile.mkdtemp()
            out_tpl = os.path.join(tmp_dir, "input.%(ext)s")
            try:
                subprocess.check_call(["yt-dlp","-f","bv*+ba/b","-o",out_tpl,yt_url])
                video_path = next((os.path.join(tmp_dir,f) for f in os.listdir(tmp_dir) if f.startswith("input.")), None)
                if video_path and os.path.exists(video_path):
                    st.session_state["hl_video_path"] = video_path
                    st.success("✅ YouTube video downloaded")
                    st.video(video_path)
                else:
                    st.error("Download finished but file not found.")
            except Exception as e:
                st.error(f"Download failed: {e}")

    cric_divider()
    section("Generate Highlights Clip", "✂️")

    c1, c2, c3 = st.columns(3)
    start_sec  = c1.number_input("Start time (seconds)", min_value=0, value=0, step=5, key="hl_start")
    clip_secs  = c2.number_input("Clip length (seconds)", min_value=5, max_value=600, value=60, step=5, key="hl_len")
    out_name   = c3.text_input("Output filename", value="highlights.mp4", key="hl_out")

    if st.button("⚡ Generate Highlights", type="primary", key="hl_process"):
        video_path = st.session_state.get("hl_video_path")
        if not video_path or not os.path.exists(video_path):
            st.error("Upload a video first.")
            return

        tmp_out = tempfile.mkdtemp()
        out_path= os.path.join(tmp_out, out_name)

        cmd = ["ffmpeg","-y","-i",video_path,
               "-ss",str(int(start_sec)),"-t",str(int(clip_secs)),
               "-c","copy", out_path]
        try:
            with st.spinner("Processing..."):
                subprocess.check_call(cmd)
            st.success("✅ Highlights generated")
            st.video(out_path)
            with open(out_path,"rb") as f:
                st.download_button("⬇ Download Highlights (MP4)", data=f.read(),
                                   file_name=out_name, mime="video/mp4", key="hl_dl")
        except Exception as e:
            st.error(f"❌ ffmpeg failed: {e}")
            st.info("Ensure ffmpeg is installed and in PATH.")


# ─────────────────────────────────────────────
# SIDEBAR + MAIN HEADER
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 1.2rem 0 0.5rem;'>
        <div style='font-size:2rem;'>🏏</div>
        <div style='font-size:1.4rem; font-weight:800; color:#00d4ff; letter-spacing:0.12em;'>CRICINTEL</div>
        <div style='font-size:0.75rem; color:#4a7a9b; margin-top:0.3rem;'>AI Cricket Analytics Platform</div>
    </div>
    <hr style='border-color:#1e3a5f; margin:1rem 0;'>
    """, unsafe_allow_html=True)

    mode = st.radio(
        "Select Mode",
        ["🔍 Scout Mode", "💰 Auction Mode", "🎬 Highlights Generator"],
        index=0
    )

    st.markdown("""
    <hr style='border-color:#1e3a5f; margin:1rem 0;'>
    <div style='font-size:0.75rem; color:#4a7a9b; text-align:center; padding-bottom:1rem;'>
        v2.0 &nbsp;|&nbsp; Built with Streamlit<br>
        <span style='color:#1e3a5f;'>─────────────────</span><br>
        Scout Mode: 2 CSVs<br>
        Auction Mode: 4 CSVs<br>
        Highlights: MP4 / YouTube
    </div>
    """, unsafe_allow_html=True)

# ── BANNER ────────────────────────────────────────────────────────────────────
mode_labels = {
    "🔍 Scout Mode":          ("🔍 Scout Mode",          "Talent identification · Similarity search · Gap-fill recommendations"),
    "💰 Auction Mode":        ("💰 Auction Mode",         "Squad optimisation · Fair salary · Best XI selection"),
    "🎬 Highlights Generator":("🎬 Highlights Generator", "Upload match video · Generate highlight clips"),
}
banner_title, banner_sub = mode_labels[mode]

st.markdown(f"""
<div class="cricintel-banner">
    <div>
        <h1>CRICINTEL</h1>
        <p>{banner_sub}</p>
    </div>
    <div style='text-align:right;'>
        <div style='font-size:1.1rem; font-weight:700; color:#00d4ff;'>{banner_title}</div>
        <div style='font-size:0.78rem; color:#4a7a9b; margin-top:0.3rem;'>AI Cricket Analytics Platform</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ── ROUTE ─────────────────────────────────────────────────────────────────────
if mode == "🔍 Scout Mode":
    run_scout_mode()
elif mode == "💰 Auction Mode":
    run_auction_mode()
else:
    run_highlights_mode()

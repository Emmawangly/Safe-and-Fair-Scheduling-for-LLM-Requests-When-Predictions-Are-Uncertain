"""
LLM Scheduler Simulation Dashboard
Run with: streamlit run dashboard.py

TO USE REAL DATA: replace load_data() with pd.read_csv() calls.
Each CSV needs: error_pct, jct, ttft, preemptions, jain_fairness, starvation_pct

After replacing CSVs, click ⋮ menu → 'Clear cache' so Streamlit reloads the files.
"""

import hashlib
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

st.set_page_config(
    page_title="LLM Scheduler Simulation",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
.block-container { padding-top: 2.8rem !important; padding-bottom: 1rem !important; }
h1 { font-size: 22px !important; font-weight: 800 !important; color: #1a1a2e !important; padding-bottom: 0.4rem !important; }
.kpi-card {
    background: #ffffff;
    border: 1px solid #e0e0e0;
    border-radius: 8px;
    padding: 12px 14px;
    margin-bottom: 10px;
}
.kpi-title { font-size: 12px; font-weight: 700; color: #444; margin-bottom: 2px; }
.kpi-hint  { font-size: 9px; color: #999; font-style: italic; margin-bottom: 8px; }
.kpi-row   { display: grid; grid-template-columns: repeat(3,1fr); gap: 6px; }
.mc        { text-align: center; padding: 6px 4px; border-radius: 6px; }
.mc-label  { font-size: 9px; font-weight: 700; margin-bottom: 2px; }
.mc-val    { font-size: 16px; font-weight: 700; }
.mc-sub    { font-size: 9px; margin-top: 2px; }
.chart-title { font-size: 12px; font-weight: 700; color: #1a1a2e; margin-bottom: 2px; }
.chart-sub   { font-size: 10px; color: #999; font-style: italic; margin-bottom: 4px; }
.legend-row  { display: flex; gap: 12px; flex-wrap: wrap; margin-bottom: 4px; }
.leg-item    { display: flex; align-items: center; gap: 5px; font-size: 10px; color: #666; }
.leg-dot     { width: 8px; height: 8px; border-radius: 50%; display: inline-block; }
[data-testid="stVerticalBlockBorderWrapper"] {
    border-radius: 8px !important;
    border-color: #e5e5e5 !important;
}
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────────
C_FCFS   = "#E24B4A";  C_LTR  = "#378ADD";  C_ROB  = "#1D9E75"
C_FCFS_A = "rgba(226,75,74,0.82)"
C_LTR_A  = "rgba(55,138,221,0.82)"
C_ROB_A  = "rgba(29,158,117,0.82)"

ERROR_LEVELS = [0, 20, 40, 60, 80]
ALPHA_VALS   = [0.1, 0.5, 1.0, 2.0]
BETA_VALS    = [0.1, 0.5, 1.0, 2.0]
REQ_SCALE    = {500: 0.55, 1000: 1.0, 2000: 1.85}

# ── Data ──────────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    """
    Replace with real CSV loading:
        return {
            "fcfs":   pd.read_csv("results_fcfs.csv"),
            "ltr":    pd.read_csv("results_ltr.csv"),
            "robust": pd.read_csv("results_robust.csv"),
        }
    Columns needed: error_pct, jct, ttft, preemptions, jain_fairness, starvation_pct
    """
    # ── Synthetic data notes ──────────────────────────────────────────────────
    # 1. FCFS degrades fastest with error (no prediction benefit).
    # 2. LTR is better than FCFS at low error but converges toward FCFS at high
    #    error because bad predictions break the size-ordering assumption.
    #    Degradation is steeper from 40%+ as the paper (Saravana Kumar et al.)
    #    shows OOD failure kicks in around that range.
    # 3. Robust at 0% error == LTR (σᵢ=0 → α·σᵢ term vanishes → scheduler
    #    reduces to LTR+aging; aging adds fairness overhead, not latency benefit,
    #    so JCT matches LTR rather than beating it).
    # 4. LTR starvation_pct > 0 even at 0% error because pure SJF always
    #    disadvantages long requests — aging in Robust prevents this entirely.
    # FCFS baseline: at 0% error FCFS/LTR ≈ 2.1x (matches Saravana Kumar et al.)
    # Head-of-Line Blocking causes long requests to delay many short ones even
    # with perfect predictions — this is the core motivation for LTR scheduling.
    fcfs = [
        {"error_pct":0,  "jct":2.10,"ttft":0.42,"preemptions":8,   "jain_fairness":0.80,"starvation_pct":4.0},
        {"error_pct":20, "jct":2.90,"ttft":0.55,"preemptions":25,  "jain_fairness":0.73,"starvation_pct":10.0},
        {"error_pct":40, "jct":4.30,"ttft":0.82,"preemptions":72,  "jain_fairness":0.60,"starvation_pct":20.0},
        {"error_pct":60, "jct":6.20,"ttft":1.20,"preemptions":128, "jain_fairness":0.50,"starvation_pct":34.0},
        {"error_pct":80, "jct":8.80,"ttft":1.75,"preemptions":230, "jain_fairness":0.38,"starvation_pct":54.0},
    ]
    # LTR degrades more steeply after 40% error (OOD failure)
    ltr = [
        {"error_pct":0,  "jct":1.00,"ttft":0.20,"preemptions":3,   "jain_fairness":0.97,"starvation_pct":1.0},
        {"error_pct":20, "jct":1.55,"ttft":0.29,"preemptions":11,  "jain_fairness":0.93,"starvation_pct":4.5},
        {"error_pct":40, "jct":2.70,"ttft":0.48,"preemptions":43,  "jain_fairness":0.80,"starvation_pct":14.0},
        {"error_pct":60, "jct":4.20,"ttft":0.76,"preemptions":85,  "jain_fairness":0.68,"starvation_pct":30.0},
        {"error_pct":80, "jct":6.50,"ttft":1.22,"preemptions":162, "jain_fairness":0.55,"starvation_pct":49.0},
    ]
    # Robust at 0% error = LTR (α·σ=0 → same ordering, aging adds no JCT benefit)
    # Robust degrades much more slowly than LTR at high error (uncertainty defense)
    rob = [
        {"error_pct":0,  "jct":1.00,"ttft":0.20,"preemptions":3,   "jain_fairness":0.99,"starvation_pct":0.0},
        {"error_pct":20, "jct":1.30,"ttft":0.25,"preemptions":8,   "jain_fairness":0.97,"starvation_pct":0.5},
        {"error_pct":40, "jct":2.10,"ttft":0.40,"preemptions":36,  "jain_fairness":0.91,"starvation_pct":2.5},
        {"error_pct":60, "jct":2.80,"ttft":0.56,"preemptions":68,  "jain_fairness":0.85,"starvation_pct":5.0},
        {"error_pct":80, "jct":3.90,"ttft":0.85,"preemptions":130, "jain_fairness":0.78,"starvation_pct":9.0},
    ]
    return {
        "fcfs":   pd.DataFrame(fcfs),
        "ltr":    pd.DataFrame(ltr),
        "robust": pd.DataFrame(rob),
    }

def get_row(df, err):
    rows = df[df["error_pct"] == err]
    if rows.empty:
        available = sorted(df["error_pct"].unique().tolist())
        st.error(f"No data for error_pct={err}. Available: {available}. Check your CSV.")
        st.stop()
    return rows.iloc[0]

# ── Parameter modifiers (synthetic only) ─────────────────────────────────────
# Based on Sᵢ = 1/(μᵢ + α·σᵢ) + β·(Twait,i / T̄wait)
# ⚠️  Remove these functions and use CSV values directly with real data.
# ⚠️  Remove cap= from rob_jct / rob_ttft calls when switching to real CSVs.

def apply_params(base_val, alpha, beta, cap=None):
    """
    JCT / TTFT modifier.
    α high → conservative → JCT increases (over-penalises uncertain predictions).
    β high → stronger aging → slight JCT increase (fairness overhead).
    Factors calibrated so differences are clearly visible in the dashboard:
      α: 0.1→−7.2%, 0.5→0% (ref), 1.0→+9.0%, 2.0→+27.0%
      β: 0.1→−3.6%, 0.5→0% (ref), 1.0→+4.5%, 2.0→+13.5%
    cap= prevents Robust exceeding LTR in synthetic data (remove for real data).
    """
    a_factor = 1.0 + (alpha - 0.5) * 0.18   # larger range → more visible
    b_factor = 1.0 + (beta  - 0.5) * 0.09   # increased from 0.03 → fairness tradeoff visible
    result = round(max(0.05, base_val * a_factor * b_factor), 2)
    if cap is not None:
        result = min(result, round(cap, 2))
    return result

def apply_params_pre(base_pre, alpha, beta):
    """
    Preemption modifier.
    α high → uncertainty penalty → scheduler avoids risky allocations → fewer preemptions.
    β high → aging balances queue → fewer head-of-line blocking events → fewer preemptions.
    α effect is stronger (direct mechanism). β effect is moderate (indirect).
    """
    a_reduction = (alpha - 0.5) * 0.22   # 0.1→+8.8%, 0.5→0%, 1.0→−11%, 2.0→−33%
    b_reduction = (beta  - 0.5) * 0.10   # 0.1→+4.0%, 0.5→0%, 1.0→−5%,  2.0→−15%
    factor = 1.0 - a_reduction - b_reduction
    return max(0, int(base_pre * factor))

def apply_params_jain(base_jain, alpha, beta):
    """
    Jain's Fairness modifier.
    β is the primary driver: stronger aging directly improves fairness.
    α has a small positive effect: fewer preemptions = more predictable scheduling.
    β: 0.1→−7.2%, 0.5→0% (ref), 1.0→+9.0%, 2.0→+27.0% (capped at 1.0)
    α: 0.1→−1.6%, 0.5→0% (ref), 1.0→+2.0%, 2.0→+6.0%
    """
    b_factor = 1.0 + (beta  - 0.5) * 0.18   # dominant effect
    a_factor = 1.0 + (alpha - 0.5) * 0.04   # minor effect
    return min(1.0, round(base_jain * b_factor * a_factor, 2))

def apply_params_starv(base_s, alpha, beta):
    """
    Starvation modifier.
    β is the primary driver: aging directly prevents indefinite waiting.
    α contributes moderately: fewer preemptions = more stable queue.
    """
    b_reduction = (beta  - 0.5) * 0.18   # strong effect
    a_reduction = (alpha - 0.5) * 0.10   # moderate effect
    factor = 1.0 - b_reduction - a_reduction
    return max(0.0, round(base_s * factor, 1))

# ── Chart helpers ─────────────────────────────────────────────────────────────
CHART_H = 200

def fmt_bar(v, y_suffix):
    """2 decimals for seconds, 1 decimal for percent — consistent within each chart."""
    return f"{v:.1f}%" if y_suffix == "%" else f"{v:.2f}s"

def bar_chart(labels, values, colors, y_suffix="s"):
    text = [fmt_bar(v, y_suffix) for v in values]
    fig = go.Figure(go.Bar(
        x=labels, y=values, marker_color=colors,
        text=text, textposition="outside",
        textfont=dict(size=10, color="#333", family="Arial"),
        width=0.45,
    ))
    fig.update_layout(
        paper_bgcolor="white", plot_bgcolor="white",
        margin=dict(l=40, r=10, t=20, b=30),
        font=dict(family="Arial", size=10, color="#999"),
        showlegend=False, height=CHART_H,
        yaxis=dict(
            gridcolor="rgba(0,0,0,0.06)", tickfont=dict(size=9, color="#bbb"),
            ticksuffix=y_suffix, rangemode="tozero",
            range=[0, max(max(values), 0.01) * 1.28],  # guard: avoids y-axis collapse if all values are 0
        ),
        xaxis=dict(showgrid=False, tickfont=dict(size=9, color="#bbb")),
    )
    return fig

def line_chart(x_labels, series_vals, series_names, series_colors, fills, y_suffix="s"):
    fig = go.Figure()
    for name, vals, color, fill in zip(series_names, series_vals, series_colors, fills):
        fig.add_trace(go.Scatter(
            x=x_labels, y=vals, name=name, mode="lines+markers",
            line=dict(color=color, width=2),
            marker=dict(size=4, color=color),
            fill="tozeroy", fillcolor=fill,
        ))
    fig.update_layout(
        paper_bgcolor="white", plot_bgcolor="white",
        margin=dict(l=40, r=10, t=20, b=30),
        font=dict(family="Arial", size=10, color="#999"),
        showlegend=False, height=CHART_H,
        yaxis=dict(
            gridcolor="rgba(0,0,0,0.06)", tickfont=dict(size=9, color="#bbb"),
            ticksuffix=y_suffix, rangemode="tozero",
        ),
        xaxis=dict(gridcolor="rgba(0,0,0,0.06)", tickfont=dict(size=9, color="#bbb")),
    )
    return fig

def heatmap_chart(base_jct, alpha, beta):
    """
    U-shaped α/β sensitivity surface. Minimum at α=0.5, β=0.5 (reference config).
    Penalties grow quadratically as α or β deviate from the optimum in either direction.
    The cell at α=0.5, β=0.5 always equals base_jct exactly (no noise at reference point)
    so it matches the KPI card value.
    With real data: replace entire function body with pivot of your sweep CSV.
    """
    ALPHA_OPT = 0.5
    BETA_OPT  = 0.5
    z_vals = []
    for a in ALPHA_VALS:
        row = []
        for b in BETA_VALS:
            a_penalty = ((a - ALPHA_OPT) ** 2) * 0.10
            b_penalty = ((b - BETA_OPT)  ** 2) * 0.05
            # No noise at reference point so heatmap matches KPI exactly
            if a == ALPHA_OPT and b == BETA_OPT:
                noise = 0.0
            else:
                noise = (int(hashlib.md5(
                    f"{a}{b}{base_jct:.2f}".encode()).hexdigest(), 16) % 80) / 10000
            row.append(round(base_jct * (1 + a_penalty + b_penalty) + noise, 2))
        z_vals.append(row)

    z_arr   = np.array(z_vals)
    min_idx = np.unravel_index(z_arr.argmin(), z_arr.shape)
    sel_ai  = ALPHA_VALS.index(alpha)
    sel_bi  = BETA_VALS.index(beta)

    colorscale = [
        [0.0, "rgba(29,158,117,0.50)"],
        [0.5, "rgba(245,220,100,0.35)"],
        [1.0, "rgba(226,75,74,0.50)"],
    ]

    fig = go.Figure(go.Heatmap(
        z=z_arr,
        x=[f"β={b}" for b in BETA_VALS],
        y=[f"α={a}" for a in ALPHA_VALS],
        colorscale=colorscale,
        showscale=False,
        hovertemplate="α=%{y}, β=%{x}<br>Avg JCT = %{z:.2f}s<extra></extra>",
        zsmooth=False,
    ))

    annotations = []
    for ai in range(len(ALPHA_VALS)):
        for bi in range(len(BETA_VALS)):
            is_best = (ai == min_idx[0] and bi == min_idx[1])
            label = ("★ " if is_best else "") + f"{z_arr[ai][bi]:.2f}"
            annotations.append(dict(
                x=bi, y=ai, text=label, showarrow=False,
                font=dict(size=11, color="#111", family="Arial"),
            ))

    shapes = []
    for ai in range(len(ALPHA_VALS)):
        for bi in range(len(BETA_VALS)):
            is_best = (ai == min_idx[0] and bi == min_idx[1])
            is_sel  = (ai == sel_ai and bi == sel_bi)
            if is_best or is_sel:
                shapes.append(dict(
                    type="rect",
                    x0=bi-0.5, x1=bi+0.5, y0=ai-0.5, y1=ai+0.5,
                    line=dict(color="#f59e0b" if is_sel else "#1D9E75", width=3),
                    fillcolor="rgba(0,0,0,0)",
                    layer="above",
                ))

    fig.update_layout(
        paper_bgcolor="white", plot_bgcolor="white",
        margin=dict(l=55, r=10, t=10, b=40),
        font=dict(family="Arial", size=10, color="#999"),
        annotations=annotations, shapes=shapes, height=240,
        xaxis=dict(tickfont=dict(size=10, color="#777"), side="bottom",
                   showgrid=False, zeroline=False, showline=False),
        yaxis=dict(tickfont=dict(size=10, color="#777"),
                   showgrid=False, zeroline=False, showline=False),
    )
    return fig

# ── HTML helpers ──────────────────────────────────────────────────────────────
def legend_html(names, colors):
    items = "".join(
        f'<span class="leg-item"><span class="leg-dot" style="background:{c}"></span>{n}</span>'
        for n, c in zip(names, colors)
    )
    return f'<div class="legend-row">{items}</div>'

def kpi_card(title, hint, fcfs_val, ltr_val, rob_val, ltr_sub, rob_sub):
    return f"""<div class="kpi-card">
      <div class="kpi-title">{title}</div>
      <div class="kpi-hint">{hint}</div>
      <div class="kpi-row">
        <div class="mc" style="background:#FEECEC">
          <div class="mc-label" style="color:#A32D2D">FCFS</div>
          <div class="mc-val"   style="color:#A32D2D">{fcfs_val}</div>
          <div class="mc-sub"   style="color:#A32D2D">baseline</div>
        </div>
        <div class="mc" style="background:#E8F1FB">
          <div class="mc-label" style="color:#185FA5">LTR</div>
          <div class="mc-val"   style="color:#185FA5">{ltr_val}</div>
          <div class="mc-sub"   style="color:#185FA5">{ltr_sub}</div>
        </div>
        <div class="mc" style="background:#E4F5EE">
          <div class="mc-label" style="color:#0F6E56">Ours</div>
          <div class="mc-val"   style="color:#0F6E56">{rob_val}</div>
          <div class="mc-sub"   style="color:#0F6E56">{rob_sub}</div>
        </div>
      </div>
    </div>"""

def chart_header(title, sub, leg):
    sub_html = f'<div class="chart-sub">{sub}</div>' if sub else ""
    return f'<div class="chart-title">{title}</div>{sub_html}{leg}'

def pct_vs(val, base):
    if base == 0:
        return "n/a"
    return f"−{round((1 - val / base) * 100)}% vs FCFS"

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    data = load_data()
    cfg  = {"displayModeBar": False}

    st.title("LLM Scheduler Simulation")

    # ── Controls ──────────────────────────────────────────────────────────────
    ctrl1, ctrl2, ctrl3, ctrl4 = st.columns([2, 1, 1, 1])
    with ctrl1:
        error_pct = st.select_slider(
            "Prediction error", options=ERROR_LEVELS, value=40,
            format_func=lambda x: f"{x}%",
        )
    with ctrl2:
        alpha = st.selectbox("Alpha (α) — uncertainty parameter",    ALPHA_VALS, index=1)
    with ctrl3:
        beta  = st.selectbox("Beta (β) — anti-starvation parameter", BETA_VALS,  index=1)
    with ctrl4:
        req = st.selectbox("Requests", list(REQ_SCALE.keys()), index=1)

    rs = REQ_SCALE[req]
    st.caption(f"α = {alpha} · β = {beta} · {req:,} requests · Error: {error_pct}%")

    # ── Fetch base rows ────────────────────────────────────────────────────────
    fcfs = get_row(data["fcfs"],   error_pct)
    ltr  = get_row(data["ltr"],    error_pct)
    rob  = get_row(data["robust"], error_pct)

    # ⚠️  SYNTHETIC DATA ONLY — remove cap= when switching to real CSVs
    rob_jct  = apply_params(rob["jct"],  alpha, beta, cap=ltr["jct"])
    rob_ttft = apply_params(rob["ttft"], alpha, beta, cap=ltr["ttft"])
    rob_pre  = apply_params_pre(int(rob["preemptions"] * rs), alpha, beta)
    rob_jai  = apply_params_jain(rob["jain_fairness"], alpha, beta)
    str_rob  = apply_params_starv(rob["starvation_pct"] * rs, alpha, beta)

    pre_fcfs = int(fcfs["preemptions"] * rs)
    pre_ltr  = int(ltr["preemptions"]  * rs)
    str_fcfs = round(fcfs["starvation_pct"] * rs, 1)
    str_ltr  = round(ltr["starvation_pct"]  * rs, 1)

    # ── KPI cards (2×2) ───────────────────────────────────────────────────────
    k1, k2 = st.columns(2)
    k3, k4 = st.columns(2)

    with k1:
        st.markdown(kpi_card(
            "Job Completion Time (JCT)", "lower is better",
            f"{fcfs['jct']:.2f}s", f"{ltr['jct']:.2f}s", f"{rob_jct:.2f}s",
            pct_vs(ltr["jct"], fcfs["jct"]), pct_vs(rob_jct, fcfs["jct"]),
        ), unsafe_allow_html=True)
    with k2:
        st.markdown(kpi_card(
            "Time to First Token (TTFT)", "lower is better",
            f"{fcfs['ttft']:.2f}s", f"{ltr['ttft']:.2f}s", f"{rob_ttft:.2f}s",
            pct_vs(ltr["ttft"], fcfs["ttft"]), pct_vs(rob_ttft, fcfs["ttft"]),
        ), unsafe_allow_html=True)
    with k3:
        st.markdown(kpi_card(
            "Preemptions", "lower is better",
            str(pre_fcfs), str(pre_ltr), str(rob_pre),
            f"{pre_ltr - pre_fcfs:+d} vs FCFS", f"{rob_pre - pre_fcfs:+d} vs FCFS",
        ), unsafe_allow_html=True)
    with k4:
        jd_ltr = round(ltr["jain_fairness"] - fcfs["jain_fairness"], 2)
        jd_rob = round(rob_jai - fcfs["jain_fairness"], 2)
        st.markdown(kpi_card(
            "Jain's Fairness Index", "closer to 1.0 is better",
            f"{fcfs['jain_fairness']:.2f}", f"{ltr['jain_fairness']:.2f}", f"{rob_jai:.2f}",
            f"{jd_ltr:+.2f} vs FCFS", f"{jd_rob:+.2f} vs FCFS",
        ), unsafe_allow_html=True)

    st.write("")

    # ── Chart shared config ────────────────────────────────────────────────────
    el_labels = [f"{e}%" for e in ERROR_LEVELS]
    leg    = legend_html(["FCFS", "LTR", "Ours"], [C_FCFS, C_LTR, C_ROB])
    names  = ["FCFS", "LTR", "Ours"]
    colors = [C_FCFS, C_LTR, C_ROB]
    fills  = ["rgba(226,75,74,0.05)", "rgba(55,138,221,0.05)", "rgba(29,158,117,0.07)"]

    ltr_jct_list  = data["ltr"]["jct"].tolist()
    ltr_ttft_list = data["ltr"]["ttft"].tolist()

    jct_line = [
        data["fcfs"]["jct"].tolist(),
        ltr_jct_list,
        [apply_params(v, alpha, beta, cap=ltr_jct_list[i])   # remove cap= for real data
         for i, v in enumerate(data["robust"]["jct"])],
    ]
    ttft_line = [
        data["fcfs"]["ttft"].tolist(),
        ltr_ttft_list,
        [apply_params(v, alpha, beta, cap=ltr_ttft_list[i])  # remove cap= for real data
         for i, v in enumerate(data["robust"]["ttft"])],
    ]
    starv_line = [
        (data["fcfs"]["starvation_pct"] * rs).round(1).tolist(),
        (data["ltr"]["starvation_pct"]  * rs).round(1).tolist(),
        [apply_params_starv(v * rs, alpha, beta)
         for v in data["robust"]["starvation_pct"]],
    ]

    # ── ROW 1: JCT ────────────────────────────────────────────────────────────
    r1a, r1b = st.columns(2)
    with r1a:
        with st.container(border=True):
            st.markdown(chart_header(
                "Job Completion Time — current error level",
                f"At {error_pct}% prediction error", leg), unsafe_allow_html=True)
            st.plotly_chart(bar_chart(
                ["FCFS", "LTR", "Ours"],
                [fcfs["jct"], ltr["jct"], rob_jct],
                [C_FCFS_A, C_LTR_A, C_ROB_A], "s"),
                use_container_width=True, config=cfg)
    with r1b:
        with st.container(border=True):
            st.markdown(chart_header(
                "Job Completion Time — all error levels",
                "Full error range 0%–80%", leg), unsafe_allow_html=True)
            st.plotly_chart(line_chart(el_labels, jct_line, names, colors, fills, "s"),
                use_container_width=True, config=cfg)

    # ── ROW 2: Starvation ─────────────────────────────────────────────────────
    r2a, r2b = st.columns(2)
    with r2a:
        with st.container(border=True):
            st.markdown(chart_header(
                "Starvation frequency — current error level",
                f"At {error_pct}% prediction error", leg), unsafe_allow_html=True)
            st.plotly_chart(bar_chart(
                ["FCFS", "LTR", "Ours"],
                [str_fcfs, str_ltr, str_rob],
                [C_FCFS_A, C_LTR_A, C_ROB_A], "%"),
                use_container_width=True, config=cfg)
    with r2b:
        with st.container(border=True):
            st.markdown(chart_header(
                "Starvation frequency — all error levels",
                "Full error range 0%–80%", leg), unsafe_allow_html=True)
            st.plotly_chart(line_chart(el_labels, starv_line, names, colors, fills, "%"),
                use_container_width=True, config=cfg)

    # ── ROW 3: TTFT ───────────────────────────────────────────────────────────
    r3a, r3b = st.columns(2)
    with r3a:
        with st.container(border=True):
            st.markdown(chart_header(
                "Time to First Token — current error level",
                f"At {error_pct}% prediction error", leg), unsafe_allow_html=True)
            st.plotly_chart(bar_chart(
                ["FCFS", "LTR", "Ours"],
                [fcfs["ttft"], ltr["ttft"], rob_ttft],
                [C_FCFS_A, C_LTR_A, C_ROB_A], "s"),
                use_container_width=True, config=cfg)
    with r3b:
        with st.container(border=True):
            st.markdown(chart_header(
                "Time to First Token — all error levels",
                "Full error range 0%–80%", leg), unsafe_allow_html=True)
            st.plotly_chart(line_chart(el_labels, ttft_line, names, colors, fills, "s"),
                use_container_width=True, config=cfg)

    # ── Heatmap ───────────────────────────────────────────────────────────────
    with st.container(border=True):
        st.markdown(
            '<div class="chart-title">α / β sensitivity — Average JCT (Ours)</div>',
            unsafe_allow_html=True)
        hm_leg = """<div class="legend-row" style="margin-bottom:8px">
          <span class="leg-item"><span style="display:inline-block;width:10px;height:10px;
            border-radius:2px;background:rgba(29,158,117,0.50);border:1px solid rgba(0,0,0,0.07)">
            </span>lower JCT — better</span>
          <span class="leg-item"><span style="display:inline-block;width:10px;height:10px;
            border-radius:2px;background:rgba(226,75,74,0.50);border:1px solid rgba(0,0,0,0.07)">
            </span>higher JCT — worse</span>
          <span class="leg-item"><span style="display:inline-block;width:10px;height:10px;
            border-radius:2px;background:rgba(245,158,11,0.15);border:2px solid #f59e0b">
            </span>current selection</span>
          <span class="leg-item"><span style="display:inline-block;width:10px;height:10px;
            border-radius:2px;background:rgba(29,158,117,0.15);border:1.5px solid #1D9E75">
            </span>best config</span>
        </div>"""
        st.markdown(hm_leg, unsafe_allow_html=True)
        # base = Robust JCT at α=0.5, β=0.5 (reference config, no parameter penalty)
        # With real data: use JCT from sweep CSV at alpha=0.5, beta=0.5, current error_pct
        base_for_hm = get_row(data["robust"], error_pct)["jct"]
        st.plotly_chart(heatmap_chart(base_for_hm, alpha, beta),
            use_container_width=True, config=cfg)
        st.caption(
            "**α** higher = more conservative with uncertain predictions = fewer preemptions, "
            "but increases JCT  ·  "
            "**β** higher = fairer to long-waiting requests, but slightly increases average JCT"
        )

if __name__ == "__main__":
    main()
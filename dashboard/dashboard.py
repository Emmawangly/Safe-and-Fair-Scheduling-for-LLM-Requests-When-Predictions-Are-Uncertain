import os
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
.kpi-card { background: #ffffff; border: 1px solid #e0e0e0; border-radius: 8px; padding: 12px 14px; margin-bottom: 10px; }
.kpi-title { font-size: 12px; font-weight: 700; color: #444; margin-bottom: 2px; }
.kpi-hint { font-size: 9px; color: #999; font-style: italic; margin-bottom: 8px; }
.kpi-row { display: grid; grid-template-columns: repeat(3,1fr); gap: 6px; }
.mc { text-align: center; padding: 6px 4px; border-radius: 6px; }
.mc-label { font-size: 9px; font-weight: 700; margin-bottom: 2px; }
.mc-val { font-size: 16px; font-weight: 700; }
.mc-sub { font-size: 9px; margin-top: 2px; }
.chart-title { font-size: 12px; font-weight: 700; color: #1a1a2e; margin-bottom: 2px; }
.chart-sub { font-size: 10px; color: #999; font-style: italic; margin-bottom: 4px; }
.legend-row { display: flex; gap: 12px; flex-wrap: wrap; margin-bottom: 4px; }
.leg-item { display: flex; align-items: center; gap: 5px; font-size: 10px; color: #666; }
.leg-dot { width: 8px; height: 8px; border-radius: 50%; display: inline-block; }
[data-testid="stVerticalBlockBorderWrapper"] { border-radius: 8px !important; border-color: #e5e5e5 !important; }
</style>
""", unsafe_allow_html=True)

fcfs_color = "#E24B4A"
ltr_color = "#378ADD"
rob_color = "#1D9E75"
ljf_color = "#7D3C98"

                 
SCENARIOS = {
    "Light Load (aprox. 43% utilization)": {
        "folder": "light",
        "color": "#27AE60",
        "bps": 15.8,
        "util": "aprox. 43%",
        "context": "Private LLM, corporate internal tool",
    },
    "Normal Load (aprox. 76% utilization) - Main Use": {
        "folder": "normal",
        "color": "#2874A7",
        "bps": 9.0,
        "util": "aprox. 76%",
        "context": "Commercial production service",
    },
    "Stress (aprox. 153% utilization)": {
        "folder": "stress",
        "color": "#BA4A00",
        "bps": 4.5,
        "util": "aprox. 153%",
        "context": "Traffic spike, viral event, peak demand",
    },
    "Saturation (aprox. 214% utilization)": {
        "folder": "saturation",
        "color": "#922B21",
        "bps": 3.2,
        "util": "aprox. 214%",
        "context": "System at capacity limit",
    },
}

error_levels = [0, 20, 40, 60, 80]
alpha_vals = [0.10, 0.25, 0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 2.00]
beta_vals = [0.10, 0.25, 0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 2.00, 5.00, 10.00, 12.00, 14.00, 16.00, 18.00, 20.00, 25.00, 30.00]

_here = os.path.dirname(os.path.abspath(__file__))
_root = os.path.dirname(_here)
_results = os.path.join(_root, "results")


@st.cache_data
def load_tuned_params():
     
    path = os.path.join(_results, "best_params_per_scenario.csv")
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        return {}
    return {row["scenario"]: (float(row["alpha"]), float(row["beta"])) for _, row in df.iterrows()}


tuned_params = load_tuned_params()


@st.cache_data
def load_ood_data():
    folder = os.path.join(_results, "ood_test")
    try:
        return {
            "fcfs": pd.read_csv(os.path.join(folder, "results_fcfs.csv")),
            "ltr": pd.read_csv(os.path.join(folder, "results_ltr.csv")),
            "robust": pd.read_csv(os.path.join(folder, "results_robust.csv")),
        }
    except FileNotFoundError:
        return None


@st.cache_data
def load_data(scenario_folder):
    folder = os.path.join(_results, scenario_folder)
    if not os.path.isdir(folder):
        folder = _results

    data = {
        "fcfs": pd.read_csv(os.path.join(folder, "results_fcfs.csv")),
        "ltr": pd.read_csv(os.path.join(folder, "results_ltr.csv")),
        "robust": pd.read_csv(os.path.join(folder, "results_robust.csv")),
    }

    try:
        data["ljf"] = pd.read_csv(os.path.join(folder, "results_ljf.csv"))
    except FileNotFoundError:
        data["ljf"] = None

    sweep_path = os.path.join(_results, "results_parameter_sweep.csv")
    try:
        sweep = pd.read_csv(sweep_path)
        if "scenario" in sweep.columns:
            sweep = sweep[sweep["scenario"] == scenario_folder]
        data["sweep"] = sweep
    except FileNotFoundError:
        data["sweep"] = None

    ablation_path = os.path.join(_results, "results_ablation.csv")
    try:
        data["ablation"] = pd.read_csv(ablation_path)
    except FileNotFoundError:
        data["ablation"] = None

    return data


def get_row(df, err):
    rows = df[df["error_pct"] == err]
    if rows.empty:
        st.error(f"No data for error_pct={err}.")
        st.stop()
    return rows.iloc[0]


def get_robust_row(sweep_df, err, alpha, beta):
    match = sweep_df[
        (sweep_df["error_pct"] == err)
        & (sweep_df["alpha"].round(3) == round(alpha, 3))
        & (sweep_df["beta"].round(3) == round(beta, 3))
    ]
    if match.empty:
        st.error(f"No sweep row for alpha={alpha}, beta={beta}, error_pct={err}.")
        st.stop()
    return match.iloc[0]


def get_robust_series(sweep_df, alpha, beta):
    match = sweep_df[
        (sweep_df["alpha"].round(3) == round(alpha, 3))
        & (sweep_df["beta"].round(3) == round(beta, 3))
    ]
    return match.sort_values("error_pct")


def safe_get(row, col, default=0.0):
    return float(row[col]) if col in row.index else default


def make_bar(labels, values, colors, suffix="s"):
    def fmt(v):
        if suffix == "%": return f"{v:.1f}%"
        if suffix == "": return f"{v:.0f}"
        return f"{v:.2f}s"

    fig = go.Figure(go.Bar(
        x=labels, y=values, marker_color=colors,
        text=[fmt(v) for v in values], textposition="outside",
        textfont=dict(size=10, color="#333"), width=0.45,
    ))
    fig.update_layout(
        paper_bgcolor="white", plot_bgcolor="white",
        margin=dict(l=40, r=10, t=20, b=30), showlegend=False, height=200,
        font=dict(family="Arial", size=10, color="#999"),
        yaxis=dict(gridcolor="rgba(0,0,0,0.06)", ticksuffix=suffix, rangemode="tozero",
                   range=[0, max(max(values), 0.01) * 1.28], tickfont=dict(size=9, color="#bbb")),
        xaxis=dict(showgrid=False, tickfont=dict(size=9, color="#bbb")),
    )
    return fig


def make_line(x, series, names, colors, suffix="s"):
    base_fills = [
        "rgba(226,75,74,0.05)", "rgba(55,138,221,0.05)",
        "rgba(29,158,117,0.07)", "rgba(125,60,152,0.05)",
    ]
    fig = go.Figure()
    for i, (vals, name, color) in enumerate(zip(series, names, colors)):
        fig.add_trace(go.Scatter(
            x=x, y=vals, name=name, mode="lines+markers",
            line=dict(color=color, width=2, dash="dash" if name == "LJF (worst)" else "solid"),
            marker=dict(size=4),
            fill="tozeroy", fillcolor=base_fills[i] if i < len(base_fills) else "rgba(0,0,0,0.03)",
        ))
    fig.update_layout(
        paper_bgcolor="white", plot_bgcolor="white",
        margin=dict(l=40, r=10, t=20, b=30), showlegend=False, height=200,
        font=dict(family="Arial", size=10, color="#999"),
        yaxis=dict(gridcolor="rgba(0,0,0,0.06)", ticksuffix=suffix, rangemode="tozero",
                   tickfont=dict(size=9, color="#bbb")),
        xaxis=dict(gridcolor="rgba(0,0,0,0.06)", tickfont=dict(size=9, color="#bbb")),
    )
    return fig


def legend_dots(names, colors):
    items = "".join(
        f'<span class="leg-item"><span class="leg-dot" style="background:{c}"></span>{n}</span>'
        for n, c in zip(names, colors)
    )
    return f'<div class="legend-row">{items}</div>'


def kpi_card(title, hint, v_fcfs, v_ltr, v_rob, sub_ltr, sub_rob):
    return f"""<div class="kpi-card">
      <div class="kpi-title">{title}</div>
      <div class="kpi-hint">{hint}</div>
      <div class="kpi-row">
        <div class="mc" style="background:#FEECEC">
          <div class="mc-label" style="color:#A32D2D">FCFS</div>
          <div class="mc-val" style="color:#A32D2D">{v_fcfs}</div>
          <div class="mc-sub" style="color:#A32D2D">baseline</div>
        </div>
        <div class="mc" style="background:#E8F1FB">
          <div class="mc-label" style="color:#185FA5">LTR</div>
          <div class="mc-val" style="color:#185FA5">{v_ltr}</div>
          <div class="mc-sub" style="color:#185FA5">{sub_ltr}</div>
        </div>
        <div class="mc" style="background:#E4F5EE">
          <div class="mc-label" style="color:#0F6E56">Ours</div>
          <div class="mc-val" style="color:#0F6E56">{v_rob}</div>
          <div class="mc-sub" style="color:#0F6E56">{sub_rob}</div>
        </div>
      </div>
    </div>"""


def diff_vs_fcfs(val, base):
    if base == 0:
        return "n/a"
    d = round((1 - val / base) * 100)
    return f"{'minus' if d > 0 else 'plus'}{abs(d)}% vs FCFS".replace("minus", "-").replace("plus", "+")


def heatmap_chart(sweep_df, err, alpha, beta, metric="jct", label="JCT", suffix="s", best_is_low=True):
    if sweep_df is None or sweep_df.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="Sweep data not available. Run python sweep.py to generate results/results_sweep.csv",
            xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False,
            font=dict(size=13, color="#888"), align="center",
        )
        fig.update_layout(
            paper_bgcolor="white", plot_bgcolor="#fafafa", height=300,
            margin=dict(l=55, r=10, t=10, b=40),
            xaxis=dict(visible=False), yaxis=dict(visible=False),
        )
        return fig

    subset = sweep_df[sweep_df["error_pct"] == err]
    av = sorted(sweep_df["alpha"].unique().tolist())
    bv = sorted(sweep_df["beta"].unique().tolist())

    z = []
    for a in av:
        row = []
        for b in bv:
            m = subset[(subset["alpha"].round(3) == round(a, 3)) & (subset["beta"].round(3) == round(b, 3))]
            row.append(round(float(m[metric].values[0]), 2) if not m.empty else float("nan"))
        z.append(row)

    z_arr = np.array(z)
    masked = np.ma.masked_invalid(z_arr)
    best = np.unravel_index(np.argmin(masked) if best_is_low else np.argmax(masked), z_arr.shape)
    sel_a = min(range(len(av)), key=lambda i: abs(av[i] - alpha))
    sel_b = min(range(len(bv)), key=lambda i: abs(bv[i] - beta))

    annotations = []
    for i in range(len(av)):
        for j in range(len(bv)):
            is_best = (i == best[0] and j == best[1])
            val = z_arr[i][j]
            txt = ("* " if is_best else "") + (f"{val:.1f}" if not np.isnan(val) else "-")
            annotations.append(dict(x=j, y=i, text=txt, showarrow=False, font=dict(size=11, color="#111")))

    shapes = []
    for i in range(len(av)):
        for j in range(len(bv)):
            if (i == best[0] and j == best[1]) or (i == sel_a and j == sel_b):
                c = "#f59e0b" if (i == sel_a and j == sel_b) else "#1D9E75"
                shapes.append(dict(type="rect", x0=j-0.5, x1=j+0.5, y0=i-0.5, y1=i+0.5,
                                   line=dict(color=c, width=2), fillcolor="rgba(0,0,0,0)", layer="above"))

    fig = go.Figure(go.Heatmap(
        z=z_arr,
        x=[f"β={b}" for b in bv],
        y=[f"α={a}" for a in av],
        colorscale=[[0.0, "rgba(29,158,117,0.55)"], [0.5, "rgba(245,220,100,0.40)"], [1.0, "rgba(226,75,74,0.55)"]],
        showscale=False,
        hovertemplate=f"α=%{{y}}, β=%{{x}}<br>{label} = %{{z:.2f}}{suffix}<extra></extra>",
    ))
    fig.update_layout(
        paper_bgcolor="white", plot_bgcolor="white",
        margin=dict(l=60, r=10, t=10, b=40),
        annotations=annotations, shapes=shapes, height=380,
        font=dict(family="Arial", size=10, color="#999"),
        xaxis=dict(tickfont=dict(size=10, color="#777"), side="bottom", showgrid=False, zeroline=False),
        yaxis=dict(tickfont=dict(size=10, color="#777"), showgrid=False, zeroline=False),
    )
    return fig


def main():
    cfg = {"displayModeBar": False}
    st.title("LLM Scheduler Simulation")

    col_scen, col_err = st.columns([3, 2])
    with col_scen:
        selected = st.selectbox("Load scenario", list(SCENARIOS.keys()), index=1)
    with col_err:
        error_pct = st.select_slider("Prediction error", options=error_levels, value=40,
                                     format_func=lambda x: f"{x}%")

    scen = SCENARIOS[selected]
    tuned_alpha, tuned_beta = tuned_params.get(scen["folder"], (0.5, 0.5))

                                                                        
                                                                            
                                                                  
    if st.session_state.get("last_scenario") != selected:
        st.session_state["last_scenario"] = selected
        st.session_state["alpha_select"] = tuned_alpha
        st.session_state["beta_select"] = tuned_beta

    col_a, col_b, col_reset = st.columns([2, 2, 1])
    with col_reset:
        st.write("")
        st.write("")
        if st.button("↻ reset to recommended"):
            st.session_state["alpha_select"] = tuned_alpha
            st.session_state["beta_select"] = tuned_beta
            st.rerun()
    with col_a:
        a_index = min(range(len(alpha_vals)), key=lambda i: abs(alpha_vals[i] - tuned_alpha))
        alpha = st.selectbox("Alpha (α) for Ours", alpha_vals, index=a_index, key="alpha_select")
    with col_b:
        b_index = min(range(len(beta_vals)), key=lambda i: abs(beta_vals[i] - tuned_beta))
        beta = st.selectbox("Beta (β) for Ours", beta_vals, index=b_index, key="beta_select")
    st.caption(
        f"recommended for this scenario: α={tuned_alpha}, β={tuned_beta}. "
        "FCFS, LTR and LJF have no tunable parameters, only Ours moves with these."
    )

    st.markdown(
        f'<div style="background:{scen["color"]};padding:10px 14px;border-radius:8px;'
        f'color:white;font-size:13px;margin-bottom:12px;">'
        f'<b>{selected}</b> &nbsp;->&nbsp; {scen["context"]}</div>',
        unsafe_allow_html=True,
    )

    data = load_data(scen["folder"])
    fcfs = get_row(data["fcfs"], error_pct)
    ltr = get_row(data["ltr"], error_pct)
    ljf = get_row(data["ljf"], error_pct) if data["ljf"] is not None else None

    if data["sweep"] is None or data["sweep"].empty:
        st.error("Sweep data not found. Run python sweep.py first.")
        st.stop()
    rob = get_robust_row(data["sweep"], error_pct, alpha, beta)

                                                                                 
                                                                                  
                                                                                
    show_ljf = False

    k1, k2 = st.columns(2)
    with k1:
        st.markdown(kpi_card(
            "Job Completion Time (JCT)", "lower is better",
            f"{fcfs['jct']:.2f}s", f"{ltr['jct']:.2f}s", f"{rob['jct']:.2f}s",
            diff_vs_fcfs(ltr["jct"], fcfs["jct"]),
            diff_vs_fcfs(rob["jct"], fcfs["jct"]),
        ), unsafe_allow_html=True)
    with k2:
        st.markdown(kpi_card(
            "Time to First Token (TTFT)", "lower is better",
            f"{fcfs['ttft']:.2f}s", f"{ltr['ttft']:.2f}s", f"{rob['ttft']:.2f}s",
            diff_vs_fcfs(ltr["ttft"], fcfs["ttft"]),
            diff_vs_fcfs(rob["ttft"], fcfs["ttft"]),
        ), unsafe_allow_html=True)

    k3, k4 = st.columns(2)
    with k3:
        fp = int(fcfs["preemptions"])
        lp = int(ltr["preemptions"])
        rp = int(rob["preemptions"])
        st.markdown(kpi_card(
            "Risky Predictions", "underestimated at admission - lower is better. FCFS shown as the scheduler-blind baseline, since it never uses predictions to choose order.",
            str(fp), str(lp), str(rp),
            f"{lp - fp:+d} vs FCFS", f"{rp - fp:+d} vs FCFS",
        ), unsafe_allow_html=True)
    with k4:
        jd_ltr = round(ltr["jain_fairness"] - fcfs["jain_fairness"], 2)
        jd_rob = round(rob["jain_fairness"] - fcfs["jain_fairness"], 2)
        st.markdown(kpi_card(
            "Jain's Fairness Index", "closer to 1.0 is better",
            f"{fcfs['jain_fairness']:.2f}", f"{ltr['jain_fairness']:.2f}", f"{rob['jain_fairness']:.2f}",
            f"{jd_ltr:+.2f} vs FCFS", f"{jd_rob:+.2f} vs FCFS",
        ), unsafe_allow_html=True)

    k5, k6 = st.columns(2)
    with k5:
        ft = safe_get(fcfs, "timed_out_pct")
        lt = safe_get(ltr, "timed_out_pct")
        rt = safe_get(rob, "timed_out_pct")
        st.markdown(kpi_card(
            "Timed Out %", "requests cancelled after waiting too long - lower is better",
            f"{ft:.1f}%", f"{lt:.1f}%", f"{rt:.1f}%",
            f"{lt - ft:+.1f}% vs FCFS", f"{rt - ft:+.1f}% vs FCFS",
        ), unsafe_allow_html=True)
    with k6:
        fp95 = safe_get(fcfs, "jct_p95"); fp99 = safe_get(fcfs, "jct_p99")
        lp95 = safe_get(ltr, "jct_p95"); lp99 = safe_get(ltr, "jct_p99")
        rp95 = safe_get(rob, "jct_p95"); rp99 = safe_get(rob, "jct_p99")
        st.markdown(f"""<div class="kpi-card">
          <div class="kpi-title">Tail Latency (P95 / P99)</div>
          <div class="kpi-hint">worst JCT for 95% and 99% of requests - lower is better</div>
          <div class="kpi-row">
            <div class="mc" style="background:#FEECEC">
              <div class="mc-label" style="color:#A32D2D">FCFS</div>
              <div class="mc-val" style="color:#A32D2D;font-size:13px">{fp95:.1f}s</div>
              <div class="mc-sub" style="color:#A32D2D">P99: {fp99:.1f}s</div>
            </div>
            <div class="mc" style="background:#E8F1FB">
              <div class="mc-label" style="color:#185FA5">LTR</div>
              <div class="mc-val" style="color:#185FA5;font-size:13px">{lp95:.1f}s</div>
              <div class="mc-sub" style="color:#185FA5">P99: {lp99:.1f}s</div>
            </div>
            <div class="mc" style="background:#E4F5EE">
              <div class="mc-label" style="color:#0F6E56">Ours</div>
              <div class="mc-val" style="color:#0F6E56;font-size:13px">{rp95:.1f}s</div>
              <div class="mc-sub" style="color:#0F6E56">P99: {rp99:.1f}s</div>
            </div>
          </div>
        </div>""", unsafe_allow_html=True)

    st.write("")

    fs = data["fcfs"].sort_values("error_pct")
    ls = data["ltr"].sort_values("error_pct")
    rs = get_robust_series(data["sweep"], alpha, beta)
    xs = data["ljf"].sort_values("error_pct") if show_ljf else None
    el = [f"{e}%" for e in error_levels]

    names = ["FCFS", "LTR", "Ours"] + (["LJF (worst)"] if show_ljf else [])
    scolors = [fcfs_color, ltr_color, rob_color] + ([ljf_color] if show_ljf else [])
    leg = legend_dots(names, scolors)

    bar_c = ["rgba(226,75,74,0.82)", "rgba(55,138,221,0.82)", "rgba(29,158,117,0.82)"] +\
            (["rgba(125,60,152,0.82)"] if show_ljf else [])

    def jct_bars():
        lbs = ["FCFS", "LTR", "Ours"] + (["LJF"] if show_ljf else [])
        vals = [fcfs["jct"], ltr["jct"], rob["jct"]] + ([ljf["jct"]] if show_ljf else [])
        return make_bar(lbs, vals, bar_c)

    def starv_bars():
        lbs = ["FCFS", "LTR", "Ours"] + (["LJF"] if show_ljf else [])
        vals = [fcfs["starvation_pct"], ltr["starvation_pct"], rob["starvation_pct"]] +\
               ([ljf["starvation_pct"]] if show_ljf else [])
        return make_bar(lbs, vals, bar_c, "%")

    def ttft_bars():
        lbs = ["FCFS", "LTR", "Ours"] + (["LJF"] if show_ljf else [])
        vals = [fcfs["ttft"], ltr["ttft"], rob["ttft"]] + ([ljf["ttft"]] if show_ljf else [])
        return make_bar(lbs, vals, bar_c)

    def pre_bars():
        lbs = ["FCFS", "LTR", "Ours"] + (["LJF"] if show_ljf else [])
        vals = [float(fcfs["preemptions"]), float(ltr["preemptions"]), float(rob["preemptions"])] +\
               ([float(ljf["preemptions"])] if show_ljf else [])
        return make_bar(lbs, vals, bar_c, "")

    jct_series = [fs["jct"].tolist(), ls["jct"].tolist(), rs["jct"].tolist()] +\
                 ([xs["jct"].tolist()] if show_ljf else [])
    starv_series = [fs["starvation_pct"].tolist(), ls["starvation_pct"].tolist(), rs["starvation_pct"].tolist()] +\
                   ([xs["starvation_pct"].tolist()] if show_ljf else [])
    ttft_series = [fs["ttft"].tolist(), ls["ttft"].tolist(), rs["ttft"].tolist()] +\
                  ([xs["ttft"].tolist()] if show_ljf else [])
    pre_series = [fs["preemptions"].tolist(), ls["preemptions"].tolist(), rs["preemptions"].tolist()] +\
                 ([xs["preemptions"].tolist()] if show_ljf else [])

    r1a, r1b = st.columns(2)
    with r1a:
        with st.container(border=True):
            st.markdown(f'<div class="chart-title">JCT at {error_pct}% error</div>' + leg, unsafe_allow_html=True)
            st.plotly_chart(jct_bars(), use_container_width=True, config=cfg)
    with r1b:
        with st.container(border=True):
            st.markdown('<div class="chart-title">JCT across error levels</div>' + leg, unsafe_allow_html=True)
            st.plotly_chart(make_line(el, jct_series, names, scolors), use_container_width=True, config=cfg)

    r2a, r2b = st.columns(2)
    with r2a:
        with st.container(border=True):
            st.markdown(f'<div class="chart-title">Starvation at {error_pct}% error</div>' + leg, unsafe_allow_html=True)
            st.plotly_chart(starv_bars(), use_container_width=True, config=cfg)
    with r2b:
        with st.container(border=True):
            st.markdown('<div class="chart-title">Starvation across error levels</div>' + leg, unsafe_allow_html=True)
            st.plotly_chart(make_line(el, starv_series, names, scolors, "%"), use_container_width=True, config=cfg)

    r3a, r3b = st.columns(2)
    with r3a:
        with st.container(border=True):
            st.markdown(f'<div class="chart-title">TTFT at {error_pct}% error</div>' + leg, unsafe_allow_html=True)
            st.plotly_chart(ttft_bars(), use_container_width=True, config=cfg)
    with r3b:
        with st.container(border=True):
            st.markdown('<div class="chart-title">TTFT across error levels</div>' + leg, unsafe_allow_html=True)
            st.plotly_chart(make_line(el, ttft_series, names, scolors), use_container_width=True, config=cfg)

    r4a, r4b = st.columns(2)
    with r4a:
        with st.container(border=True):
            st.markdown(f'<div class="chart-title">Risky Predictions at {error_pct}% error</div>' + leg, unsafe_allow_html=True)
            st.plotly_chart(pre_bars(), use_container_width=True, config=cfg)
    with r4b:
        with st.container(border=True):
            st.markdown('<div class="chart-title">Risky Predictions across error levels</div>' + leg, unsafe_allow_html=True)
            st.plotly_chart(make_line(el, pre_series, names, scolors, ""), use_container_width=True, config=cfg)

    with st.container(border=True):
        st.markdown('<div class="chart-title">alpha / beta sensitivity (Ours only)</div>', unsafe_allow_html=True)
        st.caption(
            "JCT is lowest at low beta (closer to LTR). Starvation is lowest at high beta. "
            "These two maps pull in opposite directions on purpose -- the yellow-bordered cell "
            "is the recommended point, chosen to reach near-zero starvation at the lowest JCT that gets there. "
            "The green star in each map marks whichever cell wins on that map alone, ignoring the other metric."
        )
        st.caption("colored by JCT (lower is better)")
        st.plotly_chart(
            heatmap_chart(data["sweep"], error_pct, alpha, beta, metric="jct", label="JCT", suffix="s", best_is_low=True),
            use_container_width=True, config=cfg)
        st.caption("colored by starvation % (lower is better)")
        st.plotly_chart(
            heatmap_chart(data["sweep"], error_pct, alpha, beta, metric="starvation_pct", label="Starvation", suffix="%", best_is_low=True),
            use_container_width=True, config=cfg)

    with st.expander("Ablation Study (contribution of a and b)"):
        st.write("**α = 0** removes uncertainty penalty. **β = 0** removes starvation protection.")

        abl = data["ablation"]
        if abl is not None:
            subset = abl[abl["error_pct"] == error_pct]
            row_full = subset[subset["configuration"] == "full"]
            row_na = subset[subset["configuration"] == "no_uncertainty"]
            row_nb = subset[subset["configuration"] == "no_aging"]

            if row_full.empty or row_na.empty or row_nb.empty:
                st.info(f"No ablation rows for error_pct={error_pct}.")
            else:
                r_full = row_full.iloc[0]
                r_na = row_na.iloc[0]
                r_nb = row_nb.iloc[0]

                abl_names = ["Full formula", "α = 0", "β = 0"]
                abl_colors = ["rgba(29,158,117,0.82)", "rgba(155,89,182,0.82)", "rgba(241,196,15,0.82)"]

                ab1, ab2 = st.columns(2)
                with ab1:
                    with st.container(border=True):
                        st.markdown(f'<div class="chart-title">JCT at {error_pct}% error</div>', unsafe_allow_html=True)
                        st.plotly_chart(
                            make_bar(abl_names, [r_full["jct"], r_na["jct"], r_nb["jct"]], abl_colors),
                            use_container_width=True, config=cfg)
                with ab2:
                    with st.container(border=True):
                        st.markdown(f'<div class="chart-title">Starvation at {error_pct}% error</div>', unsafe_allow_html=True)
                        st.plotly_chart(
                            make_bar(abl_names, [r_full["starvation_pct"], r_na["starvation_pct"], r_nb["starvation_pct"]], abl_colors, "%"),
                            use_container_width=True, config=cfg)
        else:
            st.info(
                "Ablation results not found. Run python sweep.py to generate results/results_ablation.csv"
            )

    with st.expander("Robustness Stress Test (Out-of-Distribution Predictions)"):
        st.write(
            "**Different noise model from the charts above.** Instead of every request getting a small, "
            "symmetric prediction error, a fraction of requests here get a prediction that is essentially "
            "disconnected from their true size -- modeling a predictor that fails to generalize to request "
            "types it has never seen, rather than one that is just imprecise. Fixed at the Normal scenario, "
            "alpha=0.25, beta=16.0."
        )
        ood = load_ood_data()
        if ood is None:
            st.info("OOD test results not found. Run python run_ood_test.py to generate results/ood_test/.")
        else:
            ood_names = ["FCFS", "LTR", "Ours"]
            ood_colors = [fcfs_color, ltr_color, rob_color]
            ood_leg = legend_dots(ood_names, ood_colors)
            ood_el = [f"{e}%" for e in sorted(ood["fcfs"]["error_pct"].unique())]

            oa, ob = st.columns(2)
            with oa:
                with st.container(border=True):
                    st.markdown('<div class="chart-title">JCT vs out-of-distribution rate</div>' + ood_leg, unsafe_allow_html=True)
                    series = [ood[k].sort_values("error_pct")["jct"].tolist() for k in ["fcfs", "ltr", "robust"]]
                    st.plotly_chart(make_line(ood_el, series, ood_names, ood_colors), use_container_width=True, config=cfg)
            with ob:
                with st.container(border=True):
                    st.markdown('<div class="chart-title">Starvation vs out-of-distribution rate</div>' + ood_leg, unsafe_allow_html=True)
                    series = [ood[k].sort_values("error_pct")["starvation_pct"].tolist() for k in ["fcfs", "ltr", "robust"]]
                    st.plotly_chart(make_line(ood_el, series, ood_names, ood_colors, "%"), use_container_width=True, config=cfg)

            with st.container(border=True):
                st.markdown(
                    '<div class="chart-title">JCT change from the 0% baseline</div>' + ood_leg,
                    unsafe_allow_html=True,
                )
                st.caption(
                    "same JCT data as the left chart above, but as % change from each scheduler's own "
                    "0%-OOD value -- FCFS and Robust share an axis with LTR fine here since they're all "
                    "plotted on the same 0-100+ scale, instead of LTR's rise getting flattened next to FCFS's ~180s"
                )
                pct_series = []
                for k in ["fcfs", "ltr", "robust"]:
                    vals = ood[k].sort_values("error_pct")["jct"].tolist()
                    base = vals[0]
                    pct_series.append([100 * (v - base) / base for v in vals])
                st.plotly_chart(make_line(ood_el, pct_series, ood_names, ood_colors, "%"), use_container_width=True, config=cfg)


if __name__ == "__main__":
    main()
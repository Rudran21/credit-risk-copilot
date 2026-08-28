"""
app.py

Streamlit front end.

Two halves, matching the architecture:

  LEFT  the deterministic engine. ECL by stage and scenario, segment
        breakdowns, a stress slider. No LLM anywhere near it. These numbers
        are identical whether or not the agent is configured.

  RIGHT the agent. Natural language in, tool calls out, and the grounding
        check run live on every answer with the full tool trace shown.

The point of putting them side by side is that a reader can see the agent's
figure and the engine's figure for the same thing and confirm they match.

Run locally:
    streamlit run app.py
"""

import os
import sys

import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ---------------------------------------------------------------------------
# Streamlit Cloud exposes secrets through st.secrets, NOT as environment
# variables. agent/runner.py reads os.getenv at import time, so the bridge has
# to happen here, before any agent import. Locally, .env already covers this
# and the loop is a no-op.
# ---------------------------------------------------------------------------
for _k in ("LLM_PROVIDER", "GEMINI_API_KEY", "GOOGLE_API_KEY",
           "ANTHROPIC_API_KEY", "ECL_AGENT_MODEL", "LLM_THROTTLE_SECONDS"):
    try:
        if _k in st.secrets and not os.getenv(_k):
            os.environ[_k] = str(st.secrets[_k])
    except Exception:                                  # no secrets.toml locally
        break

from ecl import config as C          # noqa: E402
from ecl import engine as E          # noqa: E402

st.set_page_config(page_title="Credit Risk Copilot",
                   page_icon=None, layout="wide")


# =============================================================================
# data
# =============================================================================
@st.cache_resource(show_spinner="Loading the ECL engine...")
def boot():
    E.load_state()
    E.warm_cache()
    return True


@st.cache_data(show_spinner=False)
def ecl_all_scenarios(filters_key: str):
    filters = eval(filters_key) if filters_key else None   # noqa: S307
    return {s: E.compute_ecl(filters=filters, scenario=s) for s in C.SCENARIOS}


@st.cache_data(show_spinner=False)
def ecl_by_grade(scenario: str):
    rows = []
    for g in ["A", "B", "C", "D", "E", "F", "G"]:
        r = E.compute_ecl(filters={"grade": [g]}, scenario=scenario)
        if r.get("n_loans"):
            rows.append({"Grade": g, "EAD": r["ead"], "ECL": r["ecl_total"],
                         "Coverage %": r["coverage_pct"]})
    return pd.DataFrame(rows)


def money(x, dp=1):
    return f"${x/1e6:,.{dp}f}m"


# =============================================================================
# sidebar
# =============================================================================
def sidebar():
    st.sidebar.header("Portfolio filter")
    grades = st.sidebar.multiselect(
        "Origination grade", ["A", "B", "C", "D", "E", "F", "G"], default=[])
    stages = st.sidebar.multiselect("IFRS 9 stage", ["1", "2", "3"], default=[])
    terms = st.sidebar.multiselect("Term (months)", ["36", "60"], default=[])

    f = {}
    if grades:
        f["grade"] = grades
    if stages:
        f["stage"] = stages
    if terms:
        f["term"] = terms

    st.sidebar.divider()
    st.sidebar.caption("**Model parameters**")
    st.sidebar.caption(f"LGD (portfolio): {C.LGD_PORTFOLIO:.4f}")
    st.sidebar.caption(f"SICR relative: PD ratio ≥ {C.SICR_PD_RATIO}x")
    st.sidebar.caption(f"SICR absolute: PD up ≥ {C.SICR_PD_ABS:.0%}")
    st.sidebar.caption(f"DPD backstop: {C.DPD_BACKSTOP} days")
    st.sidebar.caption(f"Reporting date: {C.REPORTING_DATE}")
    st.sidebar.divider()
    st.sidebar.caption(
        "Every parameter above is recorded in `ecl/config.py`, labelled "
        "ESTIMATED or ASSUMED with its provenance.")
    return f or None


# =============================================================================
# engine panel
# =============================================================================
def engine_panel(filters):
    key = repr(filters) if filters else ""
    res = ecl_all_scenarios(key)
    base = res["base"]

    if base.get("n_loans", 0) == 0:
        st.warning("No loans match that filter.")
        return

    weighted = sum(C.SCENARIOS[s]["weight"] * res[s]["ecl_total"]
                   for s in C.SCENARIOS)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Exposure at default", f"${base['ead']/1e9:,.2f}bn")
    c2.metric("Weighted ECL", money(weighted))
    c3.metric("Coverage", f"{weighted/base['ead']*100:.2f}%")
    c4.metric("Loans", f"{base['n_loans']:,}")

    st.subheader("ECL by stage and scenario")

    rows = []
    for stg in [1, 2, 3]:
        k = f"stage_{stg}"
        if k not in base["by_stage"]:
            continue
        row = {"Stage": stg, "Loans": base["by_stage"][k]["n_loans"],
               "EAD": base["by_stage"][k]["ead"]}
        for s in C.SCENARIOS:
            row[s.capitalize()] = res[s]["by_stage"][k]["ecl"]
        row["Coverage % (base)"] = base["by_stage"][k]["coverage_pct"]
        rows.append(row)

    total = {"Stage": "Total", "Loans": base["n_loans"], "EAD": base["ead"]}
    for s in C.SCENARIOS:
        total[s.capitalize()] = res[s]["ecl_total"]
    total["Coverage % (base)"] = base["coverage_pct"]
    rows.append(total)

    df = pd.DataFrame(rows)
    st.dataframe(
        df.style.format({
            "EAD": "${:,.0f}", "Loans": "{:,}",
            **{s.capitalize(): "${:,.0f}" for s in C.SCENARIOS},
            "Coverage % (base)": "{:.2f}%"}),
            use_container_width=True, hide_index=True, height=180)

    st.caption(
        "Stage 1 is 12-month ECL. Stage 2 is lifetime. Stage 3 is "
        "credit-impaired, so PD is 1 by definition and coverage equals LGD. "
        "Scenario weights: " +
        ", ".join(f"{s} {C.SCENARIOS[s]['weight']:.0%}" for s in C.SCENARIOS) + ".")

    # ---------------------------------------------------------- by grade
    st.subheader("By origination grade")
    scen = st.radio("Scenario", list(C.SCENARIOS), horizontal=True,
                    key="grade_scen")
    gdf = ecl_by_grade(scen)
    left, right = st.columns([3, 2])
    with left:
        st.dataframe(
            gdf.style.format({"EAD": "${:,.0f}", "ECL": "${:,.0f}",
                              "Coverage %": "{:.2f}%"}),
            use_container_width=True, hide_index=True)
    with right:
        st.bar_chart(gdf.set_index("Grade")["Coverage %"], height=280)

    # ------------------------------------------------------------ stress
    st.subheader("Ad-hoc stress")
    s1, s2 = st.columns(2)
    pd_scalar = s1.slider("Default hazard multiplier", 0.5, 3.0, 1.0, 0.05)
    lgd_addon = s2.slider("LGD add-on", 0.0, 0.08, 0.0, 0.005)

    if abs(pd_scalar - 1.0) > 1e-9 or lgd_addon > 0:
        with st.spinner("Running the shock..."):
            r = E.run_stress(pd_scalar=pd_scalar, lgd_addon=lgd_addon,
                             filters=filters)
        a, b, c = st.columns(3)
        a.metric("Base ECL", money(r["base_ecl"]))
        b.metric("Stressed ECL", money(r["stressed_ecl"]),
                 f"{r['delta_pct']:+.1f}%")
        c.metric("Stressed coverage", f"{r['stressed_coverage_pct']:.2f}%")
        st.caption(
            f"LGD used: {r['shock']['lgd_used']:.4f}. Base LGD is already "
            f"{C.LGD_PORTFOLIO:.3f}, close to its ceiling, so stress in this "
            f"portfolio acts almost entirely through PD rather than severity.")
    else:
        st.caption("Move a slider to run a shock outside the three named "
                   "scenarios.")


# =============================================================================
# agent panel
# =============================================================================
def agent_panel():
    try:
        from agent import guardrail
        from agent.runner import provider_info, run_agent
    except Exception as exc:                          # noqa: BLE001
        st.error(f"Agent layer unavailable: {exc}")
        return

    try:
        info = provider_info()
        st.caption(f"Model: `{info['model']}` via {info['provider']}")
    except Exception:                                 # noqa: BLE001
        pass

    st.caption(
        "The agent cannot compute. Every figure it states must come from a "
        "tool call, and the grounding check below verifies that on each "
        "answer.")

    if "chat" not in st.session_state:
        st.session_state.chat = []

    for m in st.session_state.chat:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])
            if m.get("guard"):
                render_guard(m["guard"], m.get("trace", []))

    prompt = st.chat_input("Ask about the provision...")

    st.caption("Try: which grade drives the provision under adverse? / "
               "what happens if hazards rise 50%? / why is loan 66310712 in "
               "Stage 2?")

    if not prompt:
        return

    st.session_state.chat.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Calling tools..."):
            try:
                out = run_agent(prompt)
            except Exception as exc:                  # noqa: BLE001
                msg = str(exc)
                if "429" in msg or "RESOURCE_EXHAUSTED" in msg:
                    st.error("Rate limited by the LLM provider. The engine "
                             "panel on the left is unaffected.")
                else:
                    st.error(f"{type(exc).__name__}: {msg[:300]}")
                return

        g = guardrail.check(out["answer"], out["session"])
        st.markdown(out["answer"])
        render_guard(g, out["session"].trace())

    st.session_state.chat.append({
        "role": "assistant", "content": out["answer"],
        "guard": g, "trace": out["session"].trace()})


def render_guard(g, trace):
    if g["passed"]:
        st.success(
            f"Grounded. {g['n_figures']} figures, all traced to "
            f"{g['n_tool_calls']} tool calls. "
            f"Citations on {g['citation_coverage']:.0%}.")
    else:
        st.error(
            f"{g['n_ungrounded']} of {g['n_figures']} figures could not be "
            f"traced to any tool call.")
        for u in g["ungrounded"]:
            st.code(f"{u['text']}   in: {u['context']}", language=None)

    with st.expander(f"Tool trace ({len(trace)} calls)"):
        for c in trace:
            st.code(f"[{c['call_id']}] {c['tool']}({c['arguments']})",
                    language="python")


# =============================================================================
# main
# =============================================================================
def main():
    boot()

    st.title("Credit Risk Copilot")
    st.caption(
        "IFRS 9 expected credit loss on a US consumer loan book. "
        "A deterministic engine computes every figure; the agent only "
        "orchestrates tools and explains what they return.")

    filters = sidebar()
    if filters:
        st.info(f"Filtered: {filters}")

    left, right = st.columns([3, 2], gap="large")
    with left:
        engine_panel(filters)
    with right:
        st.subheader("Ask the copilot")
        agent_panel()

    st.divider()
    st.caption(
        "Data: LendingClub accepted loans 2007-2018 (public, CC0). "
        "PD is a competing-risks term structure fitted on fully seasoned "
        "vintages; LGD is estimated from realised recoveries; staging uses a "
        "dual relative/absolute SICR test. Scenario parameters are "
        "judgemental and disclosed as such. See the model validation note "
        "for limitations.")


if __name__ == "__main__":
    main()

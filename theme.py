"""
theme.py -- design system for the app.

Direction: a land-record / survey-sheet feel. Ink-navy type, an indigo accent for actions, a brick-red
"registration stamp" for emphasis and a survey-green for gains. Hairline rules instead of heavy cards.
The one distinctive element is the survey RULER that shows a valuation and its calibrated range.
Colours adapt to light/dark through CSS variables.
"""
from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

from utils import fmt_compact, fmt_inr

INK, ACCENT, STAMP, SURVEY, GOLD, MUTED = "#16213A", "#2B3A8C", "#B83A2F", "#2F7D6B", "#B8892B", "#6B7488"
SEQ = [ACCENT, STAMP, SURVEY, GOLD, "#6C4F9E", "#2A8FA3"]

PAGES: dict = {}  # filled by app.py so pages can link to each other

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,500;12..96,700&family=IBM+Plex+Sans:wght@400;500;600&display=swap');
:root{--ink:#16213A;--accent:#2B3A8C;--stamp:#B83A2F;--survey:#2F7D6B;--rule:rgba(22,33,58,.16);--soft:rgba(43,58,140,.06);--muted:#6B7488;--band90:rgba(43,58,140,.14);--band80:rgba(43,58,140,.30)}
@media (prefers-color-scheme:dark){:root{--ink:#E8ECF5;--accent:#9DB0FF;--stamp:#FF8F82;--survey:#6CCDB4;--rule:rgba(255,255,255,.18);--soft:rgba(157,176,255,.08);--muted:#9AA3B8;--band90:rgba(157,176,255,.16);--band80:rgba(157,176,255,.34)}}
h1,h2,h3{font-family:'Bricolage Grotesque',system-ui,sans-serif!important;letter-spacing:-.01em}
.vz{font-family:'IBM Plex Sans',system-ui,sans-serif;color:var(--ink)}
.vz-hero{padding:.4rem 0 1.1rem;border-bottom:1px solid var(--rule);margin-bottom:1.1rem}
.vz-hero h1{font-size:2.15rem;line-height:1.12;margin:0 0 .35rem;font-weight:700}
.vz-hero p{font-size:1.02rem;max-width:46rem;margin:0;color:var(--muted)}
.vz-chips{display:flex;gap:.45rem;flex-wrap:wrap;margin-top:.7rem}
.vz-chip{font-size:.8rem;padding:.16rem .6rem;border:1px solid var(--rule);border-radius:999px;color:var(--ink)}
.vz-kpi{border-left:3px solid var(--accent);padding:.15rem 0 .15rem .8rem;margin:.2rem 0 .7rem}
.vz-kpi .l{font-size:.82rem;color:var(--muted)}
.vz-kpi .v{font-family:'Bricolage Grotesque',system-ui,sans-serif;font-size:1.55rem;font-weight:700;font-variant-numeric:tabular-nums;line-height:1.2}
.vz-kpi .n{font-size:.8rem;color:var(--muted)}
.vz-kpi.stamp{border-color:var(--stamp)}.vz-kpi.survey{border-color:var(--survey)}
.vz-price{font-family:'Bricolage Grotesque',system-ui,sans-serif;font-size:3rem;font-weight:700;line-height:1;font-variant-numeric:tabular-nums;letter-spacing:-.02em}
.vz-price small{font-size:1rem;font-weight:500;color:var(--muted);margin-left:.5rem;letter-spacing:0}
.vz-ruler{position:relative;margin:1.5rem 0 2.6rem;height:34px}
.vz-track{position:absolute;left:0;right:0;top:15px;height:2px;background:var(--rule)}
.vz-band{position:absolute;top:8px;height:16px;border-radius:2px}
.vz-b90{background:var(--band90)}.vz-b80{background:var(--band80)}
.vz-pin{position:absolute;top:0;width:3px;height:32px;background:var(--stamp);border-radius:1px}
.vz-pin:after{content:'';position:absolute;top:-5px;left:-4px;width:11px;height:11px;border-radius:50%;background:var(--stamp)}
.vz-tick{position:absolute;top:26px;transform:translateX(-50%);font-size:.74rem;color:var(--muted);white-space:nowrap;font-variant-numeric:tabular-nums}
.vz-tick b{display:block;color:var(--ink);font-weight:600}
.vz-topic{border-top:2px solid var(--accent);padding:.7rem 0 .2rem}
.vz-topic h3{margin:0 0 .25rem;font-size:1.25rem}
.vz-topic p{margin:0 0 .5rem;color:var(--muted);font-size:.95rem}
.vz-note{border-left:3px solid var(--stamp);padding:.4rem .8rem;background:var(--soft);font-size:.9rem;margin:.6rem 0}
.stTabs [data-baseweb="tab"]{font-weight:500}
.stButton>button,.stDownloadButton>button{border-radius:6px}
</style>
"""


def inject():
    st.markdown(CSS, unsafe_allow_html=True)


def persist(prefix: str):
    """Streamlit drops widget state for widgets not drawn in the last run; keep ours when the user changes page."""
    for k in list(st.session_state.keys()):
        if isinstance(k, str) and k.startswith(prefix):
            st.session_state[k] = st.session_state[k]


def hero(title: str, subtitle: str, chips: list | None = None):
    c = "".join(f"<span class='vz-chip'>{x}</span>" for x in (chips or []))
    st.markdown(f"<div class='vz vz-hero'><h1>{title}</h1><p>{subtitle}</p>{'<div class=vz-chips>' + c + '</div>' if c else ''}</div>",
                unsafe_allow_html=True)


def kpi(label: str, value: str, note: str = "", tone: str = ""):
    st.markdown(f"<div class='vz vz-kpi {tone}'><div class='l'>{label}</div><div class='v'>{value}</div>"
                f"{'<div class=n>' + note + '</div>' if note else ''}</div>", unsafe_allow_html=True)


def note(text: str):
    st.markdown(f"<div class='vz vz-note'>{text}</div>", unsafe_allow_html=True)


def ruler(price: float, lo90: float, hi90: float, lo80: float, hi80: float):
    """Survey-ruler: the memorable element. Shows the point value, the 80% band and the 90% band on one scale."""
    span_lo, span_hi = lo90 - 0.04 * (hi90 - lo90), hi90 + 0.04 * (hi90 - lo90)
    pct = lambda v: (v - span_lo) / (span_hi - span_lo) * 100
    ticks = [(lo90, "90% low"), (price, "estimate"), (hi90, "90% high")]
    t = "".join(f"<div class='vz-tick' style='left:{pct(v):.2f}%'><b>{fmt_compact(v)}</b>{lab}</div>" for v, lab in ticks)
    st.markdown(
        f"<div class='vz vz-ruler'><div class='vz-track'></div>"
        f"<div class='vz-band vz-b90' style='left:{pct(lo90):.2f}%;width:{pct(hi90) - pct(lo90):.2f}%'></div>"
        f"<div class='vz-band vz-b80' style='left:{pct(lo80):.2f}%;width:{pct(hi80) - pct(lo80):.2f}%'></div>"
        f"<div class='vz-pin' style='left:{pct(price):.2f}%'></div>{t}</div>", unsafe_allow_html=True)


def headline_price(price: float, sub: str = ""):
    st.markdown(f"<div class='vz'><div class='vz-price'>{fmt_inr(price)}<small>{sub}</small></div></div>", unsafe_allow_html=True)


def topic(title: str, body: str):
    st.markdown(f"<div class='vz vz-topic'><h3>{title}</h3><p>{body}</p></div>", unsafe_allow_html=True)


def learn(title: str, md: str, expanded: bool = False):
    with st.expander(title, expanded=expanded):
        st.markdown(md)


# ───────────────────────── Plotly helpers ─────────────────────────
def style(fig: go.Figure, height: int = 380, legend: bool = True) -> go.Figure:
    fig.update_layout(height=height, margin=dict(l=8, r=8, t=36, b=8), showlegend=legend, colorway=SEQ,
                      font=dict(family="IBM Plex Sans, system-ui, sans-serif", size=13),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", hovermode="closest",
                      legend=dict(orientation="h", y=-0.18, x=0), title=dict(font=dict(size=15), x=0, xanchor="left"))
    fig.update_xaxes(showgrid=False, zeroline=False, linecolor="rgba(128,128,128,.35)")
    fig.update_yaxes(gridcolor="rgba(128,128,128,.18)", zeroline=False)
    return fig


def show(fig: go.Figure, key: str | None = None, **kw):
    return st.plotly_chart(fig, width="stretch", key=key, config={"displaylogo": False, "displayModeBar": "hover",
                           "modeBarButtonsToRemove": ["lasso2d", "select2d", "autoScale2d"]}, **kw)

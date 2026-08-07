from __future__ import annotations

import html
import math
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from market_oracle.backtest import walk_forward_backtest
from market_oracle.auto_forward import load_automation_status
from market_oracle.catalog import (
    CATEGORIES, CRYPTO, CRYPTO_CATEGORIES, ETF_CATEGORIES, category_options,
    crypto_category_options, crypto_options, etf_options,
)
from market_oracle.data import download_history, download_profile
from market_oracle.engine import analyze_asset, scan_market_multi, signal_label
from market_oracle.forward import load_forward_cockpit
from market_oracle.journal import (
    journal_summary, load_journal, paper_portfolio, record_snapshot_signals, refresh_journal_results,
)
from market_oracle.monitor import default_universe, load_snapshot, snapshot_is_stale
from market_oracle.search import search_assets
from market_oracle.signals import DEFAULT_SIGNAL_THRESHOLD


st.set_page_config(page_title="MarketScope PRO", page_icon="📈", layout="wide")
st.markdown("""
<style>
    :root {
        --bg-0: #050816;
        --bg-1: #08111f;
        --bg-2: rgba(13, 21, 38, .88);
        --panel: rgba(13, 21, 38, .72);
        --panel-strong: rgba(15, 25, 45, .92);
        --line: rgba(148, 163, 184, .18);
        --line-strong: rgba(56, 189, 248, .34);
        --text: #f5f7ff;
        --muted: #9aa8c7;
        --cyan: #38bdf8;
        --blue: #2563eb;
        --violet: #6366f1;
        --green: #22c55e;
        --red: #fb7185;
        --amber: #fbbf24;
    }

    header[data-testid="stHeader"],
    [data-testid="stToolbar"],
    [data-testid="stDecoration"],
    #MainMenu,
    footer {
        display: none !important;
    }

    .stApp {
        background:
            radial-gradient(circle at 85% 0%, rgba(14, 165, 233, .10), transparent 30rem),
            radial-gradient(circle at 0% 12%, rgba(99, 102, 241, .06), transparent 25rem),
            linear-gradient(180deg, var(--bg-0) 0%, var(--bg-1) 54%, #050711 100%);
        color: var(--text);
    }

    .block-container {padding-top: 1.15rem; padding-bottom: 2.4rem; max-width: 1480px;}
    h1, h2, h3 {letter-spacing: -.04em; color: var(--text);}
    p, li, label, span {color: inherit;}
    hr {border-color: rgba(148, 163, 184, .14);}

    .ms-topbar {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 18px;
        padding: 2px 0 16px;
        margin: 0 0 2px;
        border-bottom: 1px solid rgba(148, 163, 184, .16);
        background: transparent;
        box-shadow: none;
    }
    .ms-brand {
        display:flex;
        gap: 11px;
        align-items:center;
        min-width: 0;
    }
    .ms-logo {
        width: 34px;
        height: 34px;
        display:grid;
        place-items:center;
        border-radius: 10px;
        color: #e0f2fe;
        background: rgba(14, 165, 233, .12);
        border: 1px solid rgba(56, 189, 248, .26);
        font-size: .84rem;
        font-weight: 900;
        letter-spacing: .02em;
    }
    .ms-brand h1 {
        margin: 0;
        font-size: 1.34rem;
        line-height: 1;
        letter-spacing: -.045em;
    }
    .ms-brand p {
        margin: 5px 0 0;
        color: var(--muted);
        font-size: .78rem;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }
    .ms-status-strip {
        display:flex;
        align-items:center;
        gap: 7px;
        flex-wrap: wrap;
        justify-content:flex-end;
    }
    .ms-chip {
        display:inline-flex;
        align-items:center;
        gap: 7px;
        padding: 7px 10px;
        border-radius: 999px;
        border: 1px solid rgba(148, 163, 184, .18);
        background: rgba(15, 23, 42, .48);
        color: #d7e2f7;
        font-size: .74rem;
        font-weight: 850;
        letter-spacing: .01em;
    }
    .ms-led {
        display:inline-block;
        width: 7px;
        height: 7px;
        border-radius:999px;
        background: var(--green);
        box-shadow: 0 0 10px rgba(34,197,94,.65);
    }
    @media (max-width: 920px) {
        .ms-topbar {align-items:flex-start; flex-direction:column;}
        .ms-status-strip {justify-content:flex-start;}
    }

    [data-testid="stMetric"] {
        position: relative;
        overflow: hidden;
        background: linear-gradient(145deg, rgba(15, 23, 42, .90), rgba(8, 13, 25, .78));
        border: 1px solid var(--line);
        padding: 14px 15px;
        border-radius: 14px;
        box-shadow: 0 10px 30px rgba(0,0,0,.18), inset 0 1px 0 rgba(255,255,255,.035);
    }
    [data-testid="stMetric"]::before {
        content: "";
        position: absolute;
        inset: 0 0 auto 0;
        height: 2px;
        background: linear-gradient(90deg, var(--cyan), transparent);
        opacity: .55;
    }
    [data-testid="stMetricLabel"] p {
        color: var(--muted);
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: .06em;
        font-size: .75rem;
    }
    [data-testid="stMetricValue"] {color: var(--text);}
    [data-testid="stMetricDelta"] svg {filter: drop-shadow(0 0 8px rgba(34,211,238,.35));}

    .pro-card {
        min-height: 132px;
        padding: 18px 19px;
        border: 1px solid rgba(148, 163, 184, .16);
        border-radius: 16px;
        background: linear-gradient(145deg, rgba(15, 23, 42, .76), rgba(8, 13, 25, .70));
        box-shadow: 0 12px 32px rgba(0, 0, 0, .18);
        transform: translateY(0) scale(1);
        transition: transform .18s ease, border-color .18s ease, box-shadow .18s ease, background .18s ease;
        will-change: transform;
    }
    .pro-card:hover {
        transform: translateY(-3px) scale(1.015);
        border-color: rgba(56, 189, 248, .38);
        background: linear-gradient(145deg, rgba(17, 28, 50, .84), rgba(8, 13, 25, .76));
        box-shadow: 0 18px 44px rgba(0, 0, 0, .26), 0 0 0 1px rgba(56,189,248,.06) inset;
    }
    .pro-card small {
        display:block;
        margin: 0 0 12px;
        color: var(--cyan);
        font-size: .70rem;
        font-weight: 900;
        letter-spacing: .12em;
        text-transform: uppercase;
    }
    .pro-card h3 {margin: 0 0 9px; color: #f8fbff; font-size: 1.28rem;}
    .pro-card p {margin: 0; color: var(--muted); line-height: 1.45; font-size: .90rem;}
    .feature-grid {
        display: grid;
        grid-template-columns: repeat(5, minmax(0, 1fr));
        gap: 14px;
        margin: 16px 0 18px;
    }
    @media (max-width: 1200px) {
        .feature-grid {grid-template-columns: repeat(3, minmax(0, 1fr));}
    }
    @media (max-width: 760px) {
        .feature-grid {grid-template-columns: 1fr;}
    }
    .ms-note {
        min-height: 2.45rem;
        display: flex;
        align-items: center;
        padding: 8px 11px;
        border-radius: 12px;
        border: 1px solid rgba(34, 211, 238, .18);
        background: rgba(15, 23, 42, .40);
        color: var(--muted);
        font-size: .84rem;
        line-height: 1.32;
    }
    .muted {opacity: .74;}

    .command-hero {
        position: relative;
        overflow: hidden;
        display: grid;
        grid-template-columns: minmax(0, 1.28fr) minmax(320px, .72fr);
        gap: 20px;
        padding: 24px;
        margin: 8px 0 22px;
        border: 1px solid rgba(56, 189, 248, .18);
        border-radius: 22px;
        background:
            radial-gradient(circle at 12% 0%, rgba(99, 102, 241, .18), transparent 22rem),
            radial-gradient(circle at 100% 10%, rgba(14, 165, 233, .12), transparent 20rem),
            linear-gradient(145deg, rgba(15, 23, 42, .82), rgba(5, 8, 22, .78));
        box-shadow: 0 24px 70px rgba(0,0,0,.28), inset 0 1px 0 rgba(255,255,255,.045);
    }
    .command-hero::after {
        content: "";
        position: absolute;
        inset: auto -8% -45% 38%;
        height: 240px;
        background: radial-gradient(circle, rgba(56,189,248,.16), transparent 68%);
        pointer-events: none;
    }
    .command-kicker {
        display:inline-flex;
        align-items:center;
        gap: 8px;
        padding: 7px 10px;
        border-radius: 999px;
        color: #bae6fd;
        border: 1px solid rgba(56,189,248,.24);
        background: rgba(14, 165, 233, .08);
        font-size: .72rem;
        font-weight: 900;
        letter-spacing: .10em;
        text-transform: uppercase;
    }
    .command-title {
        margin: 15px 0 10px;
        font-size: clamp(2.05rem, 4vw, 4.25rem);
        line-height: .96;
        letter-spacing: -.065em;
        color: #f8fafc;
        font-weight: 900;
    }
    .command-hero p {
        margin: 0;
        max-width: 820px;
        color: #aab7d5;
        font-size: 1.02rem;
        line-height: 1.55;
    }
    .hero-badges {
        display: flex;
        flex-wrap: wrap;
        gap: 9px;
        margin-top: 17px;
    }
    .hero-badge {
        display:inline-flex;
        align-items:center;
        gap: 8px;
        padding: 9px 12px;
        border-radius: 999px;
        border: 1px solid rgba(148, 163, 184, .17);
        background: rgba(15, 23, 42, .50);
        color: #dbeafe;
        font-size: .82rem;
        font-weight: 850;
    }
    .proof-panel {
        position: relative;
        z-index: 1;
        align-self: stretch;
        display: flex;
        flex-direction: column;
        justify-content: center;
        min-width: 0;
        padding: 18px;
        border-radius: 18px;
        border: 1px solid rgba(56, 189, 248, .30);
        background:
            radial-gradient(circle at 88% 8%, rgba(34, 197, 94, .08), transparent 12rem),
            linear-gradient(155deg, rgba(7, 12, 28, .98), rgba(4, 8, 20, .96));
        box-shadow:
            0 22px 60px rgba(0, 0, 0, .34),
            0 0 0 1px rgba(15, 23, 42, .74),
            inset 0 1px 0 rgba(255,255,255,.055);
        backdrop-filter: blur(10px);
    }
    .proof-panel::after {
        content: "";
        position: absolute;
        inset: 0;
        border-radius: inherit;
        pointer-events: none;
        background: linear-gradient(180deg, rgba(255,255,255,.035), transparent 34%);
    }
    .proof-panel h3 {
        margin: 0;
        font-size: .92rem;
        letter-spacing: .10em;
        text-transform: uppercase;
        color: #dbeafe;
        min-width: 0;
    }
    .proof-status-line {
        display:flex;
        align-items:center;
        justify-content:space-between;
        gap: 12px;
        margin-bottom: 16px;
        min-width: 0;
    }
    .proof-status {
        display:inline-flex;
        align-items:center;
        gap: 9px;
        flex: 0 0 auto;
        white-space: nowrap;
        padding: 7px 12px;
        border-radius: 999px;
        border: 1px solid rgba(34, 197, 94, .24);
        background: rgba(34, 197, 94, .08);
        color: #bbf7d0;
        font-weight: 900;
        font-size: .80rem;
    }
    .proof-status.warn {
        border-color: rgba(251,191,36,.32);
        background: rgba(251,191,36,.09);
        color: #fde68a;
    }
    .proof-status.bad {
        border-color: rgba(251,113,133,.34);
        background: rgba(251,113,133,.09);
        color: #fecdd3;
    }
    .proof-grid {
        display:grid;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: 11px;
        min-width: 0;
    }
    .proof-stat {
        position: relative;
        z-index: 1;
        min-width: 0;
        padding: 13px;
        border-radius: 14px;
        border: 1px solid rgba(148, 163, 184, .18);
        background: linear-gradient(160deg, rgba(15, 23, 42, .94), rgba(9, 14, 30, .90));
        box-shadow: inset 0 1px 0 rgba(255,255,255,.035);
    }
    .proof-stat small {
        display:block;
        color: #94a3b8;
        font-weight: 850;
        letter-spacing: .08em;
        text-transform: uppercase;
        font-size: .66rem;
    }
    .proof-stat strong {
        display:block;
        margin-top: 6px;
        color: #f8fafc;
        font-size: clamp(1.02rem, 1.35vw, 1.22rem);
        letter-spacing: -.035em;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }
    .dashboard-grid {
        display:grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        gap: 14px;
        margin: 16px 0 22px;
    }
    .dashboard-card {
        min-height: 122px;
        padding: 17px;
        border-radius: 17px;
        border: 1px solid rgba(148, 163, 184, .14);
        background: linear-gradient(145deg, rgba(15,23,42,.74), rgba(7,11,24,.66));
        box-shadow: 0 14px 34px rgba(0,0,0,.18);
    }
    .dashboard-card small {
        display:block;
        color: #94a3b8;
        font-size: .70rem;
        font-weight: 900;
        letter-spacing: .11em;
        text-transform: uppercase;
    }
    .dashboard-card h3 {
        margin: 8px 0 6px;
        font-size: 1.55rem;
        letter-spacing: -.055em;
    }
    .dashboard-card p {
        margin: 0;
        color: #9aa8c7;
        font-size: .88rem;
        line-height: 1.42;
    }
    .daily-brief {
        display: grid;
        grid-template-columns: minmax(0, 1.35fr) minmax(320px, .65fr);
        gap: 14px;
        margin: 18px 0 22px;
    }
    .brief-main, .brief-side {
        border-radius: 20px;
        border: 1px solid rgba(56, 189, 248, .18);
        background:
            radial-gradient(circle at 0% 0%, rgba(14, 165, 233, .10), transparent 18rem),
            linear-gradient(150deg, rgba(15, 23, 42, .86), rgba(5, 10, 23, .78));
        box-shadow: 0 18px 42px rgba(0,0,0,.20), inset 0 1px 0 rgba(255,255,255,.04);
    }
    .brief-main {
        padding: 20px;
    }
    .brief-main small {
        display:block;
        margin-bottom: 9px;
        color: #93c5fd;
        font-size: .72rem;
        font-weight: 950;
        letter-spacing: .11em;
        text-transform: uppercase;
    }
    .brief-main h3 {
        margin: 0 0 10px;
        color: #f8fafc;
        font-size: clamp(1.35rem, 2vw, 2.05rem);
        line-height: 1.08;
        letter-spacing: -.045em;
    }
    .brief-main p {
        margin: 0;
        color: #aab7d5;
        font-size: .96rem;
        line-height: 1.55;
    }
    .brief-risk {
        margin-top: 14px;
        padding: 12px 13px;
        border-radius: 14px;
        border: 1px solid rgba(251, 191, 36, .20);
        background: rgba(251, 191, 36, .07);
        color: #fde68a;
        font-size: .88rem;
        line-height: 1.42;
    }
    .brief-side {
        padding: 14px;
        display: grid;
        gap: 10px;
    }
    .brief-row {
        padding: 12px;
        border-radius: 14px;
        border: 1px solid rgba(148, 163, 184, .14);
        background: rgba(8, 13, 28, .76);
    }
    .brief-row small {
        display:block;
        color: #94a3b8;
        font-size: .66rem;
        font-weight: 900;
        letter-spacing: .09em;
        text-transform: uppercase;
    }
    .brief-row strong {
        display:block;
        margin-top: 5px;
        color: #f8fafc;
        font-size: 1.04rem;
        line-height: 1.24;
    }
    .brief-row span {
        display:block;
        margin-top: 4px;
        color: #97a7c7;
        font-size: .82rem;
        line-height: 1.35;
    }
    .signal-brief {
        display: grid;
        grid-template-columns: minmax(0, 1.1fr) minmax(360px, .9fr);
        gap: 16px;
        margin: 18px 0 22px;
    }
    .signal-brief-main, .signal-brief-grid {
        border-radius: 20px;
        border: 1px solid rgba(56, 189, 248, .18);
        background:
            radial-gradient(circle at 0% 0%, rgba(56, 189, 248, .10), transparent 20rem),
            linear-gradient(150deg, rgba(15, 23, 42, .86), rgba(5, 10, 23, .80));
        box-shadow: 0 18px 42px rgba(0,0,0,.20), inset 0 1px 0 rgba(255,255,255,.04);
    }
    .signal-brief-main {
        padding: 20px;
    }
    .signal-brief-main small {
        display:block;
        margin-bottom: 9px;
        color: #93c5fd;
        font-size: .72rem;
        font-weight: 950;
        letter-spacing: .11em;
        text-transform: uppercase;
    }
    .signal-brief-main h3 {
        margin: 0 0 10px;
        color: #f8fafc;
        font-size: clamp(1.25rem, 1.7vw, 1.85rem);
        line-height: 1.12;
        letter-spacing: -.04em;
    }
    .signal-brief-main p {
        margin: 0;
        color: #aab7d5;
        font-size: .95rem;
        line-height: 1.55;
    }
    .signal-brief-note {
        margin-top: 14px;
        padding: 12px 13px;
        border-radius: 14px;
        border: 1px solid rgba(45, 212, 191, .18);
        background: rgba(45, 212, 191, .07);
        color: #a7f3d0;
        font-size: .86rem;
        line-height: 1.42;
    }
    .signal-brief-grid {
        padding: 14px;
        display: grid;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: 10px;
    }
    .signal-brief-card {
        padding: 12px;
        border-radius: 14px;
        border: 1px solid rgba(148, 163, 184, .14);
        background: rgba(8, 13, 28, .76);
    }
    .signal-brief-card small {
        display:block;
        color: #94a3b8;
        font-size: .66rem;
        font-weight: 900;
        letter-spacing: .09em;
        text-transform: uppercase;
    }
    .signal-brief-card strong {
        display:block;
        margin-top: 5px;
        color: #f8fafc;
        font-size: 1.02rem;
        line-height: 1.24;
    }
    .signal-brief-card span {
        display:block;
        margin-top: 4px;
        color: #97a7c7;
        font-size: .82rem;
        line-height: 1.35;
    }
    .setup-cockpit {
        display: grid;
        grid-template-columns: minmax(0, 1.05fr) minmax(360px, .95fr);
        gap: 14px;
        margin: 14px 0 24px;
    }
    .setup-panel, .setup-side {
        border-radius: 18px;
        border: 1px solid rgba(56, 189, 248, .18);
        background:
            radial-gradient(circle at 0% 0%, rgba(99, 102, 241, .10), transparent 18rem),
            linear-gradient(150deg, rgba(15, 23, 42, .82), rgba(5, 10, 23, .76));
        box-shadow: 0 18px 42px rgba(0,0,0,.18), inset 0 1px 0 rgba(255,255,255,.035);
    }
    .setup-panel {
        padding: 18px;
    }
    .setup-panel small {
        display:block;
        margin-bottom: 8px;
        color: #93c5fd;
        font-size: .70rem;
        font-weight: 950;
        letter-spacing: .11em;
        text-transform: uppercase;
    }
    .setup-panel h3 {
        margin: 0 0 10px;
        color: #f8fafc;
        font-size: clamp(1.25rem, 1.7vw, 1.82rem);
        line-height: 1.10;
        letter-spacing: -.045em;
    }
    .setup-panel p {
        margin: 0;
        color: #aab7d5;
        font-size: .95rem;
        line-height: 1.55;
    }
    .setup-note {
        margin-top: 13px;
        padding: 11px 12px;
        border-radius: 13px;
        border: 1px solid rgba(251, 191, 36, .20);
        background: rgba(251, 191, 36, .07);
        color: #fde68a;
        font-size: .86rem;
        line-height: 1.42;
    }
    .setup-side {
        padding: 13px;
        display: grid;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: 10px;
    }
    .setup-tile {
        padding: 12px;
        border-radius: 14px;
        border: 1px solid rgba(148, 163, 184, .14);
        background: rgba(8, 13, 28, .76);
    }
    .setup-tile small {
        display:block;
        color: #94a3b8;
        font-size: .65rem;
        font-weight: 900;
        letter-spacing: .09em;
        text-transform: uppercase;
    }
    .setup-tile strong {
        display:block;
        margin-top: 5px;
        color: #f8fafc;
        font-size: 1.02rem;
        line-height: 1.22;
    }
    .setup-tile span {
        display:block;
        margin-top: 4px;
        color: #97a7c7;
        font-size: .81rem;
        line-height: 1.34;
    }
    .position-strip {
        display:grid;
        grid-template-columns: minmax(0, 1.2fr) repeat(4, minmax(0, .7fr));
        gap: 12px;
        margin: 10px 0 20px;
    }
    .position-tile {
        padding: 15px;
        border-radius: 16px;
        border: 1px solid rgba(56, 189, 248, .16);
        background: rgba(15, 23, 42, .55);
    }
    .position-tile small {
        display:block;
        color: #94a3b8;
        font-size: .68rem;
        font-weight: 900;
        letter-spacing: .10em;
        text-transform: uppercase;
    }
    .position-tile strong {
        display:block;
        margin-top: 6px;
        color: #f8fafc;
        font-size: 1.14rem;
    }
    .next-actions {
        display:grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 14px;
        margin: 16px 0 12px;
    }
    .action-card {
        padding: 18px;
        border-radius: 17px;
        border: 1px solid rgba(148, 163, 184, .14);
        background: linear-gradient(145deg, rgba(15, 23, 42, .70), rgba(8, 13, 25, .64));
        box-shadow: 0 14px 34px rgba(0,0,0,.18);
    }
    .action-card strong {
        display:block;
        margin-bottom: 8px;
        color: #f8fafc;
        font-size: 1.05rem;
    }
    .action-card span {
        color: var(--muted);
        line-height: 1.45;
        font-size: .90rem;
    }
    @media (max-width: 1100px) {
        .command-hero {grid-template-columns: 1fr;}
        .dashboard-grid {grid-template-columns: repeat(2, minmax(0, 1fr));}
        .daily-brief {grid-template-columns: 1fr;}
        .signal-brief {grid-template-columns: 1fr;}
        .setup-cockpit {grid-template-columns: 1fr;}
        .position-strip {grid-template-columns: repeat(2, minmax(0, 1fr));}
        .next-actions {grid-template-columns: 1fr;}
    }
    @media (max-width: 700px) {
        .dashboard-grid, .proof-grid, .position-strip {grid-template-columns: 1fr;}
    }

    div[data-testid="stTabs"] [role="tablist"] {
        gap: 18px;
        padding: 0;
        margin-top: 8px;
        border: 0;
        border-bottom: 1px solid rgba(148, 163, 184, .16);
        border-radius: 0;
        background: transparent;
        backdrop-filter: none;
    }
    div[data-testid="stTabs"] [role="tablist"] button {
        padding: 11px 0 12px;
        border-radius: 0;
        font-weight: 750;
        color: var(--muted);
        background: transparent !important;
        border: 0 !important;
        border-bottom: 2px solid transparent;
        box-shadow: none !important;
        transition: color .16s ease, border-color .16s ease;
    }
    div[data-testid="stTabs"] [role="tablist"] button:hover {
        color: #e0f2fe;
        background: transparent !important;
    }
    div[data-testid="stTabs"] [role="tablist"] button[aria-selected="true"] {
        color: #ffffff;
        background: transparent !important;
        border-bottom: 2px solid var(--cyan) !important;
        box-shadow: none !important;
    }

    .stButton > button, .stDownloadButton > button {
        min-height: 2.45rem;
        border: 1px solid rgba(148, 163, 184, .22);
        border-radius: 12px;
        color: #e5eefc;
        background: rgba(15, 23, 42, .58);
        box-shadow: 0 8px 22px rgba(0, 0, 0, .12), inset 0 1px 0 rgba(255,255,255,.035);
        font-weight: 800;
        letter-spacing: .005em;
    }
    .stButton > button:hover, .stDownloadButton > button:hover {
        transform: translateY(-1px);
        border-color: rgba(56, 189, 248, .46);
        background: rgba(14, 165, 233, .10);
        box-shadow: 0 12px 26px rgba(14,165,233,.09);
    }
    button[kind="primary"] {
        color: #ecfeff !important;
        background: linear-gradient(180deg, rgba(14, 116, 144, .96) 0%, rgba(8, 83, 116, .96) 100%) !important;
        border-color: rgba(56, 189, 248, .48) !important;
        box-shadow: 0 12px 26px rgba(14, 165, 233, .12), 0 0 0 1px rgba(255,255,255,.06) inset !important;
    }
    .stButton > button:disabled {
        opacity: .45;
        background: rgba(100, 116, 139, .30);
        box-shadow: none;
        transform: none;
    }

    [data-testid="stAlert"] {
        border-radius: 14px;
        border: 1px solid rgba(148, 163, 184, .18);
        background: rgba(15, 23, 42, .72);
        box-shadow: 0 12px 35px rgba(0,0,0,.18);
    }
    [data-testid="stDataFrame"] {
        border: 1px solid rgba(148, 163, 184, .16);
        border-radius: 14px;
        overflow: hidden;
        box-shadow: 0 18px 50px rgba(0,0,0,.20);
    }
    [data-testid="stExpander"] {
        border: 1px solid rgba(148, 163, 184, .16);
        border-radius: 14px;
        background: rgba(10, 14, 30, .52);
    }
    input, textarea, [data-baseweb="select"] > div {
        border-radius: 12px !important;
        border-color: rgba(148, 163, 184, .22) !important;
        background-color: rgba(10, 14, 30, .72) !important;
    }
    input:focus, textarea:focus {
        box-shadow: 0 0 0 1px rgba(56, 189, 248, .50), 0 0 20px rgba(14,165,233,.10) !important;
    }
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, var(--blue), var(--cyan));
    }
</style>
""", unsafe_allow_html=True)

if "training_years" not in st.session_state:
    st.session_state["training_years"] = 8
years = int(st.session_state["training_years"])
APP_DIR = Path(__file__).resolve().parent


@st.cache_data(ttl=3600, show_spinner=False)
def cached_analysis(symbol: str, horizons: tuple[int, ...], years: int):
    return analyze_asset(symbol, horizons=horizons, years=years)


@st.cache_data(ttl=3600, show_spinner=False)
def cached_profile(symbol: str):
    return download_profile(symbol)


@st.cache_data(ttl=3600, show_spinner=False)
def cached_search(query: str):
    return search_assets(query)


def start_signal_scan_background() -> None:
    subprocess.Popen(
        [sys.executable, str(APP_DIR / "run_scan_once.py")],
        cwd=str(APP_DIR),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )


def auto_start_signal_scan(snapshot: dict | None, stale: bool) -> bool:
    if os.getenv("MARKETSCOPE_AUTO_SCAN", "1") == "0":
        return False
    if snapshot and snapshot.get("status") == "running":
        return False
    if not stale:
        return False
    signature = "missing"
    if snapshot:
        signature = "|".join(
            str(snapshot.get(key, "")) for key in ("schema_version", "updated_at", "completed", "total")
        )
    if st.session_state.get("_auto_scan_signature") == signature:
        return False
    st.session_state["_auto_scan_signature"] = signature
    start_signal_scan_background()
    return True


def pct(value: float) -> str:
    return f"{value:.1%}"


def compact_number(value) -> str:
    if value is None or not isinstance(value, (int, float)) or not math.isfinite(value):
        return "—"
    for unit, divisor in (("bln", 1e12), ("mld", 1e9), ("mln", 1e6)):
        if abs(value) >= divisor:
            return f"{value / divisor:.2f} {unit}"
    return f"{value:,.0f}"


def clean_text(value, default: str = "—") -> str:
    if value is None or value == "":
        return default
    return html.escape(str(value))


def short_datetime(value, default: str = "—") -> str:
    if not value:
        return default
    text = str(value).replace("T", " ")
    return html.escape(text[:16])


def value_pct(value, default: str = "—") -> str:
    if value is None or not isinstance(value, (int, float)) or not math.isfinite(value):
        return default
    return pct(value)


def signed_pct(value, default: str = "—") -> str:
    if value is None or not isinstance(value, (int, float)) or not math.isfinite(value):
        return default
    return f"{value:+.1%}"


def safe_load_forward_cockpit() -> tuple[dict | None, str | None]:
    try:
        return load_forward_cockpit(), None
    except Exception as exc:
        return None, str(exc)


def safe_load_automation_status() -> tuple[dict | None, str | None]:
    try:
        return load_automation_status(), None
    except Exception as exc:
        return None, str(exc)


def proof_state(cockpit: dict | None, automation: dict | None, cockpit_error: str | None, automation_error: str | None) -> dict:
    if cockpit_error or automation_error:
        return {"label": "Wymaga uwagi", "klass": "bad", "detail": cockpit_error or automation_error or "Błąd statusu"}
    if not cockpit:
        return {"label": "Brak danych", "klass": "warn", "detail": "Forward ledger nie jest jeszcze dostępny"}
    launchd = (automation or {}).get("launchd") or {}
    stored = (automation or {}).get("stored") or {}
    plan = (automation or {}).get("plan") or {}
    problems = cockpit.get("problems") or []
    if problems or launchd.get("privacy_block_detected") or stored.get("automation_status") == "FAILED":
        return {"label": "Wymaga uwagi", "klass": "bad", "detail": (problems[0] if problems else "Problem automatyzacji")}
    if cockpit.get("healthy") and launchd.get("loaded") and str(launchd.get("last_exit_code")) in {"0", "(never exited)", "None"}:
        return {"label": "OK", "klass": "", "detail": f"następny skan: {short_datetime(plan.get('next_planned_run_local'))}"}
    if cockpit.get("healthy"):
        return {"label": "Ledger OK", "klass": "warn", "detail": "Ledger jest zdrowy, sprawdź automatyzację"}
    return {"label": "Niepełny", "klass": "warn", "detail": "Brakuje pełnego pokrycia lub audytu"}


def display_forward_status(value: str | None) -> str:
    labels = {
        "OPEN": "Otwarta",
        "ACCEPTED": "Czeka na wejście",
        "SKIPPED": "Pominięto",
        "CLOSED": "Zamknięta",
        "OBSERVED": "Zauważono",
        "AUDITED": "Audyt",
    }
    return labels.get(str(value or ""), str(value or "—"))


def display_skip_reason(value: str | None) -> str:
    labels = {
        "POSITION_SKIPPED_SYMBOL_OPEN": "symbol jest już otwarty w portfelu forward",
        "POSITION_SKIPPED_NO_FREE_SLOT": "brak wolnego slotu w portfelu",
        "POSITION_SKIPPED_MAX_POSITIONS": "osiągnięto limit pozycji",
        "POSITION_SKIPPED_SAME_DAY_REENTRY": "blokada ponownego wejścia tego samego dnia",
    }
    if not value:
        return "—"
    return labels.get(str(value), str(value))


def prepare_start_observations(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    output = frame.copy()
    if "Status" in output:
        output["Status"] = output["Status"].map(display_forward_status)
    if "Powód pominięcia" in output:
        output["Powód pominięcia"] = output["Powód pominięcia"].map(display_skip_reason)
    return output


def format_price(value, default: str = "—") -> str:
    if value is None or not isinstance(value, (int, float)) or not math.isfinite(value):
        return default
    return f"{value:,.2f}"


def latest_observation(cockpit: dict | None) -> dict:
    observations = (cockpit or {}).get("latest_observations") or []
    return observations[0] if observations else {}


def active_position(cockpit: dict | None) -> dict:
    positions = (cockpit or {}).get("open_positions") or []
    return positions[0] if positions else {}


def build_daily_brief(cockpit: dict | None, automation: dict | None, state: dict) -> dict:
    obs = latest_observation(cockpit)
    pos = active_position(cockpit)
    plan = (automation or {}).get("plan") or {}
    portfolio = (cockpit or {}).get("portfolio") or {}
    latest_audit = (cockpit or {}).get("latest_audit_date") or plan.get("latest_audit_date")
    next_run = plan.get("next_planned_run_local")
    symbol = obs.get("Symbol") or pos.get("Symbol") or "—"
    obs_date = obs.get("Data sygnału") or (cockpit or {}).get("latest_signal_date") or latest_audit
    obs_status = str(obs.get("Status") or "")
    skip_reason_raw = obs.get("Powód pominięcia")
    skip_reason = display_skip_reason(skip_reason_raw)
    probability = value_pct(obs.get("P(wzrost)") if obs else pos.get("P(wzrost)"))
    expected = signed_pct(obs.get("Oczekiwany ruch") if obs else pos.get("Oczekiwany ruch"))
    open_count = portfolio.get("open", 0)
    slots = portfolio.get("slots", 5)
    new_positions = 1 if obs_status in {"ACCEPTED", "OPEN"} else 0
    skipped = 1 if obs_status == "SKIPPED" else 0
    observed = 1 if obs else 0

    if obs_status == "SKIPPED" and "symbol jest już otwarty" in skip_reason and pos:
        headline = f"{symbol} nadal spełnia warunki LONG Candidate v1, ale system nie dubluje pozycji."
        body = (
            f"Ostatni audyt z {obs_date} ponownie zauważył setup na {symbol}. "
            f"Nowa pozycja nie została otwarta, bo {skip_reason}. "
            f"Aktywna pozycja została otwarta {pos.get('Data wejścia')} po {format_price(pos.get('Cena wejścia'))}; "
            f"do planowego rozliczenia zostało około {pos.get('Sesje do wyjścia')} sesji."
        )
        action = "Obserwuj istniejącą pozycję"
    elif obs_status == "SKIPPED":
        headline = f"{symbol} pojawił się w sygnałach, ale nie trafił do portfela."
        body = (
            f"System zauważył setup z P(wzrost) {probability} i oczekiwanym ruchem {expected}, "
            f"ale pozycja została pominięta: {skip_reason}."
        )
        action = "Sprawdź powód pominięcia"
    elif obs_status in {"ACCEPTED", "OPEN"}:
        headline = f"{symbol} trafił do forward testu jako nowa hipoteza."
        body = (
            f"System zaakceptował sygnał z P(wzrost) {probability} i oczekiwanym ruchem {expected}. "
            "To zapis testowy po zamrożonych regułach Candidate v1, nie rekomendacja kupna."
        )
        action = "Śledź wejście i rozliczenie"
    elif pos:
        headline = f"Najważniejsza aktywna hipoteza: {pos.get('Symbol')} w portfelu forward."
        body = (
            f"Pozycja jest otwarta od {pos.get('Data wejścia')} po {format_price(pos.get('Cena wejścia'))}. "
            f"Model zapisał P(wzrost) {value_pct(pos.get('P(wzrost)'))}, a do planowego rozliczenia zostało około "
            f"{pos.get('Sesje do wyjścia')} sesji."
        )
        action = "Monitoruj aktywną hipotezę"
    else:
        headline = "Candidate v1 nie ma teraz aktywnej pozycji."
        body = (
            "To też jest poprawny stan. Zamrożony system może milczeć, jeśli nie widzi wystarczająco mocnego setupu "
            "albo portfel nie powinien przyjmować nowej ekspozycji."
        )
        action = "Czekaj na kolejny audyt"

    risk = (
        "To forward test i paper-performance: sygnał pokazuje hipotezę badawczą, nie gwarancję ruchu ani polecenie kupna. "
        "Najważniejsze jest, że system nie zwiększa ekspozycji na symbol, który już jest w portfelu."
        if pos
        else "To forward test i paper-performance: brak pozycji nie jest błędem, tylko częścią selektywnej strategii."
    )
    return {
        "headline": headline,
        "body": body,
        "risk": risk,
        "rows": [
            ("Ostatni audyt", latest_audit or "—", f"Proof flow: {state.get('label') or '—'}"),
            ("Dzisiejszy sygnał", symbol, f"P(wzrost): {probability} · oczekiwany ruch: {expected}"),
            ("Decyzja", action, f"Nowe pozycje: {new_positions} · pominięte: {skipped} · obserwacje: {observed}"),
            ("Następny skan", short_datetime(next_run), f"Portfel forward: {open_count}/{slots}"),
        ],
    }


def signal_brief_value(row: pd.Series | None, column: str, default: str = "—") -> str:
    if row is None or column not in row:
        return default
    value = row.get(column)
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return default
    if pd.isna(value):
        return default
    return str(value)


def signal_brief_horizon(value) -> str:
    if value is None:
        return "—"
    try:
        return f"{int(float(value))}d"
    except (TypeError, ValueError):
        return str(value)


def classify_signal_errors(errors: dict | None) -> dict[str, dict]:
    provider_no_data: dict[str, str] = {}
    no_data: dict[str, str] = {}
    failed: dict[str, str] = {}
    for symbol, message in (errors or {}).items():
        normalized_symbol = str(symbol)
        text = str(message)
        lowered = text.lower()
        if any(marker in lowered for marker in [
            "brak danych",
            "no price data",
            "possibly delisted",
            "no timezone",
            "quote not found",
            "not found",
        ]):
            provider_no_data[normalized_symbol] = text
            if normalized_symbol.upper().endswith("-USD"):
                no_data[normalized_symbol] = text
        else:
            failed[normalized_symbol] = text
    return {"no_data": no_data, "provider_no_data": provider_no_data, "failed": failed}


def signal_scan_contract(frame: pd.DataFrame, snapshot: dict) -> dict:
    errors = classify_signal_errors(snapshot.get("errors") or {})
    fast_rows = frame[frame["Tryb analizy"].astype(str).eq("FAST")] if "Tryb analizy" in frame else pd.DataFrame()
    ml_rows = frame[frame["Tryb analizy"].astype(str).eq("ML")] if "Tryb analizy" in frame else pd.DataFrame()
    universe_total = int(snapshot.get("universe_total") or snapshot.get("total") or _unique_symbols(frame))
    fast_completed = int(snapshot.get("fast_completed") or (universe_total if not frame.empty else 0))
    ml_attempted = int(snapshot.get("ml_completed") or 0)
    ml_total = int(snapshot.get("ml_total") or 0)
    return {
        "universe_total": universe_total,
        "fast_completed": fast_completed,
        "ranked_symbols": _unique_symbols(frame),
        "ranked_rows": int(len(frame)),
        "fast_rows": int(len(fast_rows)),
        "fast_symbols": _unique_symbols(fast_rows),
        "ml_rows": int(len(ml_rows)),
        "ml_symbols": _unique_symbols(ml_rows),
        "ml_attempted": ml_attempted,
        "ml_total": ml_total,
        "deep_limit": snapshot.get("deep_limit"),
        "no_data": errors["no_data"],
        "provider_no_data": errors["provider_no_data"],
        "failed": errors["failed"],
    }


def with_signal_display_columns(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    if output.empty:
        return output

    def direction_text(row: pd.Series) -> str:
        if str(row.get("Tryb analizy")) == "ML":
            return f"ML P(wzrost): {value_pct(row.get('P(wzrost)'))}"
        return "FAST · bez potwierdzenia ML"

    def move_text(row: pd.Series) -> str:
        expected = row.get("Oczekiwany ruch")
        if str(row.get("Tryb analizy")) == "ML":
            return signed_pct(expected)
        return f"ruch FAST: {signed_pct(expected)}"

    output["Ocena kierunku"] = output.apply(direction_text, axis=1)
    output["Ruch / impet"] = output.apply(move_text, axis=1)
    return output


def build_signal_radar_brief(
    frame: pd.DataFrame,
    snapshot: dict,
    *,
    bullish_labels: set[str],
    bearish_labels: set[str],
) -> dict:
    """Human-readable summary for the Signals tab without changing ranking logic."""
    if frame.empty:
        return {
            "headline": "Radar nie ma jeszcze danych do interpretacji.",
            "body": "Uruchom pełny skan albo poczekaj, aż monitor zapisze pierwsze wiersze rankingu.",
            "note": "To tylko warstwa prezentacji. Nie zmienia modeli, progów ani zapisanego rankingu.",
            "cards": [
                ("Status", "Brak danych", "Ranking nie został jeszcze policzony"),
                ("Lider", "—", "Brak instrumentu do oceny"),
                ("Ryzyko", "—", "Brak danych o ryzyku"),
                ("Następny krok", "Uruchom skan", "Pełny obraz pojawi się po zakończeniu"),
            ],
        }

    sort_columns = [column for column in ["Deep score", "Setup score", "Edge score", "Radar score", "Score"] if column in frame.columns]
    confirmed = frame[
        frame["Ocena"].isin(bullish_labels)
        & frame["Tryb analizy"].astype(str).eq("ML")
    ].copy()
    priority = frame[
        frame["Akcja radaru"].isin(["PRIORYTET DO ANALIZY", "WATCHLIST", "FAST SHORTLIST", "MOMENTUM DO SPRAWDZENIA"])
    ].copy()
    if not confirmed.empty:
        pool = confirmed
        leader_kind = "confirmed"
    elif not priority.empty:
        pool = priority
        leader_kind = "priority"
    else:
        pool = frame.copy()
        leader_kind = "neutral"
    if sort_columns:
        leader = pool.sort_values(sort_columns, ascending=False).iloc[0]
    else:
        leader = pool.iloc[0]

    contract = signal_scan_contract(frame, snapshot)
    status = str(snapshot.get("status") or "unknown")
    if status == "running":
        status_text = f"FAST {contract['fast_completed']}/{contract['universe_total']} · skan trwa"
    elif status == "complete":
        status_text = (
            f"FAST complete · ML rows {contract['ml_rows']} · "
            f"krypto bez danych {len(contract['no_data'])} · błędy {len(contract['failed'])}"
        )
    else:
        status_text = f"Status: {status}"
    symbol = signal_brief_value(leader, "Symbol")
    horizon = signal_brief_horizon(leader.get("Horyzont") if "Horyzont" in leader else None)
    action = signal_brief_value(leader, "Akcja radaru")
    thesis = signal_brief_value(leader, "Teza radaru")
    leader_mode = signal_brief_value(leader, "Tryb analizy")
    probability = value_pct(leader.get("P(wzrost)") if leader_mode == "ML" and "P(wzrost)" in leader else None)
    expected = signed_pct(leader.get("Oczekiwany ruch") if "Oczekiwany ruch" in leader else None)
    quality = signal_brief_value(leader, "Jakość modelu")
    setup_grade = signal_brief_value(leader, "Setup grade")
    hot_symbols = _unique_symbols(frame[frame["Radar momentum"].isin({"PEREŁKA MOMENTUM", "BREAKOUT WATCH", "MOMENTUM WATCH"})])
    confirmed_symbols = _unique_symbols(confirmed)
    bearish_symbols = _unique_symbols(frame[frame["Ocena"].isin(bearish_labels)])
    risk_value = f"{bearish_symbols} alertów" if bearish_symbols else "brak alertów ryzyka"

    if leader_kind == "confirmed":
        headline = f"{symbol} jest teraz najmocniejszym potwierdzonym setupem ML w radarze."
        body = (
            f"Horyzont {horizon}, P(wzrost) {probability}, oczekiwany ruch {expected}. "
            f"Teza systemu: {thesis}. To priorytet do ręcznej analizy, nie automatyczna rekomendacja kupna."
        )
    elif leader_kind == "priority":
        headline = f"{symbol} jest najwyżej w radarze FAST, ale nie ma jeszcze potwierdzenia ML."
        body = (
            f"System oznaczył go jako {action} na horyzoncie {horizon}. "
            f"Setup: {setup_grade}; teza: {thesis}. Liczby w tym trybie są heurystyką FAST, nie skalibrowanym P(wzrost) z ML."
        )
    else:
        headline = "Radar nie ma wyraźnego potwierdzonego lidera."
        body = (
            f"Najwyżej w częściowym rankingu jest {symbol}, ale obecny stan bardziej przypomina obserwację niż gotową hipotezę. "
            "W takim momencie najcenniejsza może być selekcja i odrzucanie szumu."
        )

    note = (
        "Ranking jest jeszcze częściowy — pełny obraz pojawi się po zakończeniu Deep ML."
        if status == "running"
        else "To radar badawczy: pomaga ustalić kolejność analizy, ale nie zastępuje decyzji inwestora."
    )
    leader_detail = (
        f"P(wzrost): {probability} · oczekiwany ruch: {expected}"
        if leader_mode == "ML"
        else f"FAST · bez ML · ruch/impet: {expected}"
    )
    return {
        "headline": headline,
        "body": body,
        "note": note,
        "cards": [
            ("Lider radaru", f"{symbol} · {horizon}", f"{action} · {leader_detail}"),
            ("Potwierdzone ML", f"{confirmed_symbols} symboli", f"ML rows: {contract['ml_rows']} · FAST rows: {contract['fast_rows']} · jakość lidera: {quality}"),
            ("Momentum", f"{hot_symbols} hot movers", "Perełki, breakouty i szybkie ruchy do sprawdzenia"),
            ("Ryzyko / dane", risk_value, status_text),
        ],
    }


def radar_row_number(row: pd.Series, column: str) -> float | None:
    if column not in row:
        return None
    value = row.get(column)
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


def setup_destination(symbol: str, klass: str) -> tuple[str, str]:
    upper = str(symbol).upper()
    label = "Spółki"
    detail = "Kliknij przycisk pod panelem — MarketScope uruchomi pełną analizę tutaj."
    if upper.endswith("-USD"):
        label = "Krypto"
        detail = "Kliknij przycisk pod panelem — MarketScope uruchomi pełną analizę tutaj."
    elif "ETF" in str(klass).upper() or upper in {"SPY", "QQQ", "VOO", "VTI", "IWM", "GLD", "IAU"}:
        label = "ETF-y / Spółki"
        detail = "Kliknij przycisk pod panelem — MarketScope uruchomi pełną analizę tutaj."
    return label, detail


def setup_risk_note(row: pd.Series) -> tuple[str, str]:
    notes: list[str] = []
    mode = signal_brief_value(row, "Tryb analizy")
    quality = signal_brief_value(row, "Jakość modelu")
    rsi = radar_row_number(row, "RSI 14")
    risk_control = radar_row_number(row, "Risk control")
    max_drawdown = radar_row_number(row, "Max drawdown")
    risk_reward = radar_row_number(row, "Risk/reward")
    if mode != "ML":
        notes.append("brak potwierdzenia ML")
    if "BRAK" in quality.upper() or "FAST" in quality.upper():
        notes.append("jakość modelu nie potwierdza przewagi")
    if rsi is not None and rsi >= 72:
        notes.append("RSI wysoko — możliwe przegrzanie")
    elif rsi is not None and rsi <= 30:
        notes.append("RSI nisko — ruch może być nerwowy")
    if risk_reward is not None and risk_reward < 1:
        notes.append("risk/reward poniżej 1")
    if risk_control is not None and risk_control < 45:
        notes.append("słabsza kontrola ryzyka")
    if max_drawdown is not None and max_drawdown <= -0.30:
        notes.append("historycznie duże obsunięcia")
    if not notes:
        return "bez alertu", "Brak dodatkowego alertu według obecnych filtrów. To nadal nie znaczy brak ryzyka rynkowego."
    return "uwaga", " · ".join(dict.fromkeys(notes))


def build_setup_drilldown(row: pd.Series, bullish_labels: set[str]) -> dict:
    symbol = signal_brief_value(row, "Symbol")
    klass = signal_brief_value(row, "Klasa")
    mode = signal_brief_value(row, "Tryb analizy")
    horizon = signal_brief_horizon(row.get("Horyzont") if "Horyzont" in row else None)
    action = signal_brief_value(row, "Akcja radaru")
    thesis = signal_brief_value(row, "Teza radaru")
    grade = signal_brief_value(row, "Setup grade")
    direction = signal_brief_value(row, "Ocena kierunku")
    move = signal_brief_value(row, "Ruch / impet")
    quality = signal_brief_value(row, "Jakość modelu")
    score = radar_row_number(row, "Deep score")
    risk_label, risk_detail = setup_risk_note(row)
    dest_label, dest_detail = setup_destination(symbol, klass)
    confirmed = mode == "ML" and signal_brief_value(row, "Ocena") in bullish_labels

    if confirmed:
        headline = f"{symbol} ma potwierdzony setup ML na horyzoncie {horizon}."
        body = (
            f"Radar widzi {action.lower()} i modelowy kierunek: {direction}. "
            f"Teza: {thesis}. To jest kandydat do pełnej ręcznej analizy, nadal bez gwarancji ruchu."
        )
    elif mode == "ML":
        headline = f"{symbol} ma policzony ML, ale bez wzrostowego potwierdzenia."
        body = (
            f"Model został uruchomiony, lecz ocena nie uzyskała wzrostowego potwierdzenia ML. "
            f"Teza radaru: {thesis}. Najrozsądniej traktować to jako obserwację, nie sygnał."
        )
    else:
        headline = f"{symbol} jest na watchliście FAST — dobry kandydat do sprawdzenia, nie sygnał ML."
        body = (
            f"FAST wskazuje {action.lower()} na horyzoncie {horizon}. "
            f"Teza: {thesis}. Liczby są heurystyką radaru, a nie skalibrowanym P(wzrost) z modelu."
        )

    return {
        "headline": headline,
        "body": body,
        "note": "To briefing badawczy: pomaga ustalić kolejność pracy, ale nie jest rekomendacją kupna ani sprzedaży.",
        "cards": [
            ("Instrument", f"{symbol} · {klass}", f"horyzont {horizon}"),
            ("Tryb", mode, "ML = pełny model; FAST = szybka heurystyka radaru"),
            ("Setup", grade, action),
            ("Kierunek", direction, move),
            ("Ryzyko", risk_label, risk_detail),
            ("Pełna analiza", dest_label, dest_detail),
            ("Jakość", quality, f"Deep score: {score:.0f}" if score is not None else "Deep score: —"),
        ],
    }


def render_setup_cockpit(frame: pd.DataFrame, bullish_labels: set[str]) -> None:
    if frame.empty or "Symbol" not in frame:
        return
    sort_columns = [column for column in ["Deep score", "Setup score", "Radar score", "Edge score"] if column in frame.columns]
    work = frame.copy()
    if sort_columns:
        work = work.sort_values(sort_columns, ascending=False)
    work = work.drop_duplicates("Symbol").head(20).reset_index(drop=True)
    options = {
        f"{row['Symbol']} · {signal_brief_horizon(row.get('Horyzont'))} · {row.get('Tryb analizy', '—')} · {row.get('Akcja radaru', '—')}": idx
        for idx, row in work.iterrows()
    }
    if not options:
        return
    selected = st.selectbox("Rozłóż setup na czynniki", list(options), key="radar_setup_focus")
    row = work.iloc[options[selected]]
    drilldown = build_setup_drilldown(row, bullish_labels)
    cards = "".join(
        '<div class="setup-tile">'
        f"<small>{clean_text(label)}</small>"
        f"<strong>{clean_text(value)}</strong>"
        f"<span>{clean_text(detail)}</span>"
        "</div>"
        for label, value, detail in drilldown["cards"]
    )
    st.markdown(
        '<div class="setup-cockpit">'
        '<div class="setup-panel">'
        "<small>Analiza wybranego setupu</small>"
        f"<h3>{clean_text(drilldown['headline'])}</h3>"
        f"<p>{clean_text(drilldown['body'])}</p>"
        f"<div class=\"setup-note\">⚠️ {clean_text(drilldown['note'])}</div>"
        "</div>"
        f'<div class="setup-side">{cards}</div>'
        "</div>",
        unsafe_allow_html=True,
    )
    symbol = signal_brief_value(row, "Symbol")
    button_key = f"radar_full_analysis_{str(symbol).replace('.', '_').replace('-', '_')}"
    if st.button(f"Uruchom pełną analizę: {symbol}", type="primary", key=button_key, use_container_width=True):
        try:
            with st.spinner(f"Pobieram dane i liczę pełny model dla {symbol}…"):
                result = cached_analysis(symbol, (1, 5, 20, 60), years)
                profile = cached_profile(symbol)
            st.session_state["radar_full_analysis"] = {"result": result, "profile": profile, "years": years}
        except Exception as exc:
            st.error(f"Nie udało się uruchomić pełnej analizy dla {symbol}: {exc}")
    saved = st.session_state.get("radar_full_analysis")
    if saved and saved["result"]["symbol"] == symbol and saved.get("years") == years:
        st.caption("Pełna analiza uruchomiona z radaru — bez przepisywania tickera.")
        render_analysis(saved["result"], saved["profile"])


def render_start_dashboard(
    *,
    snapshot: dict,
    journal: dict,
    universe_size: int,
    cockpit: dict | None,
    automation: dict | None,
    cockpit_error: str | None = None,
    automation_error: str | None = None,
) -> None:
    coverage = (cockpit or {}).get("coverage") or {}
    portfolio = (cockpit or {}).get("portfolio") or {}
    summary = (cockpit or {}).get("summary") or {}
    plan = (automation or {}).get("plan") or {}
    stored = (automation or {}).get("stored") or {}
    launchd = (automation or {}).get("launchd") or {}
    state = proof_state(cockpit, automation, cockpit_error, automation_error)

    requested = coverage.get("requested") or 0
    completed = coverage.get("completed") or 0
    coverage_text = f"{completed}/{requested}" if requested else "—"
    audit_days = (cockpit or {}).get("audit_days", "—")
    signal_days = (cockpit or {}).get("signal_days", "—")
    open_positions = portfolio.get("open", 0)
    slots = portfolio.get("slots", 5)
    free_slots = portfolio.get("free_slots", "—")
    latest_audit = (cockpit or {}).get("latest_audit_date") or plan.get("latest_audit_date")
    latest_signal = (cockpit or {}).get("latest_signal_date")
    next_run = plan.get("next_planned_run_local")
    last_auto = stored.get("automation_status") or "—"
    launchd_text = "aktywny" if launchd.get("loaded") else "nieaktywny"
    launchd_exit = launchd.get("last_exit_code") if launchd.get("last_exit_code") is not None else "—"

    st.markdown(f"""
    <div class="command-hero">
        <div>
            <span class="command-kicker">MarketScope command center</span>
            <div class="command-title">Rynek, sygnały i forward test w jednym miejscu.</div>
            <p>
                Start pokazuje, czy automat po sesji USA wykonał skan, co obserwuje Candidate v1
                i czy portfel testowy ma aktywne pozycje. To nadal warstwa badawcza —
                nie rekomendacja kupna ani sprzedaży.
            </p>
            <div class="hero-badges">
                <span class="hero-badge">🧊 Candidate v1: zamrożony</span>
                <span class="hero-badge">🟢 Automat: {clean_text(launchd_text)}</span>
                <span class="hero-badge">📡 Pokrycie: {clean_text(coverage_text)}</span>
                <span class="hero-badge">🧾 Dni audytu: {clean_text(audit_days)}</span>
            </div>
        </div>
        <div class="proof-panel">
            <div class="proof-status-line">
                <h3>Stan proof</h3>
                <span class="proof-status {state['klass']}"><i class="ms-led"></i>{clean_text(state['label'])}</span>
            </div>
            <div class="proof-grid">
                <div class="proof-stat"><small>Ostatni audyt</small><strong>{clean_text(latest_audit)}</strong></div>
                <div class="proof-stat"><small>Następny skan</small><strong>{short_datetime(next_run)}</strong></div>
                <div class="proof-stat"><small>Portfel testowy</small><strong>{clean_text(open_positions)}/{clean_text(slots)}</strong></div>
                <div class="proof-stat"><small>Automat</small><strong>{clean_text(last_auto)}</strong></div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if cockpit_error:
        st.error(f"Forward cockpit error: {cockpit_error}")
    if automation_error:
        st.warning(f"Automation status error: {automation_error}")
    if state["klass"] == "bad":
        st.error(state["detail"])
    elif state["klass"] == "warn":
        st.warning(state["detail"])
    else:
        st.caption(f"✅ Proof flow zdrowy · {state['detail']}")

    brief = build_daily_brief(cockpit, automation, state)
    brief_rows = "".join(
        '<div class="brief-row">'
        f"<small>{clean_text(label)}</small>"
        f"<strong>{clean_text(value)}</strong>"
        f"<span>{clean_text(detail)}</span>"
        "</div>"
        for label, value, detail in brief["rows"]
    )
    st.markdown(
        '<div class="daily-brief">'
        '<div class="brief-main">'
        "<small>Co dziś jest ważne?</small>"
        f"<h3>{clean_text(brief['headline'])}</h3>"
        f"<p>{clean_text(brief['body'])}</p>"
        f"<div class=\"brief-risk\">⚠️ {clean_text(brief['risk'])}</div>"
        "</div>"
        f'<div class="brief-side">{brief_rows}</div>'
        "</div>",
        unsafe_allow_html=True,
    )

    st.markdown(f"""
    <div class="dashboard-grid">
        <div class="dashboard-card"><small>Audyt forward</small><h3>{clean_text(summary.get('events', 0))}</h3><p>Nieusuwalna historia obserwacji, wejść, pominięć i rozliczeń.</p></div>
        <div class="dashboard-card"><small>Obserwacje sygnałów</small><h3>{clean_text(summary.get('signals', 0))}</h3><p>Ile razy zamrożony system 20D LONG znalazł setup do obserwacji.</p></div>
        <div class="dashboard-card"><small>Portfel testowy</small><h3>{clean_text(open_positions)}/{clean_text(slots)}</h3><p>Aktywne pozycje forward; wolne sloty: {clean_text(free_slots)}.</p></div>
        <div class="dashboard-card"><small>Ostatni sygnał</small><h3>{clean_text(latest_signal)}</h3><p>Najnowszy dzień, w którym system coś zauważył.</p></div>
        <div class="dashboard-card"><small>Instrumenty</small><h3>{clean_text(universe_size)}</h3><p>Szeroki radar MarketScope: akcje, ETF-y i krypto.</p></div>
        <div class="dashboard-card"><small>Skan rynku</small><h3>{clean_text(snapshot.get('completed', 0))}/{clean_text(snapshot.get('total', universe_size))}</h3><p>Ostatni zapisany status radaru: {clean_text(snapshot.get('status', 'offline'))}.</p></div>
        <div class="dashboard-card"><small>Dni z sygnałem</small><h3>{clean_text(signal_days)}</h3><p>Dni forward, w których pojawiła się co najmniej jedna obserwacja.</p></div>
        <div class="dashboard-card"><small>Dziennik</small><h3>{clean_text(journal.get('total', 0))}</h3><p>Osobny paper-performance historycznych sygnałów poza Candidate v1.</p></div>
    </div>
    """, unsafe_allow_html=True)

    st.subheader("Aktywna pozycja w teście forward")
    open_rows = (cockpit or {}).get("open_positions") or []
    if open_rows:
        row = open_rows[0]
        st.markdown(f"""
        <div class="position-strip">
            <div class="position-tile"><small>Pozycja</small><strong>{clean_text(row.get('Symbol'))} · {clean_text(display_forward_status(row.get('Status')).lower())}</strong></div>
            <div class="position-tile"><small>Wejście</small><strong>{clean_text(row.get('Data wejścia'))} @ {clean_text(f"{row.get('Cena wejścia', 0):.2f}" if isinstance(row.get('Cena wejścia'), (int, float)) else row.get('Cena wejścia'))}</strong></div>
            <div class="position-tile"><small>P(wzrost)</small><strong>{value_pct(row.get('P(wzrost)'))}</strong></div>
            <div class="position-tile"><small>Oczekiwany ruch</small><strong>{signed_pct(row.get('Oczekiwany ruch'))}</strong></div>
            <div class="position-tile"><small>Do wyjścia</small><strong>{clean_text(row.get('Sesje do wyjścia'))} sesji</strong></div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.info("Candidate v1 nie ma teraz otwartej pozycji. To poprawny stan — system ma prawo milczeć.")

    latest = pd.DataFrame((cockpit or {}).get("latest_observations") or [])
    if not latest.empty:
        st.subheader("Co system zrobił ostatnio?")
        latest = prepare_start_observations(latest)
        columns = ["Symbol", "Status", "Data sygnału", "P(wzrost)", "Oczekiwany ruch", "Jakość", "Decyzja", "Powód pominięcia"]
        present = [column for column in columns if column in latest.columns]
        st.dataframe(
            latest[present].style.format({"P(wzrost)": "{:.1%}", "Oczekiwany ruch": "{:+.1%}"}, na_rep="—"),
            use_container_width=True,
            hide_index=True,
        )

    st.subheader("Co warto sprawdzić dalej?")
    st.markdown("""
    <div class="next-actions">
        <div class="action-card"><strong>Forward</strong><span>Sprawdź pełną historię audytów, pozycji i decyzji Candidate v1.</span></div>
        <div class="action-card"><strong>Sygnały</strong><span>Zobacz ranking setupów: hot movers, swing, trend i ryzyko spadku.</span></div>
        <div class="action-card"><strong>Analiza instrumentu</strong><span>Wejdź w Spółki, ETF-y albo Krypto i zobacz pełną prognozę dla symbolu.</span></div>
    </div>
    """, unsafe_allow_html=True)

    with st.expander("Jak czytać prognozę?"):
        st.markdown("""
        - **P(wzrost)** mówi, jak często model oczekuje ceny wyższej po danym horyzoncie.
        - **AUC** mierzy przewagę kierunkową poza próbką; około 0,50 oznacza brak przewagi.
        - **Brier** ocenia jakość prawdopodobieństw; mniej znaczy lepiej.
        - **Forward test** to realne, append-only obserwacje po zamrożeniu Candidate v1 — tego nie tuningujemy po fakcie.
        """)


def profile_name(profile: dict, fallback: str) -> str:
    return profile.get("longName") or profile.get("shortName") or fallback


def model_mix(weights: dict | None) -> str:
    if not weights:
        return "—"
    names = {"linear": "linear", "boosting": "boosting", "extra_trees": "ExtraTrees"}
    ordered = sorted(weights.items(), key=lambda item: item[1], reverse=True)
    return " · ".join(f"{names.get(name, name)} {weight:.0%}" for name, weight in ordered)


def horizon_text(horizon: int, crypto: bool) -> str:
    unit = "dni" if crypto else "sesji"
    names = {1: "1 dzień" if crypto else "1 sesja", 5: "5 dni" if crypto else "5 sesji", 20: f"20 {unit}", 60: f"60 {unit}"}
    return names.get(horizon, f"{horizon} {unit}")


def best_confirmed_forecast(forecasts: dict[int, dict]) -> tuple[int, dict] | tuple[None, None]:
    confirmed = [
        (horizon, forecast) for horizon, forecast in forecasts.items()
        if not forecast["quality"].startswith("NISKA")
    ]
    if not confirmed:
        return None, None
    return max(
        confirmed,
        key=lambda item: (
            item[1]["quality"] == "WYSOKA",
            abs(item[1]["probability_up"] - 0.5),
            item[0],
        ),
    )


def aggregate_model_view(result: dict) -> dict:
    forecasts = result["forecasts"]
    best_horizon, best = best_confirmed_forecast(forecasts)
    five_day = forecasts.get(5) or next(iter(forecasts.values()))
    technical = result["technical"]
    trend_points = sum([
        technical["return_20d"] > 0, technical["rsi_14"] >= 50,
        technical["above_sma_50"], technical["above_sma_200"],
    ])
    trend_label = "POZYTYWNY" if trend_points >= 3 else ("NEGATYWNY" if trend_points <= 1 else "MIESZANY")
    if best is None:
        if trend_label == "POZYTYWNY":
            verdict = "Trend pozytywny, model ostrożny"
            tone = "info"
            detail = (
                f"Technicznie instrument wygląda pozytywnie, ale modele kierunkowe nie potwierdziły jeszcze "
                f"stabilnej przewagi poza próbką. Model 5d: AUC {five_day['auc']:.3f}, Brier {five_day['brier']:.3f}."
            )
        elif trend_label == "NEGATYWNY":
            verdict = "Słaby trend, brak przewagi modelu"
            tone = "warning"
            detail = (
                f"Trend i walidacja modelu są słabe. Model 5d: AUC {five_day['auc']:.3f}, "
                f"Brier {five_day['brier']:.3f}. To raczej kandydat do obserwacji niż do pochopnej decyzji."
            )
        else:
            verdict = "Brak czytelnej przewagi"
            tone = "warning"
            detail = (
                f"Rynek jest mieszany, a model 5d nie ma przewagi: AUC {five_day['auc']:.3f}, "
                f"Brier {five_day['brier']:.3f}. Warto patrzeć na hot movers, trend i dłuższe horyzonty."
            )
        best_label = "Brak potwierdzonego horyzontu"
    else:
        direction = signal_label(best["probability_up"], best["quality"]).lower()
        best_label = f"{horizon_text(best_horizon, result['symbol'].endswith('-USD'))} · {best['quality']}"
        if best["probability_up"] >= 0.54:
            verdict = f"Potwierdzony sygnał wzrostowy na {horizon_text(best_horizon, result['symbol'].endswith('-USD'))}"
            tone = "success"
        elif best["probability_up"] <= 0.46:
            verdict = f"Potwierdzone ryzyko spadku na {horizon_text(best_horizon, result['symbol'].endswith('-USD'))}"
            tone = "error"
        else:
            verdict = f"Potwierdzony model, ale sygnał neutralny na {horizon_text(best_horizon, result['symbol'].endswith('-USD'))}"
            tone = "info"
        detail = (
            f"Najlepszy potwierdzony horyzont: **{best_label}**. "
            f"Sygnał: **{direction}**, P(wzrost) {pct(best['probability_up'])}, "
            f"AUC {best['auc']:.3f}, Brier {best['brier']:.3f}. "
            f"Model 5d może być neutralny/słaby, ale to nie przekreśla dłuższego horyzontu."
        )
    return {"trend_label": trend_label, "best_label": best_label, "verdict": verdict, "tone": tone, "detail": detail}


def render_profile(profile: dict) -> None:
    if not profile:
        return
    fields = [
        ("Sektor", profile.get("sector") or profile.get("category") or "—"),
        ("Branża", profile.get("industry") or "—"),
        ("Kapitalizacja", compact_number(profile.get("marketCap") or profile.get("totalAssets"))),
        ("C/Z (historyczne)", f"{profile['trailingPE']:.2f}" if profile.get("trailingPE") else "—"),
        ("C/Z (prognozowane)", f"{profile['forwardPE']:.2f}" if profile.get("forwardPE") else "—"),
        ("Beta", f"{profile['beta']:.2f}" if profile.get("beta") is not None else "—"),
    ]
    st.subheader("Profil instrumentu")
    columns = st.columns(6)
    for column, (label, value) in zip(columns, fields):
        column.metric(label, value)


def render_analysis(result: dict, profile: dict) -> None:
    symbol = result["symbol"]
    st.divider()
    title_col, date_col = st.columns([3, 1])
    title_col.subheader(f"{profile_name(profile, symbol)} · {symbol}")
    date_col.caption(f"Dane do {result['last_date'].date()} · benchmark: {result['benchmark']}")

    technical = result["technical"]
    view = aggregate_model_view(result)
    summary_cols = st.columns(3)
    summary_cols[0].metric("Ostatnia cena", f"{result['last_price']:,.2f} {profile.get('currency', '')}".strip())
    summary_cols[1].metric("Trend techniczny", view["trend_label"], help="Opis bieżącego trendu, nie prognoza przyszłej ceny.")
    summary_cols[2].metric("Najlepszy horyzont modelu", view["best_label"])

    message = f"**{view['verdict']}.** {view['detail']}"
    if view["tone"] == "success":
        st.success(message)
    elif view["tone"] == "error":
        st.error(message)
    elif view["tone"] == "warning":
        st.warning(message)
    else:
        st.info(message)

    unit = "dzień" if symbol.endswith("-USD") else "sesja"
    horizon_names = {1: f"Następny {unit}", 5: "Najbliższy tydzień", 20: "Około miesiąca", 60: "Około kwartału"}
    columns = st.columns(len(result["forecasts"]))
    for column, (horizon, forecast) in zip(columns, result["forecasts"].items()):
        with column:
            st.metric(
                f"{horizon_names.get(horizon, str(horizon))} · {signal_label(forecast['probability_up'], forecast['quality'])}",
                f"P(wzrost): {pct(forecast['probability_up'])}",
                f"oczekiwany ruch {pct(forecast['expected_return'])}",
            )
            st.caption(
                f"Zakres 90%: {pct(forecast['lower_return'])} – {pct(forecast['upper_return'])}  ·  "
                f"AUC {forecast['auc']:.3f}  ·  Brier {forecast['brier']:.3f}  ·  {forecast['quality']}"
            )

    with st.expander("Diagnostyka prognozy — czy model naprawdę ma przewagę?"):
        diagnostic_rows = []
        for horizon, forecast in result["forecasts"].items():
            diagnostic_rows.append({
                "Horyzont": f"{horizon} dni" if symbol.endswith("-USD") else f"{horizon} sesji",
                "AUC": forecast["auc"], "Brier": forecast["brier"],
                "Trafność modelu": forecast["accuracy"],
                "Trafność prostego bazowego": forecast["baseline_accuracy"],
                "Przewaga trafności": forecast["accuracy"] - forecast["baseline_accuracy"],
                "Okres walidacji": f"{forecast['validation_start']} → {forecast['validation_end']}",
                "Liczba obserwacji": forecast["samples"],
                "Udział modelu liniowego": forecast["linear_weight"],
                "Folds walk-forward": forecast.get("validation_folds", 0),
                "Skład ensemble": model_mix(forecast.get("model_weights")),
            })
        diagnostics = pd.DataFrame(diagnostic_rows)
        st.dataframe(
            diagnostics.style.format({
                "AUC": "{:.3f}", "Brier": "{:.3f}", "Trafność modelu": "{:.1%}",
                "Trafność prostego bazowego": "{:.1%}", "Przewaga trafności": "{:+.1%}",
                "Udział modelu liniowego": "{:.0%}",
            }),
            use_container_width=True, hide_index=True,
        )
        st.caption("Model ma sens dopiero wtedy, gdy pokonuje prostą strategię przewidywania częstszej klasy. AUC około 0,50 oznacza brak zdolności rozróżniania kierunku.")

    history = result["history"].tail(500).copy()
    history["SMA 50"] = history["Close"].rolling(50).mean()
    history["SMA 200"] = history["Close"].rolling(200).mean()
    figure = go.Figure()
    figure.add_trace(go.Candlestick(
        x=history.index, open=history.Open, high=history.High, low=history.Low, close=history.Close,
        name=symbol,
    ))
    figure.add_trace(go.Scatter(x=history.index, y=history["SMA 50"], name="SMA 50", line=dict(width=1.5, color="#00bcd4")))
    figure.add_trace(go.Scatter(x=history.index, y=history["SMA 200"], name="SMA 200", line=dict(width=1.5, color="#ff9800")))
    figure.update_layout(
        height=520, xaxis_rangeslider_visible=False, margin=dict(l=20, r=20, t=25, b=20),
        legend=dict(orientation="h", font=dict(color="#dbeafe")),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(8,11,25,.55)",
        font=dict(color="#dbeafe"),
        xaxis=dict(gridcolor="rgba(166,125,255,.10)", linecolor="rgba(166,125,255,.18)"),
        yaxis=dict(gridcolor="rgba(166,125,255,.12)", linecolor="rgba(166,125,255,.18)"),
    )
    st.plotly_chart(figure, use_container_width=True)

    st.subheader("Momentum i trend")
    tech_cols = st.columns(6)
    values = [
        ("1 dzień", pct(technical["return_1d"])), ("5 sesji", pct(technical["return_5d"])),
        ("20 sesji", pct(technical["return_20d"])), ("RSI 14", f"{technical['rsi_14']:.1f}"),
        ("Nad SMA 50", "TAK" if technical["above_sma_50"] else "NIE"),
        ("Nad SMA 200", "TAK" if technical["above_sma_200"] else "NIE"),
    ]
    for column, (label, value) in zip(tech_cols, values):
        column.metric(label, value)

    risk = result["risk"]
    st.subheader("Ryzyko historyczne")
    risk_cols = st.columns(6)
    risk_values = [
        ("Zwrot roczny*", pct(risk["annual_return"])), ("Zmienność roczna", pct(risk["annual_volatility"])),
        ("Zmienność spadkowa", pct(risk["downside_volatility"])), ("Max drawdown", pct(risk["max_drawdown"])),
        ("Dzienny CVaR 95%", pct(risk["cvar_95_daily"])), ("Sharpe*", f"{risk['sharpe_zero_rf']:.2f}"),
    ]
    for column, (label, value) in zip(risk_cols, risk_values):
        column.metric(label, value)
    calendar = "365 dni" if risk["periods_per_year"] == 365 else "252 sesje"
    st.caption(f"*Estymacja historyczna z maksymalnie 3 lat; roczna skala: {calendar}; stopa wolna od ryzyka przyjęta jako 0.")

    render_profile(profile)
    with st.expander("Co najmocniej wpływa na model tygodniowy?"):
        st.bar_chart(result["forecasts"][5]["importance"])


def analysis_action(symbol: str, state_key: str, button_key: str) -> None:
    if st.button("Uruchom pełną analizę", type="primary", key=button_key, disabled=not symbol, use_container_width=True):
        try:
            with st.spinner("Pobieram dane, liczę wskaźniki i trenuję modele…"):
                result = cached_analysis(symbol, (1, 5, 20, 60), years)
                profile = cached_profile(symbol)
            st.session_state[state_key] = {"result": result, "profile": profile, "years": years}
        except Exception as exc:
            st.error(str(exc))
    saved = st.session_state.get(state_key)
    if saved and saved["result"]["symbol"] == symbol and saved.get("years") == years:
        render_analysis(saved["result"], saved["profile"])


def search_picker(prefix: str) -> str:
    with st.form(f"{prefix}_search_form"):
        query = st.text_input("Nazwa firmy lub instrumentu", placeholder="np. CD Projekt, Berkshire, uranium ETF", key=f"{prefix}_query")
        submitted = st.form_submit_button("Szukaj", type="primary")
    if submitted:
        try:
            with st.spinner("Szukam na światowych giełdach…"):
                st.session_state[f"{prefix}_results"] = cached_search(query)
        except Exception as exc:
            st.session_state[f"{prefix}_results"] = []
            st.error(f"Wyszukiwanie nie powiodło się: {exc}")
    results = st.session_state.get(f"{prefix}_results", [])
    if not results:
        if submitted:
            st.warning("Brak wyników. Spróbuj krótszej nazwy albo symbolu.")
        return ""
    options = {
        f"{item['name']}  ·  {item['symbol']}  ·  {item['exchange']}  ·  {item['type']}": item["symbol"]
        for item in results
    }
    selected = st.selectbox("Wyniki", list(options), key=f"{prefix}_selected")
    return options[selected]


def _render_ranking_table(frame: pd.DataFrame, title: str, empty_text: str) -> None:
    st.subheader(title)
    if frame.empty:
        st.info(empty_text)
        return
    formats = {
        "Cena": "{:.2f}", "P(wzrost)": "{:.1%}", "Oczekiwany ruch": "{:.1%}",
        "Zwrot 1d": "{:+.1%}", "Zwrot 5d": "{:+.1%}", "Zwrot 20d": "{:+.1%}", "RSI 14": "{:.1f}",
        "AUC walidacji": "{:.3f}", "Brier": "{:.3f}", "Pewność": "{:.1%}",
        "Zmienność roczna": "{:.1%}", "Max drawdown": "{:.1%}", "Score": "{:.2f}", "Radar score": "{:.1f}",
        "Risk/reward": "{:.2f}", "Edge score": "{:.2f}",
        "Deep score": "{:.0f}",
        "Setup score": "{:.0f}", "Momentum score": "{:.0f}", "Trend score": "{:.0f}",
        "Risk control": "{:.0f}", "Liquidity score": "{:.0f}", "Model edge": "{:.0f}",
    }
    columns = [
        "Symbol", "Klasa", "Tryb analizy", "Setup", "Setup grade", "Akcja radaru", "Radar momentum", "Teza radaru",
        "Setup score", "Ocena", "Ocena kierunku", "Ruch / impet",
        "Risk/reward", "Edge score", "Deep score", "Zwrot 1d", "Zwrot 5d", "Zwrot 20d", "RSI 14", "AUC walidacji", "Jakość modelu", "Score",
    ]
    present = [column for column in columns if column in frame.columns]
    st.dataframe(frame[present].style.format(formats, na_rep="—"), use_container_width=True, hide_index=True)


def _unique_symbols(frame: pd.DataFrame) -> int:
    return int(frame["Symbol"].nunique()) if "Symbol" in frame and not frame.empty else 0


def _class_from_symbol(symbol: str) -> str:
    symbol = str(symbol).upper()
    if symbol.endswith("-USD"):
        return "Krypto"
    if symbol.endswith(".WA"):
        return "GPW"
    if "." in symbol:
        return "ETF / Europa"
    if symbol.startswith("^"):
        return "Indeks"
    return "USA / ETF"


def _ensure_radar_columns(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    for column in [
        "Zwrot 1d", "Zwrot 5d", "Zwrot 20d", "Zmienność roczna", "RSI 14",
        "P(wzrost)", "Oczekiwany ruch", "Dolna granica 90%", "Górna granica 90%",
        "AUC walidacji", "Brier", "Risk/reward", "Edge score", "Radar score",
        "Deep score",
        "Setup score", "Momentum score", "Trend score",
        "Risk control", "Liquidity score", "Model edge",
    ]:
        if column in frame:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    def numeric(column: str) -> pd.Series:
        if column in frame:
            return pd.to_numeric(frame[column], errors="coerce").fillna(0)
        return pd.Series(0.0, index=frame.index)
    if "Radar score" not in frame:
        frame["Radar score"] = (
            numeric("Zwrot 1d") * 300
            + numeric("Zwrot 5d") * 180
            + numeric("Zwrot 20d") * 80
            - numeric("Zmienność roczna").clip(upper=3) * 0.9
        )
    if "Radar momentum" not in frame:
        def label(row: pd.Series) -> str:
            crypto = str(row.get("Symbol", "")).upper().endswith("-USD")
            hot_1d = 0.08 if crypto else 0.025
            hot_5d = 0.18 if crypto else 0.06
            hot_20d = 0.35 if crypto else 0.12
            r1 = float(row.get("Zwrot 1d") or 0)
            r5 = float(row.get("Zwrot 5d") or 0)
            r20 = float(row.get("Zwrot 20d") or 0)
            if r1 <= -hot_1d or r5 <= -hot_5d or r20 <= -hot_20d:
                return "PANIKA / RYZYKO"
            if r1 >= hot_1d or r5 >= hot_5d or r20 >= hot_20d:
                return "PEREŁKA MOMENTUM"
            if r5 > (0.08 if crypto else 0.03) and r20 > (0.06 if crypto else 0.025):
                return "MOMENTUM WATCH"
            return "—"
        frame["Radar momentum"] = frame.apply(label, axis=1)
    if "Risk/reward" not in frame:
        downside = numeric("Dolna granica 90%").clip(upper=0).abs()
        fallback_risk = numeric("Zmienność roczna").clip(lower=0.0) / 16
        downside = downside.where(downside > 0.005, fallback_risk).clip(lower=0.005)
        upside = pd.concat([
            numeric("Górna granica 90%"),
            numeric("Oczekiwany ruch"),
            pd.Series(0.0, index=frame.index),
        ], axis=1).max(axis=1).clip(lower=0)
        frame["Risk/reward"] = (upside / downside).replace([float("inf"), -float("inf")], 0).fillna(0)
    if "Edge score" not in frame:
        auc_quality = ((numeric("AUC walidacji") - 0.50) / 0.12).clip(lower=0, upper=1)
        brier_quality = ((0.27 - numeric("Brier")) / 0.06).clip(lower=0, upper=1)
        frame["Edge score"] = (
            numeric("Oczekiwany ruch") * 120
            + (numeric("P(wzrost)") - 0.5) * 90
            + frame["Risk/reward"].clip(upper=4) * 0.9
            + auc_quality * brier_quality * 2.4
            - numeric("Zmienność roczna").clip(upper=3) * 0.35
        )
    if "Akcja radaru" not in frame:
        def action(row: pd.Series) -> str:
            if str(row.get("Jakość modelu", "")).startswith("NISKA"):
                return "OBSERWUJ — BRAK EDGE ML"
            if float(row.get("Edge score") or 0) >= 4.5 and float(row.get("Oczekiwany ruch") or 0) > 0:
                return "PRIORYTET DO ANALIZY"
            if float(row.get("Edge score") or 0) >= 2.5 and float(row.get("Oczekiwany ruch") or 0) > 0:
                return "WATCHLIST"
            if float(row.get("Oczekiwany ruch") or 0) < 0 or float(row.get("P(wzrost)") or 0.5) < 0.45:
                return "RYZYKO / UNIKAJ"
            return "NEUTRALNIE"
        frame["Akcja radaru"] = frame.apply(action, axis=1)
    if "Tryb analizy" not in frame:
        frame["Tryb analizy"] = "ML"
    if "Momentum score" not in frame:
        frame["Momentum score"] = (
            45
            + numeric("Zwrot 1d") * 260
            + numeric("Zwrot 5d") * 150
            + numeric("Zwrot 20d") * 70
            - (numeric("RSI 14") - 82).clip(lower=0) * 0.7
        ).clip(lower=0, upper=100)
    if "Trend score" not in frame:
        frame["Trend score"] = (
            42
            + (numeric("Zwrot 20d") > 0).astype(float) * 18
            + (numeric("Zwrot 5d") > 0).astype(float) * 10
            + (numeric("RSI 14").between(45, 76)).astype(float) * 14
            + numeric("Zwrot 20d").clip(lower=-0.12, upper=0.12) * 120
        ).clip(lower=0, upper=100)
    if "Risk control" not in frame:
        frame["Risk control"] = (
            86
            - numeric("Zmienność roczna").clip(upper=3) * 55
            - numeric("Max drawdown").abs().clip(upper=0.90) * 28
            - (numeric("RSI 14") - 82).clip(lower=0) * 0.65
        ).clip(lower=0, upper=100)
    if "Model edge" not in frame:
        auc_quality = ((numeric("AUC walidacji") - 0.50) / 0.12).clip(lower=0, upper=1)
        brier_quality = ((0.27 - numeric("Brier")) / 0.06).clip(lower=0, upper=1)
        frame["Model edge"] = (
            auc_quality * brier_quality * 62
            + (numeric("P(wzrost)") - 0.5).clip(lower=0) * 120
            + numeric("Oczekiwany ruch").clip(lower=0) * 180
        ).clip(lower=0, upper=100)
    if "Liquidity score" not in frame:
        frame["Liquidity score"] = 50.0
    if "Setup score" not in frame:
        frame["Setup score"] = (
            numeric("Momentum score") * 0.23
            + numeric("Trend score") * 0.20
            + numeric("Risk control") * 0.18
            + numeric("Model edge") * 0.22
            + numeric("Liquidity score") * 0.07
            + frame["Risk/reward"].clip(upper=4) / 4 * 100 * 0.10
        ).clip(lower=0, upper=100)
    if "Deep score" not in frame:
        frame["Deep score"] = (
            numeric("Setup score") * 0.70
            + numeric("Edge score").clip(lower=0, upper=10) * 4
            + numeric("Radar score").clip(lower=-5, upper=15) * 1.5
        ).clip(lower=0, upper=100)
    if "Setup grade" not in frame:
        def setup_grade(row: pd.Series) -> str:
            setup_score = float(row.get("Setup score") or 0)
            risk_control = float(row.get("Risk control") or 0)
            model_edge = float(row.get("Model edge") or 0)
            momentum = float(row.get("Momentum score") or 0)
            if setup_score >= 72 and model_edge >= 45 and risk_control >= 45:
                return "A — czysty setup"
            if setup_score >= 62 and risk_control >= 35:
                return "B — watchlist"
            if momentum >= 72 and risk_control >= 30:
                return "M — momentum do sprawdzenia"
            if risk_control < 28:
                return "R — ryzyko dominuje"
            return "C — obserwuj"
        frame["Setup grade"] = frame.apply(setup_grade, axis=1)
    if "Teza radaru" not in frame:
        def thesis(row: pd.Series) -> str:
            reasons = []
            if float(row.get("Momentum score") or 0) >= 70:
                reasons.append("silne momentum")
            if float(row.get("Trend score") or 0) >= 70:
                reasons.append("trend wspiera ruch")
            if float(row.get("Model edge") or 0) >= 55:
                reasons.append("ML potwierdza edge")
            elif str(row.get("Jakość modelu", "")).startswith("NISKA"):
                reasons.append("ML bez przewagi")
            if float(row.get("Risk control") or 0) < 35:
                reasons.append("wysokie ryzyko/zmienność")
            return " · ".join(reasons[:4]) or "brak dominującego czynnika"
        frame["Teza radaru"] = frame.apply(thesis, axis=1)
    return frame


def _journal_dataframe(entries: list[dict]) -> pd.DataFrame:
    frame = pd.DataFrame(entries)
    if frame.empty:
        return frame
    frame["Wynik kierunkowy"] = frame["strategy_return"]
    frame["Zwrot instrumentu"] = frame["underlying_return"]
    frame["Trafiony"] = frame["hit"].map({True: "TAK", False: "NIE"}).fillna("—")
    frame["Status"] = frame["status"].map({"open": "otwarty", "closed": "zamknięty"}).fillna(frame["status"])
    frame["Kierunek"] = frame["direction"].map({"LONG": "wzrost", "SHORT": "spadek"}).fillna(frame["direction"])
    return frame.rename(columns={
        "signal_date": "Data sygnału", "symbol": "Symbol", "asset_class": "Klasa",
        "horizon": "Horyzont", "setup": "Setup", "label": "Ocena",
        "probability_up": "P(wzrost)", "expected_return": "Oczekiwany ruch",
        "quality": "Jakość", "signal_price": "Cena sygnału", "entry_date": "Data wejścia",
        "entry_price": "Cena wejścia", "execution": "Egzekucja", "target_date": "Data oceny",
        "target_price": "Cena oceny", "bars_elapsed": "Upłynęło", "bars_remaining": "Pozostało",
        "score": "Score",
    })


def render_signal_journal() -> None:
    st.header("Signal Journal")
    st.write("To dziennik skuteczności. MarketScope zapisuje directional signals z pełnych skanów i później sprawdza, czy po zadanym horyzoncie kierunek faktycznie zadziałał.")
    st.caption("To nadal paper-performance, nie historia realnych transakcji. Nie uwzględnia poślizgu, podatków ani wielkości pozycji.")

    actions = st.columns([1, 1, 1.35])
    if actions[0].button("Zapisz sygnały z ostatniego rankingu", key="journal_record", use_container_width=True):
        added = record_snapshot_signals(load_snapshot() or {})
        st.toast(f"Dodano nowych sygnałów: {added}", icon="📒")
        st.rerun()
    if actions[1].button("Aktualizuj wyniki", key="journal_refresh", use_container_width=True):
        with st.spinner("Sprawdzam, które sygnały dojrzały do oceny…"):
            _, errors = refresh_journal_results()
        if errors:
            st.warning(f"Nie udało się odświeżyć części symboli: {len(errors)}")
        st.toast("Journal odświeżony", icon="✅")
        st.rerun()
    actions[2].markdown(
        "<div class='ms-note'>Pełny skan zapisuje sygnały automatycznie. Ręczny zapis jest przydatny dla gotowego rankingu z poprzedniego uruchomienia.</div>",
        unsafe_allow_html=True,
    )

    entries = load_journal()
    summary = journal_summary(entries)
    metrics = st.columns(6)
    metrics[0].metric("Wszystkie sygnały", summary["total"])
    metrics[1].metric("Zamknięte", summary["closed"])
    metrics[2].metric("Otwarte", summary["open"])
    metrics[3].metric("Trafność", "—" if summary["hit_rate"] is None else pct(summary["hit_rate"]))
    metrics[4].metric("Śr. wynik", "—" if summary["average_return"] is None else pct(summary["average_return"]))
    metrics[5].metric("Mediana", "—" if summary["median_return"] is None else pct(summary["median_return"]))
    risk_metrics_cols = st.columns(4)
    risk_metrics_cols[0].metric("Profit factor", "—" if summary.get("profit_factor") is None else f"{summary['profit_factor']:.2f}")
    risk_metrics_cols[1].metric("Expectancy", "—" if summary.get("expectancy") is None else pct(summary["expectancy"]))
    risk_metrics_cols[2].metric("Max DD paper", "—" if summary.get("max_drawdown") is None else pct(summary["max_drawdown"]))
    risk_metrics_cols[3].metric("Payoff ratio", "—" if summary.get("payoff_ratio") is None else f"{summary['payoff_ratio']:.2f}")

    if not entries:
        st.info("Journal jest pusty. Uruchom pełny skan w zakładce **Sygnały**, a po zakończeniu directional signals zapiszą się automatycznie.")
        return

    frame = _journal_dataframe(entries)
    formats = {
        "P(wzrost)": "{:.1%}", "Oczekiwany ruch": "{:+.1%}", "Cena sygnału": "{:.2f}", "Cena wejścia": "{:.2f}",
        "Cena oceny": "{:.2f}", "Zwrot instrumentu": "{:+.1%}", "Wynik kierunkowy": "{:+.1%}",
        "Score": "{:.2f}",
    }
    closed = frame[frame["Status"] == "zamknięty"].sort_values("Data sygnału", ascending=False)
    open_entries = frame[frame["Status"] != "zamknięty"].sort_values(["Data sygnału", "Pozostało"], ascending=[False, True])

    tabs = st.tabs(["Performance Lab", "Otwarte sygnały", "Zamknięte wyniki", "Statystyki"])
    with tabs[0]:
        if closed.empty:
            st.info("Performance Lab ruszy po zamknięciu pierwszych sygnałów. Najpierw potrzebujemy historii paper-performance.")
        else:
            st.subheader("Paper Portfolio")
            controls = st.columns(3)
            starting_capital = controls[0].number_input(
                "Kapitał startowy",
                min_value=1_000,
                max_value=1_000_000,
                value=10_000,
                step=1_000,
                key="paper_starting_capital",
            )
            position_fraction = controls[1].slider(
                "Wielkość pozycji",
                min_value=1,
                max_value=50,
                value=10,
                step=1,
                format="%d%%",
                key="paper_position_fraction",
            ) / 100
            round_trip_cost_bps = controls[2].slider(
                "Koszt round-trip",
                min_value=0,
                max_value=150,
                value=20,
                step=5,
                help="Łączny koszt wejścia i wyjścia: prowizja, spread i uproszczony poślizg.",
                key="paper_round_trip_cost_bps",
            )
            portfolio_curve, portfolio = paper_portfolio(
                entries,
                starting_capital=float(starting_capital),
                position_fraction=float(position_fraction),
                round_trip_cost_bps=float(round_trip_cost_bps),
            )
            portfolio_cols = st.columns(6)
            portfolio_cols[0].metric("Transakcje", portfolio["trades"])
            portfolio_cols[1].metric("Kapitał końcowy", f"{portfolio['final_capital']:,.0f}")
            portfolio_cols[2].metric("Zwrot portfela", pct(portfolio["total_return"]))
            portfolio_cols[3].metric("Max DD", "—" if portfolio["max_drawdown"] is None else pct(portfolio["max_drawdown"]))
            portfolio_cols[4].metric("PF netto", "—" if portfolio["profit_factor"] is None else f"{portfolio['profit_factor']:.2f}")
            portfolio_cols[5].metric("Trafność netto", "—" if portfolio["hit_rate"] is None else pct(portfolio["hit_rate"]))

            performance = closed.copy().sort_values("Data sygnału")
            performance["Equity paper"] = (1 + performance["strategy_return"].fillna(0)).cumprod()
            performance["Drawdown"] = performance["Equity paper"] / performance["Equity paper"].cummax() - 1
            fig = go.Figure()
            if not portfolio_curve.empty:
                fig.add_trace(go.Scatter(
                    x=portfolio_curve["Data oceny"], y=portfolio_curve["Kapitał"],
                    mode="lines+markers", name="Paper portfolio",
                ))
                fig.add_trace(go.Scatter(
                    x=portfolio_curve["Data oceny"], y=portfolio_curve["Drawdown"],
                    mode="lines", name="Drawdown", yaxis="y2",
                ))
            fig.update_layout(
                template="plotly_dark",
                height=380,
                margin=dict(l=10, r=10, t=25, b=10),
                legend=dict(orientation="h"),
                yaxis=dict(title="Kapitał paper", tickformat=",.0f"),
                yaxis2=dict(title="Drawdown", overlaying="y", side="right", tickformat=".0%"),
            )
            st.plotly_chart(fig, use_container_width=True)
            with st.expander("Ostatnie transakcje paper portfolio"):
                portfolio_columns = [
                    "Data sygnału", "Data oceny", "Symbol", "Klasa", "Horyzont", "Kierunek",
                    "Zwrot brutto", "Zwrot netto", "Pozycja", "P&L", "Kapitał", "Drawdown",
                ]
                st.dataframe(
                    portfolio_curve[portfolio_columns].tail(30).style.format({
                        "Zwrot brutto": "{:+.1%}", "Zwrot netto": "{:+.1%}", "Pozycja": "{:.0%}",
                        "P&L": "{:+,.0f}", "Kapitał": "{:,.0f}", "Drawdown": "{:.1%}",
                    }),
                    use_container_width=True,
                    hide_index=True,
                )
            lab_cols = st.columns(2)
            by_class = closed.groupby("Klasa").agg(
                Liczba=("Symbol", "count"),
                Trafność=("hit", "mean"),
                Średni_wynik=("strategy_return", "mean"),
                Mediana=("strategy_return", "median"),
            ).sort_values("Średni_wynik", ascending=False).reset_index()
            by_direction = closed.groupby(["Kierunek", "Horyzont"]).agg(
                Liczba=("Symbol", "count"),
                Trafność=("hit", "mean"),
                Średni_wynik=("strategy_return", "mean"),
                Najgorszy=("strategy_return", "min"),
            ).sort_values("Średni_wynik", ascending=False).reset_index()
            lab_cols[0].subheader("Edge po klasach aktywów")
            lab_cols[0].dataframe(
                by_class.style.format({"Trafność": "{:.1%}", "Średni_wynik": "{:+.1%}", "Mediana": "{:+.1%}"}),
                use_container_width=True, hide_index=True,
            )
            lab_cols[1].subheader("Edge po kierunku i horyzoncie")
            lab_cols[1].dataframe(
                by_direction.style.format({"Trafność": "{:.1%}", "Średni_wynik": "{:+.1%}", "Najgorszy": "{:+.1%}"}),
                use_container_width=True, hide_index=True,
            )
            st.caption("To paper-performance sygnałów z journalu. Uwzględnia uproszczony koszt round-trip i sizing pozycji, ale nadal nie uwzględnia podatków, realnej płynności ani pełnego poślizgu z rynku.")
    with tabs[1]:
        columns = [
            "Data sygnału", "Symbol", "Klasa", "Horyzont", "Kierunek", "Setup", "Ocena",
            "P(wzrost)", "Cena sygnału", "Data wejścia", "Cena wejścia", "Egzekucja", "Upłynęło", "Pozostało", "Jakość", "Score",
        ]
        st.dataframe(open_entries[[c for c in columns if c in open_entries]].style.format(formats, na_rep="—"), use_container_width=True, hide_index=True)
    with tabs[2]:
        columns = [
            "Data sygnału", "Data oceny", "Symbol", "Horyzont", "Kierunek", "Trafiony",
            "Cena sygnału", "Data wejścia", "Cena wejścia", "Cena oceny", "Zwrot instrumentu", "Wynik kierunkowy", "Jakość", "Setup",
        ]
        if closed.empty:
            st.info("Jeszcze żaden sygnał nie dojrzał do oceny. Wróć po upływie horyzontu 1/5/20 dni lub sesji.")
        else:
            st.dataframe(closed[[c for c in columns if c in closed]].style.format(formats, na_rep="—"), use_container_width=True, hide_index=True)
    with tabs[3]:
        if closed.empty:
            st.info("Statystyki pojawią się po zamknięciu pierwszych sygnałów.")
        else:
            by_horizon = closed.groupby("Horyzont").agg(
                Liczba=("Symbol", "count"),
                Trafność=("hit", "mean"),
                Średni_wynik=("strategy_return", "mean"),
                Mediana=("strategy_return", "median"),
            ).reset_index()
            st.dataframe(
                by_horizon.style.format({"Trafność": "{:.1%}", "Średni_wynik": "{:+.1%}", "Mediana": "{:+.1%}"}),
                use_container_width=True, hide_index=True,
            )


def render_forward_cockpit() -> None:
    st.header("Live Forward Cockpit")
    st.write(
        "Read-only podgląd Candidate v1. Ta zakładka niczego nie zapisuje i nie zmienia — tylko czyta "
        "zamrożony proof ledger, ostatni snapshot i pokazuje, co realnie stało się z sygnałami."
    )
    cockpit = load_forward_cockpit()
    coverage = cockpit["coverage"]
    portfolio = cockpit["portfolio"]
    snapshot = cockpit["snapshot"]
    summary = cockpit["summary"]

    if cockpit["healthy"]:
        st.success("✅ Proof flow zdrowy — ledger, snapshot i pokrycie universe wyglądają poprawnie.")
    else:
        st.error("⚠️ Forward proof wymaga uwagi.")
        for problem in cockpit["problems"]:
            st.write(f"- {problem}")

    metric_cols = st.columns(6)
    requested = coverage["requested"] or 0
    completed = coverage["completed"] or 0
    metric_cols[0].metric("Ostatni snapshot", snapshot["updated_at_local"])
    metric_cols[1].metric("Universe coverage", f"{completed}/{requested}" if requested else "—")
    metric_cols[2].metric("Audit dni", cockpit["audit_days"], help="Unikalne dni, w których ledger zapisał SNAPSHOT_AUDIT.")
    metric_cols[3].metric("Signal dni", cockpit["signal_days"], help="Unikalne dni, w których pojawił się SIGNAL_OBSERVED.")
    metric_cols[4].metric("Otwarte pozycje", f"{portfolio['open']}/{portfolio['slots']}")
    metric_cols[5].metric("Wolne sloty", portfolio["free_slots"])

    event_cols = st.columns(5)
    event_cols[0].metric("Eventy", summary.get("events", 0))
    event_cols[1].metric("Obserwacje", summary.get("signals", 0), help="SIGNAL_OBSERVED, czyli sygnały zauważone przez model.")
    event_cols[2].metric("Pominięte", summary.get("skipped", 0))
    event_cols[3].metric("Zamknięte", summary.get("closed", 0))
    event_cols[4].metric("Snapshot errors", len(snapshot.get("errors") or {}))

    automation = load_automation_status()
    auto_stored = automation.get("stored") or {}
    auto_plan = automation.get("plan") or {}
    auto_launchd = automation.get("launchd") or {}
    st.subheader("Automatyzacja proof-flow")
    auto_cols = st.columns(5)
    auto_cols[0].metric("Auto status", auto_stored.get("automation_status") or "brak")
    auto_cols[1].metric("LaunchAgent", "loaded" if auto_launchd.get("loaded") else ("installed" if auto_launchd.get("plist_exists") else "nie zainstalowano"))
    auto_cols[2].metric("Target sesji", auto_stored.get("target_session_date") or auto_plan.get("target_session_date") or "—")
    auto_cols[3].metric("Następny run", auto_plan.get("next_planned_run_local", "—")[0:16] if auto_plan.get("next_planned_run_local") else "—")
    auto_cols[4].metric("Exit code", auto_stored.get("exit_code") if auto_stored.get("exit_code") is not None else "—")
    if automation.get("status_error"):
        st.warning(automation["status_error"])
    if auto_launchd.get("privacy_block_detected"):
        st.error(auto_launchd.get("privacy_hint") or "macOS privacy blocked the LaunchAgent.")
    elif auto_launchd.get("last_exit_code") not in {None, "0", 0}:
        st.warning(f"LaunchAgent jest załadowany, ale ostatni exit code to {auto_launchd.get('last_exit_code')}. Sprawdź log stderr w expanderze.")
    if auto_plan.get("missed_session_warning"):
        st.warning(auto_plan["missed_session_warning"])
    if auto_stored.get("automation_status") == "FAILED":
        st.error("Ostatni automatyczny run zakończył się błędem. Sprawdź stderr/log poniżej.")
    elif auto_stored.get("automation_status") == "OK":
        st.success("Automatyczny proof-flow ostatnio zakończył się poprawnie.")
    else:
        st.info("Automat jest gotowy jako osobny wrapper. Instalacja jest świadomą komendą w terminalu — UI niczego tu nie uruchamia.")

    with st.expander("Komendy automatyzacji i logi"):
        commands = automation.get("commands") or {}
        st.code(
            "\n".join([
                f"install:   {commands.get('install', '—')}",
                f"status:    {commands.get('status', '—')}",
                f"dry-run:   {commands.get('dry_run', '—')}",
                f"run-now:   {commands.get('run_now', '—')}",
                f"uninstall: {commands.get('uninstall', '—')}",
            ]),
            language="bash",
        )
        log_rows = []
        if auto_stored.get("stdout_log"):
            log_rows.append({"Typ": "stdout", "Plik": auto_stored.get("stdout_log")})
        if auto_stored.get("stderr_log"):
            log_rows.append({"Typ": "stderr", "Plik": auto_stored.get("stderr_log")})
        log_rows.append({"Typ": "launchd plist", "Plik": auto_launchd.get("plist")})
        st.dataframe(pd.DataFrame(log_rows), use_container_width=True, hide_index=True)
        if auto_stored.get("runner_summary_text"):
            st.text(str(auto_stored["runner_summary_text"]).strip())

    formats = {
        "Cena wejścia": "{:.2f}",
        "P(wzrost)": "{:.1%}",
        "Oczekiwany ruch": "{:+.1%}",
        "Zwrot netto": "{:+.1%}",
        "Cena wyjścia": "{:.2f}",
    }

    st.subheader("Aktywne pozycje i sloty")
    open_frame = pd.DataFrame(cockpit["open_positions"])
    if open_frame.empty:
        st.info("Brak aktywnych pozycji Candidate v1. To też jest poprawny stan — system nie wymusza transakcji.")
    else:
        columns = [
            "Symbol", "Status", "Slot", "Data sygnału", "Data wejścia", "Cena wejścia",
            "Planowane wyjście (≈)", "Sesje minęły (≈)", "Sesje do wyjścia (≈)",
            "P(wzrost)", "Oczekiwany ruch", "Jakość", "Decyzja",
        ]
        open_frame = open_frame.rename(columns={
            "Planowane wyjście": "Planowane wyjście (≈)",
            "Sesje minęły": "Sesje minęły (≈)",
            "Sesje do wyjścia": "Sesje do wyjścia (≈)",
        })
        present = [column for column in columns if column in open_frame]
        st.dataframe(open_frame[present].style.format(formats, na_rep="—"), use_container_width=True, hide_index=True)
        st.caption("Daty i liczba sesji w cockpicie są przybliżeniem na bazie dni roboczych. Ledger zamyka pozycję dopiero, gdy realna cena `Open` dla wyjścia jest dostępna.")

    st.subheader("Ostatni dzień forward testu")
    latest = pd.DataFrame(cockpit["latest_observations"])
    if latest.empty:
        st.info("Nie ma jeszcze obserwacji sygnałów w ledgerze.")
    else:
        st.caption(f"Najnowsza data sygnału: **{cockpit['latest_signal_date']}**")
        columns = [
            "Symbol", "Status", "Slot", "Data sygnału", "P(wzrost)", "Oczekiwany ruch",
            "Jakość", "Decyzja", "Powód pominięcia",
        ]
        present = [column for column in columns if column in latest]
        st.dataframe(latest[present].style.format(formats, na_rep="—"), use_container_width=True, hide_index=True)

    st.subheader("Historia eventów")
    events = pd.DataFrame(cockpit["recent_events"])
    if events.empty:
        st.info("Ledger jest pusty.")
    else:
        columns = [
            "Czas eventu", "Event", "Symbol", "Status", "Data sygnału", "Slot",
            "Wejście", "Cena wejścia", "Wyjście", "Cena wyjścia", "Powód/Decyzja",
        ]
        present = [column for column in columns if column in events]
        st.dataframe(events[present].style.format(formats, na_rep="—"), use_container_width=True, hide_index=True)

    with st.expander("Co oznaczają eventy?"):
        st.markdown("""
        - **SNAPSHOT_AUDIT** — MarketScope zapisał dowód, że pełny Candidate v1 universe został sprawdzony.
        - **SIGNAL_OBSERVED** — model zobaczył sygnał spełniający warunki Candidate v1.
        - **POSITION_ACCEPTED** — bramka portfela przydzieliła wolny slot.
        - **ENTRY_FILLED** — wejście zostało uzupełnione po następnym `Open`.
        - **POSITION_SKIPPED** — sygnał był realny, ale portfolio go nie przyjęło, np. symbol już był otwarty.
        - **POSITION_CLOSED** — po 20 sesjach pozycja została rozliczona.
        """)
        st.caption("Forward Cockpit jest tylko do odczytu. Główny run nadal odpalamy raz dziennie po 22:20.")


@st.fragment(run_every="30s")
def render_signal_dashboard() -> None:
    snapshot = load_snapshot()
    if not snapshot:
        if auto_start_signal_scan(None, True):
            st.info("Ranking nie został jeszcze policzony, więc MarketScope automatycznie wystartował pierwszy skan w tle. Postęp pojawi się za chwilę.")
        else:
            st.info("Ranking nie został jeszcze policzony. Uruchom aplikację plikiem **Uruchom MarketScope.command** albo użyj przycisku pełnego skanu poniżej.")
        return

    stale_snapshot = snapshot_is_stale(snapshot)
    status = snapshot.get("status")
    completed, total = snapshot.get("completed", 0), snapshot.get("total", 0)
    auto_started = auto_start_signal_scan(snapshot, stale_snapshot)
    if status == "running":
        phase = snapshot.get("scan_phase")
        if phase == "fast_radar":
            phase_text = (
                f"Etap **1/2: Fast Radar** — szybki techniczny skan całego rynku "
                f"(**{snapshot.get('fast_completed', completed)}/{snapshot.get('universe_total', total)}** instrumentów)."
            )
        elif phase == "deep_ml":
            phase_text = (
                f"Etap **2/2: Deep ML** — pełny model tylko dla shortlisty "
                f"(**{snapshot.get('ml_completed', 0)}/{snapshot.get('ml_total', 0)}** instrumentów)."
            )
        else:
            phase_text = "Monitor analizuje rynek w tle."
        progress_text = ""
        try:
            started_at = pd.Timestamp(snapshot.get("started_at"))
            if started_at.tzinfo is None:
                started_at = started_at.tz_localize("UTC")
            elapsed_seconds = max(1.0, (pd.Timestamp.now(tz="UTC") - started_at).total_seconds())
            if completed and total and completed < total:
                seconds_per_symbol = elapsed_seconds / completed
                remaining_seconds = max(0.0, (total - completed) * seconds_per_symbol)
                progress_text = (
                    f" Tempo: ok. **{seconds_per_symbol:.0f}s/instrument** · "
                    f"szacunkowo zostało **{remaining_seconds / 60:.0f} min**."
                )
            elif completed:
                progress_text = f" Czas pracy: **{elapsed_seconds / 60:.0f} min**."
        except Exception:
            progress_text = ""
        st.info(
            f"{phase_text} Łączny postęp: **{completed}/{total}** kroków. "
            "Ranking FAST pojawia się szybko, a wiersze ML zastępują go stopniowo dla najlepszych kandydatów. "
            f"Poniżej widzisz ranking częściowy — pełny obraz pojawi się po zakończeniu skanu.{progress_text}"
        )
        st.progress(completed / total if total else 0)
    elif status == "error":
        if auto_started:
            st.warning(f"Ostatni skan został przerwany: {snapshot.get('error', 'nieznany błąd')}. Startuję nowy skan w tle.")
        else:
            st.error(f"Ostatni skan został przerwany: {snapshot.get('error', 'nieznany błąd')}")
    else:
        updated = pd.Timestamp(snapshot["updated_at"])
        if updated.tzinfo is not None:
            updated = updated.tz_convert("Europe/Warsaw")
        horizons = snapshot.get("horizons") or [snapshot.get("horizon", 20)]
        horizon_text = ", ".join(f"{h}d" for h in horizons)
        if stale_snapshot:
            if auto_started:
                st.warning(
                    f"Ten ranking był stary albo niepełny (**{horizon_text}**, {snapshot.get('total', 0)} instrumentów), "
                    "więc MarketScope automatycznie wystartował świeży skan w tle."
                )
            else:
                st.warning(
                    f"Ten ranking jest ze starego formatu albo ma niepełny zakres (**{horizon_text}**, "
                    f"{snapshot.get('total', 0)} instrumentów). Uruchom ponownie aplikację albo kliknij **Przelicz cały ranking teraz**, "
                    "żeby dostać radar 1d/5d/20d z hot movers."
                )
        else:
            st.success(f"Skan zapisany · aktualizacja: **{updated.strftime('%Y-%m-%d %H:%M')}** · horyzonty: **{horizon_text}**")

    frame = pd.DataFrame(snapshot.get("records", []))
    if frame.empty:
        st.warning("Monitor nie ma jeszcze wystarczającej liczby ukończonych analiz.")
        return
    if "Horyzont" not in frame:
        frame["Horyzont"] = snapshot.get("horizon", 20)
    if "Klasa" not in frame:
        frame["Klasa"] = "Rynek"
    if "Setup" not in frame:
        frame["Setup"] = "—"
    frame = _ensure_radar_columns(frame)
    frame = with_signal_display_columns(frame)
    fast_rows = frame[frame["Tryb analizy"].astype(str).eq("FAST")]
    ml_rows = frame[frame["Tryb analizy"].astype(str).eq("ML")]

    bullish_labels = {"SILNY KANDYDAT WZROSTOWY", "KANDYDAT WZROSTOWY"}
    bearish_labels = {"SILNE RYZYKO SPADKU", "RYZYKO SPADKU"}
    bullish = frame[frame["Ocena"].isin(bullish_labels)]
    bearish = frame[frame["Ocena"].isin(bearish_labels)]
    discoveries = frame[frame["Radar momentum"].isin({"PEREŁKA MOMENTUM", "BREAKOUT WATCH", "MOMENTUM WATCH"})]
    data_contract = signal_scan_contract(frame, snapshot)
    crypto_no_data = [symbol for symbol in data_contract["no_data"] if str(symbol).upper().endswith("-USD")]
    crypto_failed = [symbol for symbol in data_contract["failed"] if str(symbol).upper().endswith("-USD")]
    summary = st.columns(6)
    summary[0].metric(
        "FAST skan",
        f"{data_contract['fast_completed']}/{data_contract['universe_total']}",
        help="Ile instrumentów przeszło lekki skan techniczny całego universe.",
    )
    summary[1].metric(
        "W rankingu",
        data_contract["ranked_symbols"],
        help="Instrumenty, dla których zapisano wiersze rankingu; horyzonty 1d/5d/20d tworzą osobne wiersze.",
    )
    summary[1].caption(f"{data_contract['ranked_rows']} wierszy")
    summary[2].metric("FAST rows", data_contract["fast_rows"], help="Wiersze techniczne bez pełnego potwierdzenia ML.")
    summary[3].metric("ML rows", data_contract["ml_rows"], help="Wiersze po pełnej walidacji modelu.")
    risk_label = "brak" if _unique_symbols(bearish) == 0 else str(_unique_symbols(bearish))
    summary[4].metric("Alerty ryzyka", risk_label, help="To brak wykrytych alertów w radarze, nie gwarancja braku ryzyka.")
    summary[4].caption("wykrytych alertów ryzyka")
    summary[5].metric(
        "Brak danych / błędy",
        f"{len(data_contract['no_data'])}/{len(data_contract['failed'])}",
        help="Pierwsza liczba to brak danych krypto z providera, druga to prawdziwe błędy runtime/pobierania.",
    )
    summary[5].caption("krypto / runtime")
    if snapshot.get("scan_mode") == "two_stage":
        st.caption(
            f"Tryb dwustopniowy: FAST skanuje cały rynek (**{data_contract['fast_completed']}/{data_contract['universe_total']}**), "
            f"a Deep ML wzbogaca shortlistę. Próby ML: **{data_contract['ml_attempted']}/{data_contract['ml_total']}**, "
            f"zapisane ML rows: **{data_contract['ml_rows']}**. FAST rows nie są zapisywane do Journala jako directional signals."
        )

    if crypto_no_data:
        st.warning(
            f"Ten zapisany skan ma **{len(crypto_no_data)} krypto bez danych** w momencie pobierania. "
            "To missing data z providera, nie awaria MarketScope. Ranking krypto jest więc niepełny."
        )
    if crypto_failed:
        st.error(f"Prawdziwe błędy pobierania krypto: **{len(crypto_failed)}**. Szczegóły są w surowym snapshotcie.")

    radar_brief = build_signal_radar_brief(
        frame,
        snapshot,
        bullish_labels=bullish_labels,
        bearish_labels=bearish_labels,
    )
    radar_cards = "".join(
        '<div class="signal-brief-card">'
        f"<small>{clean_text(label)}</small>"
        f"<strong>{clean_text(value)}</strong>"
        f"<span>{clean_text(detail)}</span>"
        "</div>"
        for label, value, detail in radar_brief["cards"]
    )
    st.markdown(
        '<div class="signal-brief">'
        '<div class="signal-brief-main">'
        "<small>Co teraz pokazuje radar?</small>"
        f"<h3>{clean_text(radar_brief['headline'])}</h3>"
        f"<p>{clean_text(radar_brief['body'])}</p>"
        f"<div class=\"signal-brief-note\">📡 {clean_text(radar_brief['note'])}</div>"
        "</div>"
        f'<div class="signal-brief-grid">{radar_cards}</div>'
        "</div>",
        unsafe_allow_html=True,
    )

    formats = {
        "Cena": "{:.2f}", "P(wzrost)": "{:.1%}", "Oczekiwany ruch": "{:.1%}",
        "Zwrot 1d": "{:+.1%}", "Zwrot 5d": "{:+.1%}", "Zwrot 20d": "{:+.1%}", "RSI 14": "{:.1f}",
        "AUC walidacji": "{:.3f}", "Brier": "{:.3f}", "Pewność": "{:.1%}", "Zmienność roczna": "{:.1%}",
        "Max drawdown": "{:.1%}", "Score": "{:.2f}", "Radar score": "{:.1f}",
        "Risk/reward": "{:.2f}", "Edge score": "{:.2f}",
        "Deep score": "{:.0f}",
        "Setup score": "{:.0f}", "Momentum score": "{:.0f}", "Trend score": "{:.0f}",
        "Risk control": "{:.0f}", "Liquidity score": "{:.0f}", "Model edge": "{:.0f}",
    }

    horizon_tabs = st.tabs([
        "Dzisiejszy radar", "Setup intelligence", "Perełki momentum", "Risk/reward",
        "Szybki ruch 1d", "Swing 5d", "Trend 20d", "Wszystko",
    ])
    with horizon_tabs[0]:
        st.subheader("Dzisiejszy radar")
        st.caption("Szybki briefing: gdzie patrzeć najpierw. To shortlist badawcza, nie automatyczna rekomendacja transakcji.")
        base = frame.copy()
        priority = (
            base[base["Akcja radaru"].isin(["PRIORYTET DO ANALIZY", "WATCHLIST", "FAST SHORTLIST", "MOMENTUM DO SPRAWDZENIA"])]
            .sort_values(["Deep score", "Setup score", "Edge score", "Risk/reward"], ascending=False)
            .drop_duplicates("Symbol")
            .head(5)
        )
        render_setup_cockpit(base, bullish_labels)
        hot_pool = base.copy()
        return_columns = [column for column in ["Zwrot 1d", "Zwrot 5d", "Zwrot 20d"] if column in hot_pool.columns]
        hot_pool["_hot_move"] = hot_pool[return_columns].abs().max(axis=1).fillna(0.0) if return_columns else 0.0
        priority_symbols = set(priority["Symbol"].dropna().astype(str))
        hot_candidates = hot_pool[~hot_pool["Symbol"].astype(str).isin(priority_symbols)]
        if hot_candidates.empty:
            hot_candidates = hot_pool
        hot_now = (
            hot_candidates.sort_values(["_hot_move", "Radar score", "Deep score"], ascending=False)
            .drop_duplicates("Symbol")
            .head(5)
            .drop(columns=["_hot_move"], errors="ignore")
        )
        risk_now = (
            base[
                base["Akcja radaru"].eq("RYZYKO / UNIKAJ")
                | base["Ocena"].isin(bearish_labels)
                | base["Radar momentum"].eq("PANIKA / RYZYKO")
            ]
            .sort_values(["Edge score", "Score"], ascending=True)
            .drop_duplicates("Symbol")
            .head(5)
        )
        compact_columns = [
            "Symbol", "Klasa", "Tryb analizy", "Horyzont", "Setup grade", "Akcja radaru", "Teza radaru",
            "Ocena kierunku", "Ruch / impet", "Deep score", "Setup score", "Risk/reward", "Edge score",
        ]
        st.markdown("#### 1. Najważniejsze setupy do dalszej analizy")
        st.dataframe(priority[[c for c in compact_columns if c in priority]].style.format(formats, na_rep="—"), use_container_width=True, hide_index=True)
        st.markdown("#### 2. Autentyczne hot movers — wybrane innym kryterium")
        st.dataframe(hot_now[[c for c in compact_columns if c in hot_now]].style.format(formats, na_rep="—"), use_container_width=True, hide_index=True)
        st.markdown("#### 3. Alerty ryzyka / unikaj")
        if risk_now.empty:
            st.info("Brak wykrytych alertów ryzyka w zapisanym rankingu. To nie oznacza braku ryzyka rynkowego — tylko brak alertu według obecnych filtrów.")
        else:
            st.dataframe(risk_now[[c for c in compact_columns if c in risk_now]].style.format(formats, na_rep="—"), use_container_width=True, hide_index=True)
    with horizon_tabs[1]:
        st.subheader("Setup intelligence")
        st.caption("Rozbicie score na elementy, które trader sprawdza ręcznie: impet, trend, kontrolę ryzyka, płynność i potwierdzenie modelu.")
        setup_frame = (
            frame.sort_values(["Deep score", "Setup score", "Edge score", "Radar score"], ascending=False)
            .drop_duplicates("Symbol")
            .head(25)
        )
        setup_columns = [
            "Symbol", "Klasa", "Tryb analizy", "Horyzont", "Setup grade", "Teza radaru", "Deep score", "Setup score",
            "Momentum score", "Trend score", "Risk control", "Model edge", "Liquidity score",
            "P(wzrost)", "Risk/reward", "AUC walidacji", "Jakość modelu",
        ]
        st.dataframe(setup_frame[[c for c in setup_columns if c in setup_frame]].style.format(formats, na_rep="—"), use_container_width=True, hide_index=True)
        st.info(
            "**Setup score** to nie sygnał kupna. To priorytet dalszej analizy: im wyżej, tym bardziej spójny jest układ momentum + trend + ryzyko + ML."
        )
    with horizon_tabs[2]:
        base = frame[frame["Horyzont"] == frame["Horyzont"].min()].copy()
        if base.empty:
            base = frame.copy()
        hot = base.sort_values("Radar score", ascending=False).head(18)
        st.subheader("Perełki momentum i najmocniejsze aktualne ruchy")
        st.caption("Ten widok nie wymaga potwierdzenia ML. Łapie gwałtowne ruchy i breakouty do szybkiego sprawdzenia — szczególnie przy krypto.")
        hot_columns = [
            "Symbol", "Klasa", "Radar momentum", "Setup", "Zwrot 1d", "Zwrot 5d", "Zwrot 20d",
            "RSI 14", "Radar score", "Ocena", "Ocena kierunku", "Ruch / impet", "AUC walidacji", "Jakość modelu",
        ]
        present = [column for column in hot_columns if column in hot.columns]
        st.dataframe(hot[present].style.format(formats, na_rep="—"), use_container_width=True, hide_index=True)
    with horizon_tabs[3]:
        rr = frame.sort_values(["Edge score", "Deep score", "Setup score", "Risk/reward"], ascending=False).drop_duplicates("Symbol").head(20)
        st.subheader("Najlepszy stosunek potencjału do ryzyka")
        st.caption("Ranking łączy oczekiwany ruch, przedział niepewności, prawdopodobieństwo, AUC/Brier i zmienność. Wysoki wynik oznacza priorytet analizy, nie pewność zysku.")
        rr_columns = [
            "Symbol", "Klasa", "Tryb analizy", "Horyzont", "Setup grade", "Akcja radaru", "Teza radaru", "Setup", "Ocena",
            "Ocena kierunku", "Ruch / impet", "Risk/reward", "Edge score",
            "Deep score", "Setup score", "Risk control", "AUC walidacji", "Brier", "Jakość modelu", "Zmienność roczna",
        ]
        st.dataframe(rr[[c for c in rr_columns if c in rr]].style.format(formats, na_rep="—"), use_container_width=True, hide_index=True)
    for tab, horizon, title in [
        (horizon_tabs[4], 1, "Najciekawsze setupy krótkoterminowe"),
        (horizon_tabs[5], 5, "Najciekawsze setupy swingowe"),
        (horizon_tabs[6], 20, "Najciekawsze setupy trendowe"),
    ]:
        with tab:
            scoped = frame[frame["Horyzont"] == horizon].sort_values("Score", ascending=False)
            candidates = scoped[scoped["Ocena"].isin(bullish_labels)]
            if candidates.empty:
                _render_ranking_table(scoped.head(12), title, "Brak potwierdzonych kandydatów; pokazuję najwyżej oceniane obserwacje.")
            else:
                _render_ranking_table(candidates.head(12), title, "Brak potwierdzonych kandydatów.")
            st.caption("To shortlist do dalszej analizy. Nie jest rekomendacją kupna ani gwarancją ruchu.")

    with horizon_tabs[7]:
        filters = st.columns(4)
        selected_horizon = filters[0].selectbox("Horyzont", ["Wszystkie", 1, 5, 20, 60], key="radar_horizon_filter")
        selected_class = filters[1].selectbox("Klasa", ["Wszystkie"] + sorted(frame["Klasa"].dropna().unique().tolist()), key="radar_class_filter")
        only_discoveries = filters[2].checkbox("Tylko perełki momentum", value=False, key="radar_discoveries_only")
        only_candidates = filters[3].checkbox("Tylko kandydaci ML", value=False, key="radar_candidates_only")
        filtered = frame.copy()
        if selected_horizon != "Wszystkie":
            filtered = filtered[filtered["Horyzont"] == selected_horizon]
        if selected_class != "Wszystkie":
            filtered = filtered[filtered["Klasa"] == selected_class]
        if only_discoveries:
            filtered = filtered[filtered["Radar momentum"].isin({"PEREŁKA MOMENTUM", "BREAKOUT WATCH", "MOMENTUM WATCH"})]
        if only_candidates:
            filtered = filtered[filtered["Ocena"].isin(bullish_labels)]
        filtered = filtered.sort_values(["Deep score", "Radar score", "Score"], ascending=False)
        columns = [
            "Symbol", "Klasa", "Tryb analizy", "Horyzont", "Setup grade", "Akcja radaru", "Radar momentum", "Teza radaru",
            "Deep score", "Setup score", "Setup", "Cena", "Ocena", "P(wzrost)", "Oczekiwany ruch",
            "Zwrot 1d", "Zwrot 5d", "Zwrot 20d", "RSI 14", "AUC walidacji", "Brier", "Jakość modelu",
            "Momentum score", "Trend score", "Risk control", "Model edge", "Liquidity score",
            "Pewność", "Zmienność roczna", "Max drawdown", "Risk/reward", "Edge score", "Radar score", "Score",
        ]
        present = [column for column in columns if column in filtered.columns]
        st.dataframe(filtered[present].style.format(formats, na_rep="—"), use_container_width=True, hide_index=True)
        st.download_button("Pobierz ranking CSV", filtered.to_csv(index=False).encode(), "marketscope_signals.csv", "text/csv")

    if not bearish.empty:
        with st.expander(f"Alerty ryzyka ({_unique_symbols(bearish)} symboli)"):
            columns = [
                "Symbol", "Klasa", "Tryb analizy", "Horyzont", "Setup grade", "Akcja radaru",
                "Teza radaru", "Ocena kierunku", "Ruch / impet", "AUC walidacji", "Jakość modelu",
            ]
            present = [column for column in columns if column in bearish.columns]
            st.dataframe(bearish[present].style.format(formats, na_rep="—"), use_container_width=True, hide_index=True)

    no_data_errors = data_contract["no_data"]
    provider_no_data_errors = data_contract["provider_no_data"]
    failed_errors = data_contract["failed"]
    if no_data_errors or failed_errors:
        with st.expander(f"Brak danych i błędy ({len(no_data_errors)} brak danych / {len(failed_errors)} błędów)"):
            error_rows = [
                {"Symbol": symbol, "Klasa": _class_from_symbol(symbol), "Typ": "Brak danych", "Opis": message}
                for symbol, message in sorted(no_data_errors.items())
            ]
            error_rows.extend(
                {"Symbol": symbol, "Klasa": _class_from_symbol(symbol), "Typ": "Błąd", "Opis": message}
                for symbol, message in sorted(failed_errors.items())
            )
            st.dataframe(pd.DataFrame(error_rows), use_container_width=True, hide_index=True)
            if len(provider_no_data_errors) > len(no_data_errors):
                st.caption(
                    f"Surowy log providera zawiera {len(provider_no_data_errors)} wpisy no-data. "
                    f"W głównym panelu pokazuję {len(no_data_errors)} braków krypto, bo to one wpływają na kompletność segmentu krypto."
                )


_hero_snapshot = load_snapshot() or {}
_hero_journal = journal_summary(load_journal())
_hero_universe = len(default_universe())
_hero_forward, _hero_forward_error = safe_load_forward_cockpit()
_hero_automation, _hero_automation_error = safe_load_automation_status()
_hero_proof = proof_state(_hero_forward, _hero_automation, _hero_forward_error, _hero_automation_error)
_hero_portfolio = (_hero_forward or {}).get("portfolio") or {}
_hero_coverage = (_hero_forward or {}).get("coverage") or {}
_hero_auto_launchd = (_hero_automation or {}).get("launchd") or {}
_hero_status = "Skan trwa" if _hero_snapshot.get("status") == "running" else ("Gotowy" if _hero_snapshot.get("status") == "complete" else "Offline")
_hero_completed = _hero_snapshot.get("fast_completed", _hero_snapshot.get("completed", 0))
_hero_total = _hero_snapshot.get("universe_total", _hero_snapshot.get("total", _hero_universe))
_hero_forward_coverage = (
    f"{_hero_coverage.get('completed', 0)}/{_hero_coverage.get('requested', 0)}"
    if _hero_coverage.get("requested")
    else "—"
)
st.markdown(f"""
<div class="ms-topbar">
    <div class="ms-brand">
        <div class="ms-logo">MS</div>
        <div>
            <h1>MarketScope PRO</h1>
            <p>Quant radar · akcje · ETF-y · krypto · risk & performance</p>
        </div>
    </div>
    <div class="ms-status-strip">
        <span class="ms-chip"><i class="ms-led"></i>{_hero_status}</span>
        <span class="ms-chip">FAST {_hero_completed}/{_hero_total}</span>
        <span class="ms-chip">Proof {clean_text(_hero_proof["label"])}</span>
        <span class="ms-chip">Forward {_hero_forward_coverage}</span>
        <span class="ms-chip">Portfolio {_hero_portfolio.get("open", 0)}/{_hero_portfolio.get("slots", 5)}</span>
        <span class="ms-chip">Auto {"ON" if _hero_auto_launchd.get("loaded") else "OFF"}</span>
        <span class="ms-chip">Engine 3×ML</span>
    </div>
</div>
""", unsafe_allow_html=True)

home, stocks, etfs, crypto, radar, forward_tab, journal, backtest, settings, method = st.tabs([
    "Start", "Spółki", "ETF-y", "Krypto", "Sygnały", "Forward", "Journal", "Backtest", "Model", "Metodologia",
])

with home:
    render_start_dashboard(
        snapshot=_hero_snapshot,
        journal=_hero_journal,
        universe_size=_hero_universe,
        cockpit=_hero_forward,
        automation=_hero_automation,
        cockpit_error=_hero_forward_error,
        automation_error=_hero_automation_error,
    )

with stocks:
    st.header("Analiza spółek")
    stock_mode = st.radio("Sposób wyboru", ["Katalog", "Wyszukiwarka globalna", "Wpisz symbol"], horizontal=True, key="stock_mode")
    stock_symbol = ""
    if stock_mode == "Katalog":
        stock_category = st.selectbox("Rynek / sektor", list(CATEGORIES), key="stock_category")
        stock_options = category_options(stock_category)
        stock_choice = st.selectbox("Spółka", list(stock_options), key="stock_choice")
        stock_symbol = stock_options[stock_choice]
    elif stock_mode == "Wyszukiwarka globalna":
        stock_symbol = search_picker("stocks")
    else:
        stock_symbol = st.text_input("Symbol", "AAPL", help="GPW: np. PKO.WA, CDR.WA. USA: np. AAPL.", key="stock_manual").strip().upper()
    if stock_symbol:
        st.caption(f"Wybrany instrument: `{stock_symbol}`")
    analysis_action(stock_symbol, "stock_analysis", "stock_analyze")

with etfs:
    st.header("Analiza ETF-ów")
    st.write("ETF pozwala analizować cały rynek, sektor, obligacje lub surowiec jednym instrumentem.")
    etf_category = st.selectbox("Kategoria ETF", list(ETF_CATEGORIES), key="etf_category")
    available_etfs = etf_options(etf_category)
    etf_choice = st.selectbox("ETF", list(available_etfs), key="etf_choice")
    etf_symbol = available_etfs[etf_choice]
    st.caption(f"Wybrany ETF: `{etf_symbol}`")
    analysis_action(etf_symbol, "etf_analysis", "etf_analyze")

with crypto:
    st.header("Analiza kryptowalut")
    st.warning("Krypto może poruszać się gwałtownie 24/7. Przedziały niepewności i ryzyko są tu szczególnie ważne.")
    crypto_mode = st.radio("Sposób wyboru", ["Szukaj w całym krypto", "Segmenty", "Wpisz symbol"], horizontal=True, key="crypto_mode")
    crypto_symbol = ""
    if crypto_mode == "Szukaj w całym krypto":
        crypto_available = crypto_options()
        crypto_choice = st.selectbox(
            "Kryptowaluta",
            list(crypto_available),
            key="crypto_global_choice",
            help="To pole przeszukuje cały katalog krypto, więc znajdziesz tu np. DeXe, Uniswap, Render albo Bitcoin.",
        )
        crypto_symbol = crypto_available[crypto_choice]
    elif crypto_mode == "Segmenty":
        crypto_category = st.selectbox("Segment krypto", list(CRYPTO_CATEGORIES), key="crypto_category")
        crypto_available = crypto_category_options(crypto_category)
        crypto_choice = st.selectbox("Kryptowaluta", list(crypto_available), key="crypto_choice")
        crypto_symbol = crypto_available[crypto_choice]
    else:
        crypto_symbol = st.text_input("Symbol", "DEXE-USD", help="Yahoo/yfinance format, np. BTC-USD, ETH-USD, DEXE-USD, UNI-USD.", key="crypto_manual").strip().upper()
    if crypto_symbol:
        st.caption(f"Wybrana para: `{crypto_symbol}`")
    analysis_action(crypto_symbol, "crypto_analysis", "crypto_analyze")

with radar:
    st.header("Automatyczny ranking rynku")
    st.write(f"Monitor śledzi **{len(default_universe())}** instrumentów z GPW, USA, ETF-ów i krypto oraz liczy kilka horyzontów: szybki ruch, swing i trend.")
    st.caption("To lista badawcza, nie automatyczna rekomendacja zakupu ani sprzedaży.")
    st.warning(
        "Ranking działa dwustopniowo: najpierw szybki FAST Radar skanuje cały rynek, potem Deep ML liczy pełne modele tylko dla shortlisty. "
        "W trakcie dashboard pokazuje wynik częściowy i stopniowo zastępuje wiersze FAST wierszami ML.",
        icon="⏳",
    )
    render_signal_dashboard()
    radar_snapshot = load_snapshot()
    scan_running = bool(radar_snapshot and radar_snapshot.get("status") == "running")
    if scan_running:
        st.caption("Pełny skan już trwa. Przycisk przeliczenia jest zablokowany, żeby nie startować drugiego procesu na tych samych danych.")
    if st.button(
        "Przelicz cały ranking teraz",
        key="signals_refresh",
        help="Startuje dwustopniowy skan: szybki FAST Radar całego rynku, potem Deep ML dla shortlisty. Dashboard pokazuje postęp.",
        disabled=scan_running,
    ):
        start_signal_scan_background()
        st.toast("Startuję dwustopniowy skan: FAST Radar, potem Deep ML shortlisty. Postęp pojawi się za chwilę.", icon="📡")
        st.rerun()
    with st.expander("Szybki skan własnych symboli"):
        st.write("Tu możesz sprawdzić instrumenty spoza głównego radaru, np. świeże krypto albo małe spółki.")
        custom_symbols = st.text_area(
            "Symbole oddzielone przecinkami",
            "DEXE-USD, BTC-USD, ETH-USD, AAPL, NVO, PKO.WA",
            key="custom_scan_symbols",
        )
        custom_horizons = st.multiselect("Horyzonty", [1, 5, 20, 60], default=[1, 5, 20], key="custom_scan_horizons")
        if st.button("Skanuj tę listę", key="custom_scan_button"):
            symbols = [part.strip().upper() for part in custom_symbols.replace("\n", ",").split(",") if part.strip()]
            try:
                with st.spinner("Liczenie prywatnego skanu…"):
                    custom_frame, custom_errors = scan_market_multi(symbols, tuple(custom_horizons or [5]), years)
                st.session_state["custom_scan"] = (custom_frame, custom_errors)
            except Exception as exc:
                st.error(str(exc))
        if "custom_scan" in st.session_state:
            custom_frame, custom_errors = st.session_state["custom_scan"]
            _render_ranking_table(custom_frame.sort_values("Score", ascending=False), "Wynik szybkiego skanu", "Brak danych do pokazania.")
            if custom_errors:
                st.caption(f"Pominięte / bez danych: {', '.join(custom_errors)}")

with forward_tab:
    render_forward_cockpit()

with journal:
    render_signal_journal()

with backtest:
    st.header("Backtest walk-forward")
    st.write("Model zoo jest wielokrotnie trenowany wyłącznie na przeszłości, a następnie testowany na kolejnych, niewidzianych danych.")
    st.caption("Sygnał powstaje po zamknięciu świecy, wejście odbywa się na następnym open, a wynik uwzględnia koszt i uproszczony poślizg.")
    bt_symbol = st.text_input("Symbol", "SPY", help="Np. AAPL, CDR.WA, SPY, BTC-USD", key="bt_symbol").strip().upper()
    b1, b2, b3, b4 = st.columns(4)
    bt_horizon = b1.selectbox("Horyzont", [1, 5, 20, 60], index=1, key="bt_horizon")
    threshold = b2.slider("Minimalna pewność wejścia", 0.51, 0.70, DEFAULT_SIGNAL_THRESHOLD, 0.01, key="bt_threshold")
    cost_bps = b3.number_input("Koszt transakcji (punkty bazowe)", 0.0, 100.0, 10.0, 1.0, key="bt_cost")
    slippage_bps = b4.number_input("Poślizg (punkty bazowe)", 0.0, 100.0, 5.0, 1.0, key="bt_slippage")
    if st.button("Uruchom test historyczny", type="primary", key="bt_run", use_container_width=True):
        try:
            with st.spinner("Symuluję ensemble out-of-sample z wejściem na następnym open…"):
                data = download_history(bt_symbol, years)
                curve, metrics = walk_forward_backtest(data, bt_horizon, threshold, cost_bps, slippage_bps)
            st.session_state["bt_result"] = (bt_symbol, curve, metrics)
        except Exception as exc:
            st.error(str(exc))
    if "bt_result" in st.session_state and st.session_state["bt_result"][0] == bt_symbol:
        _, curve, metrics = st.session_state["bt_result"]
        cols = st.columns(8)
        values = [
            ("Łączny zwrot", pct(metrics["total_return"])), ("CAGR", pct(metrics["annual_return"])),
            ("Zmienność", pct(metrics["annual_volatility"])), ("Sharpe", f"{metrics['sharpe']:.2f}"),
            ("Max drawdown", pct(metrics["max_drawdown"])), ("Trafność", pct(metrics["hit_rate"])),
            ("AUC", f"{metrics['auc']:.3f}"), ("Brier", f"{metrics['brier']:.3f}"),
        ]
        for col, (label, value) in zip(cols, values):
            col.metric(label, value)
        st.line_chart(curve[["Equity", "BuyHold"]])
        st.caption(
            f"Aktywne sygnały: {metrics['trades']}. Egzekucja: next open. "
            f"Koszt: {metrics['cost_bps']:.0f} bps, poślizg: {metrics['slippage_bps']:.0f} bps. "
            "Wyniki historyczne nie gwarantują przyszłych."
        )

with settings:
    st.header("Model i ustawienia")
    st.write("Ta sekcja tłumaczy ustawienia normalnym językiem. Domyślna konfiguracja jest zalecana — więcej danych lub bardziej agresywny sygnał nie oznacza automatycznie lepszej prognozy.")

    st.subheader("Ile historii wykorzystać?")
    selected_years = st.slider(
        "Lata danych do treningu", 3, 15, key="training_years",
        help="Model uczy się na tej historii, a jej najnowsza część zostaje odłożona do uczciwej walidacji.",
    )
    if selected_years == 8:
        st.success("**8 lat — ustawienie zalecane.** Zwykle obejmuje kilka faz rynku bez nadmiernego sięgania do bardzo starych zależności.")
    elif selected_years < 6:
        st.warning("Krótka historia szybciej reaguje na nowy reżim, ale daje mniej danych i bardziej niestabilną ocenę jakości.")
    else:
        st.info("Długa historia daje więcej przykładów, ale starsze zachowania rynku mogą być mniej przydatne dzisiaj.")

    st.subheader("Co program robi po kliknięciu Analizuj?")
    s1, s2, s3, s4 = st.columns(4)
    s1.markdown("<div class='pro-card'><h3>1. Dane</h3><p>Pobiera ceny, wolumen i benchmark rynku. Krypto liczy w skali 365 dni, giełdy w 252 sesjach.</p></div>", unsafe_allow_html=True)
    s2.markdown("<div class='pro-card'><h3>2. Cechy</h3><p>Buduje momentum, trend, RSI, MACD, ATR, tail ratio, presję ceny/wolumenu i relatywną siłę.</p></div>", unsafe_allow_html=True)
    s3.markdown("<div class='pro-card'><h3>3. Radar</h3><p>Liczy momentum, setup intelligence, risk/reward i Edge score, żeby ustalić priorytet dalszej analizy.</p></div>", unsafe_allow_html=True)
    s4.markdown("<div class='pro-card'><h3>4. Model zoo</h3><p>Porównuje regresję logistyczną, gradient boosting i ExtraTrees, a wagi dobiera przez walk-forward.</p></div>", unsafe_allow_html=True)

    st.subheader("Dlaczego aplikacja czasem mówi „wstrzymaj się”?")
    st.markdown("""
    To zabezpieczenie, nie awaria. Sygnał jest wygaszany, gdy:

    - **AUC jest blisko 0,50** — model nie rozróżnia wzrostów od spadków lepiej niż przypadek;
    - **Brier jest wysoki** — deklarowane prawdopodobieństwa nie sprawdzają się;
    - model nie pokonuje prostej strategii przewidywania częstszego kierunku;
    - rynek zmienił reżim i zależności z treningu nie działają w późniejszym okresie.

    Profesjonalny system powinien mieć prawo odpowiedzieć „nie wiem”. Wymuszanie sygnału każdego dnia zwykle tylko zwiększa liczbę fałszywych transakcji.
    """)

    st.subheader("Automatyczny monitor rynku")
    st.write("Launcher uruchamia obok aplikacji lekki proces o obniżonym priorytecie. Domyślnie co 6 godzin sprawdza reprezentatywne spółki GPW i USA, ETF-y oraz krypto. Jeśli ranking jest stary, niepełny albo w starym formacie, zakładka **Sygnały** potrafi sama wystartować świeży skan w tle. Wynik zapisuje lokalnie, dlatego dashboard pokazuje ostatni gotowy ranking i odświeża postęp bez przeładowywania całej aplikacji.")

    with st.expander("Znaczenie parametrów i skrótów"):
        st.markdown("""
        - **P(wzrost)** — skalibrowane prawdopodobieństwo dodatniego zwrotu po wybranym czasie.
        - **AUC** — 0,50 to brak przewagi; około 0,55 może oznaczać małą przewagę; 0,60+ jest interesujące, jeśli utrzymuje się w czasie.
        - **Brier** — błąd prognoz probabilistycznych; niżej jest lepiej, okolice 0,25 odpowiadają niepewności bliskiej 50/50.
        - **Zakres 90%** — szeroki przedział możliwego ruchu, a nie obietnica ceny docelowej.
        - **Benchmark** — rynek odniesienia: S&P 500, WIG20 Total Return albo Bitcoin dla altcoinów.
        - **Purge gap** — luka między treningiem i walidacją chroniąca przed podglądaniem przyszłości.
        - **Risk/reward** — relacja potencjalnego ruchu dodatniego do szacowanego downside z przedziału niepewności.
        - **Edge score** — priorytet analizy łączący oczekiwany ruch, P(wzrost), AUC/Brier, trend i zmienność.
        - **Setup score** — ocena spójności układu: momentum, trend, kontrola ryzyka, płynność i potwierdzenie modelu.
        - **Setup grade** — szybka etykieta jakości setupu: A/B/M/R/C. To skrót do ręcznej analizy, nie komenda transakcyjna.
        - **Teza radaru** — krótka lista powodów, dla których instrument znalazł się wysoko lub wymaga ostrożności.
        - **FAST** — lekki pierwszy przebieg techniczny po całym rynku; służy do odkrywania kandydatów, nie do zapisu sygnałów w Journalu.
        - **ML** — pełna walidowana prognoza modelu dla instrumentów z shortlisty; dopiero te wiersze mogą tworzyć directional signals.
        - **Deep score** — priorytet do pełnego ML, łączący Setup score, momentum, Edge score i kontrolę ryzyka.
        - **Target modelu** — kierunek close-to-close, zgodny w produkcji i backteście.
        - **SignalInputs** — wspólny pakiet finalnej decyzji: P(wzrost), oczekiwany ruch i jakość modelu. Produkcja i backtest używają tej samej bramki decyzyjnej.
        - **DecisionReason** — audyt decyzji: potwierdzony LONG/SHORT albo powód odrzucenia, np. niska jakość, brak przebicia progu lub konflikt probability z expected return.
        - **Aggregate Validation** — cięższy egzamin edge: foldy train/cal/test rozłożone po historii, osobny holdout, fingerprint danych, benchmarki, stress kosztów i analiza koncentracji wyniku.
        - **Next open execution** — sygnał powstaje po zamknięciu świecy, a test/Journal liczy wejście dopiero od następnego otwarcia.
        - **Auto scan** — można wyłączyć przez `MARKETSCOPE_AUTO_SCAN=0` albo zmienić rytm monitora przez `MARKETSCOPE_SCAN_INTERVAL_HOURS`.
        """)

with method:
    st.header("Metodologia i ograniczenia")
    st.markdown("""
### Silnik prognostyczny

Model korzysta z kilkudziesięciu cech: stóp zwrotu, RSI Wildera, MACD, ATR, pasm Bollingera, średnich 10–200 sesji, momentum, realized Sharpe, downside volatility, tail ratio, presji ceny/wolumenu, luk cenowych, anomalii wolumenu, odległości od wybicia/paniki oraz relatywnej siły względem rynku. Dla GPW kontekstem jest fundusz śledzący WIG20 Total Return, dla rynku amerykańskiego S&P 500, a dla altcoinów Bitcoin.

Kierunek liczy adaptacyjny ensemble: regularizowana regresja logistyczna, histogram gradient boosting i ExtraTrees. Wagi modeli są dobierane osobno dla każdego instrumentu i horyzontu na walk-forward validation z luką chroniącą przed podglądaniem przyszłości. Oczekiwany ruch liczy osobny ensemble regresyjny: Ridge, Random Forest, histogram gradient boosting i ExtraTrees. Prawdopodobieństwo jest kalibrowane i automatycznie ściągane do 50%, gdy AUC i Brier na późniejszym okresie nie potwierdzają jakości modelu. Po walidacji modele produkcyjne są ponownie trenowane na całej dostępnej historii.

Sygnały mają dwie warstwy. **Perełki momentum** łapią nietypowy ruch ceny, wybicia i silne przyspieszenie — to radar odkrywania okazji do dalszego sprawdzenia, szczególnie przy krypto. **Kandydaci ML** wymagają dodatkowo potwierdzonej jakości modelu poza próbką, dlatego pojawiają się rzadziej. Dzięki temu aplikacja nie gubi gorących ruchów, ale też nie udaje, że każdy szybki wzrost jest statystycznie potwierdzoną przewagą.

Widok **Dzisiejszy radar** dodaje trzecią warstwę: priorytet analizy. **Risk/reward** porównuje górny potencjał z downside z przedziału niepewności, a **Edge score** łączy oczekiwany ruch, P(wzrost), jakość AUC/Brier, trend techniczny i zmienność. **Setup intelligence** rozbija ranking na momentum, trend, kontrolę ryzyka, płynność i model edge, a **Teza radaru** tłumaczy najważniejsze powody.

Skaner działa dwustopniowo. **FAST Radar** lekko skanuje cały rynek i wybiera shortlistę przez **Deep score**. Potem **Deep ML** trenuje pełne modele tylko dla najlepszych kandydatów i zastępuje ich wiersze FAST wierszami ML. To skraca czas oczekiwania i zmniejsza szum, ale nadal nie jest poleceniem kupna — to kolejność, w jakiej warto sprawdzać setupy.

Backtest używa tej samej definicji celu co produkcja: model przewiduje kierunek **close-to-close**. Finalny sygnał przechodzi przez wspólny obiekt decyzyjny **SignalInputs**: skalibrowane prawdopodobieństwo, expected return z ensemble regresyjnego oraz jakość walidacji. Dzięki temu backtest nie testuje już luźno podobnej strategii opartej tylko o próg prawdopodobieństwa, tylko coraz wierniej odtwarza produkcyjną bramkę sygnału. Wynik finansowy jest liczony osobno bardziej konserwatywnie: sygnał pojawia się po zamknięciu dnia `t`, wejście jest liczone od otwarcia kolejnej sesji, a wynik uwzględnia koszt oraz uproszczony poślizg. Dzięki temu można odróżnić jakość prognozy kierunku od tego, czy dało się ją wykonać po realistycznej cenie.

**Aggregate Validation** to osobny, cięższy egzamin systemu. Przechodzi chronologicznie po wielu tickerach i horyzontach, w każdym foldzie trenuje tylko na wcześniejszej historii, zostawia purge gap, a później zapisuje każdą potencjalną decyzję: również te odrzucone. Produkcja, backtest i walidacja korzystają ze wspólnego **FittedForecastState**, więc probability, expected return, skill, quality i finalne **SignalInputs** powstają tą samą ścieżką. Parametr **refit_every** określa, jak często wewnątrz folda model jest trenowany ponownie: `1` jest najbliżej codziennego zachowania aplikacji, a `5/20` przyspiesza eksperyment diagnostyczny. Foldy rozwojowe są rozłożone po historii, a holdout jest raportowany osobno, żeby nie mieszać okresu egzaminacyjnego z okresem roboczym. Raport zachowuje metadane train/calibration/test, fingerprint danych — także benchmark/context histories — zakres dat instrumentów, identyfikator eksperymentu, `run_id`, checksum artefaktów, timestamp i commit hash. Porównuje MarketScope z always-long, buy-hold proxy, prostym momentum i samotną Logistic Regression, liczy stress kosztów 1×/2×/3×, niepokrywające się transakcje, ekspozycję w czasie trwania pozycji, drawdown, Sharpe/Sortino i koncentrację wyniku. To nadal nie jest pełny broker-grade symulator portfela z intrapozycyjną ścieżką ceny, ale jest uczciwsze niż liczenie całego wyniku w dniu sygnału.

### Ochrona przed fałszywie dobrym wynikiem

- cechy nie korzystają z przyszłych danych;
- trening zawsze poprzedza walidację;
- między treningiem i walidacją jest luka równa horyzontowi prognozy;
- dobór wag modeli korzysta z expanding walk-forward validation;
- backtest jest chronologiczny walk-forward i uwzględnia koszt transakcji;
- Aggregate Validation zapisuje powód każdej decyzji, także odrzuconych sygnałów;
- raport Aggregate Validation pokazuje benchmarki, holdout, stress kosztów oraz koncentrację zysków;
- manifest Aggregate Validation obejmuje commit, konfigurację, universe, fingerprint danych i zakres dat;
- aplikacja pokazuje przedział niepewności oraz jakość poza próbką;
- skaner nie składa zleceń, nie korzysta z dźwigni i nie obiecuje zysku.

### Czego model nie wie

Nie zna przyszłych raportów, decyzji banków centralnych, wydarzeń geopolitycznych ani nagłych problemów z płynnością. Dane Yahoo przez yfinance są odpowiednie do badań i paper tradingu; przed użyciem realnego kapitału potrzebne są licencjonowane dane oraz niezależna kontrola ryzyka.
""")

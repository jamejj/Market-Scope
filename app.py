from __future__ import annotations

import math

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from market_oracle.backtest import walk_forward_backtest
from market_oracle.catalog import (
    CATEGORIES, CRYPTO, ETF_CATEGORIES, category_options, crypto_options, etf_options,
)
from market_oracle.data import download_history, download_profile
from market_oracle.engine import analyze_asset, signal_label
from market_oracle.monitor import default_universe, load_snapshot, run_signal_scan
from market_oracle.search import search_assets


st.set_page_config(page_title="MarketScope PRO", page_icon="📈", layout="wide")
st.markdown("""
<style>
    .block-container {padding-top: 1.6rem; padding-bottom: 3rem; max-width: 1500px;}
    [data-testid="stMetric"] {background: rgba(120,120,120,.08); border: 1px solid rgba(120,120,120,.18); padding: 12px; border-radius: 12px;}
    div[data-testid="stTabs"] button {font-weight: 650;}
    .pro-card {padding: 18px; border: 1px solid rgba(120,120,120,.22); border-radius: 14px; background: rgba(120,120,120,.055); min-height: 145px;}
    .pro-card h3 {margin-top: 0;}
    .muted {opacity: .72;}
</style>
""", unsafe_allow_html=True)

if "training_years" not in st.session_state:
    st.session_state["training_years"] = 8
years = int(st.session_state["training_years"])


@st.cache_data(ttl=3600, show_spinner=False)
def cached_analysis(symbol: str, horizons: tuple[int, ...], years: int):
    return analyze_asset(symbol, horizons=horizons, years=years)


@st.cache_data(ttl=3600, show_spinner=False)
def cached_profile(symbol: str):
    return download_profile(symbol)


@st.cache_data(ttl=3600, show_spinner=False)
def cached_search(query: str):
    return search_assets(query)


def pct(value: float) -> str:
    return f"{value:.1%}"


def compact_number(value) -> str:
    if value is None or not isinstance(value, (int, float)) or not math.isfinite(value):
        return "—"
    for unit, divisor in (("bln", 1e12), ("mld", 1e9), ("mln", 1e6)):
        if abs(value) >= divisor:
            return f"{value / divisor:.2f} {unit}"
    return f"{value:,.0f}"


def profile_name(profile: dict, fallback: str) -> str:
    return profile.get("longName") or profile.get("shortName") or fallback


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

    five_day = result["forecasts"].get(5) or next(iter(result["forecasts"].values()))
    technical = result["technical"]
    trend_points = sum([
        technical["return_20d"] > 0, technical["rsi_14"] >= 50,
        technical["above_sma_50"], technical["above_sma_200"],
    ])
    trend_label = "POZYTYWNY" if trend_points >= 3 else ("NEGATYWNY" if trend_points <= 1 else "MIESZANY")
    summary_cols = st.columns(3)
    summary_cols[0].metric("Ostatnia cena", f"{result['last_price']:,.2f} {profile.get('currency', '')}".strip())
    summary_cols[1].metric("Trend techniczny", trend_label, help="Opis bieżącego trendu, nie prognoza przyszłej ceny.")
    summary_cols[2].metric("Jakość modelu 5-dniowego", five_day["quality"])

    if five_day["quality"] == "NISKA — BRAK PRZEWAGI":
        st.warning(
            f"**Brak potwierdzonej przewagi modelu.** W walidacji tygodniowej AUC wyniosło "
            f"{five_day['auc']:.3f}, a Brier {five_day['brier']:.3f}. Prognoza została celowo "
            "ściągnięta w stronę 50%. Najrozsądniejszy sygnał: obserwuj / brak pozycji."
        )
    elif five_day["probability_up"] >= 0.54:
        st.success(f"Model wykrywa przewagę wzrostową, a jakość walidacji jest: **{five_day['quality'].lower()}**.")
    elif five_day["probability_up"] <= 0.46:
        st.error(f"Model wykrywa przewagę spadkową, a jakość walidacji jest: **{five_day['quality'].lower()}**.")
    else:
        st.info(f"Sygnał neutralny. Jakość walidacji: **{five_day['quality'].lower()}**.")

    unit = "dzień" if symbol.endswith("-USD") else "sesja"
    horizon_names = {1: f"Następny {unit}", 5: "Najbliższy tydzień", 20: "Około miesiąca"}
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
    figure.update_layout(height=520, xaxis_rangeslider_visible=False, margin=dict(l=20, r=20, t=25, b=20), legend=dict(orientation="h"))
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
                result = cached_analysis(symbol, (1, 5, 20), years)
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


@st.fragment(run_every="30s")
def render_signal_dashboard() -> None:
    snapshot = load_snapshot()
    if not snapshot:
        st.info("Ranking nie został jeszcze policzony. Uruchom aplikację plikiem **Uruchom MarketScope.command** albo użyj przycisku pełnego skanu poniżej.")
        return

    status = snapshot.get("status")
    completed, total = snapshot.get("completed", 0), snapshot.get("total", 0)
    if status == "running":
        st.info(f"Monitor analizuje rynek w tle: **{completed}/{total}** instrumentów.")
        st.progress(completed / total if total else 0)
    elif status == "error":
        st.error(f"Ostatni skan został przerwany: {snapshot.get('error', 'nieznany błąd')}")
    else:
        updated = pd.Timestamp(snapshot["updated_at"])
        if updated.tzinfo is not None:
            updated = updated.tz_convert("Europe/Warsaw")
        st.success(f"Ranking gotowy · aktualizacja: **{updated.strftime('%Y-%m-%d %H:%M')}** · horyzont: **{snapshot.get('horizon', 20)} dni/sesji**")

    frame = pd.DataFrame(snapshot.get("records", []))
    if frame.empty:
        st.warning("Monitor nie ma jeszcze wystarczającej liczby ukończonych analiz.")
        return

    bullish_labels = {"SILNY KANDYDAT WZROSTOWY", "KANDYDAT WZROSTOWY"}
    bearish_labels = {"SILNE RYZYKO SPADKU", "RYZYKO SPADKU"}
    bullish = frame[frame["Ocena"].isin(bullish_labels)]
    bearish = frame[frame["Ocena"].isin(bearish_labels)]
    no_edge = frame[frame["Ocena"] == "BRAK SYGNAŁU"]
    summary = st.columns(4)
    summary[0].metric("Przeskanowane", len(frame))
    summary[1].metric("Kandydaci wzrostowi", len(bullish))
    summary[2].metric("Ryzyko spadku", len(bearish))
    summary[3].metric("Brak przewagi", len(no_edge))

    formats = {
        "Cena": "{:.2f}", "P(wzrost)": "{:.1%}", "Oczekiwany ruch": "{:.1%}",
        "AUC walidacji": "{:.3f}", "Pewność": "{:.1%}", "Zmienność roczna": "{:.1%}",
        "Max drawdown": "{:.1%}", "Score": "{:.2f}",
    }
    st.subheader("Kandydaci wzrostowi")
    if bullish.empty:
        st.info("Obecnie żaden instrument nie spełnia pełnego filtra wzrostowego. Monitor nie uzupełnia listy na siłę.")
    else:
        columns = ["Symbol", "Ocena", "P(wzrost)", "Oczekiwany ruch", "AUC walidacji", "Jakość modelu", "Pewność", "Zmienność roczna"]
        st.dataframe(bullish[columns].style.format(formats), use_container_width=True, hide_index=True)

    if not bearish.empty:
        with st.expander(f"Ryzyko spadku ({len(bearish)})"):
            columns = ["Symbol", "Ocena", "P(wzrost)", "Oczekiwany ruch", "AUC walidacji", "Jakość modelu"]
            st.dataframe(bearish[columns].style.format(formats), use_container_width=True, hide_index=True)

    with st.expander(f"Pełny ranking ({len(frame)} instrumentów)"):
        columns = ["Symbol", "Cena", "Ocena", "P(wzrost)", "Oczekiwany ruch", "AUC walidacji", "Jakość modelu", "Pewność", "Zmienność roczna", "Max drawdown", "Score"]
        st.dataframe(frame[columns].style.format(formats), use_container_width=True, hide_index=True)
        st.download_button("Pobierz ranking CSV", frame.to_csv(index=False).encode(), "marketscope_signals.csv", "text/csv")

    errors = snapshot.get("errors", {})
    if errors:
        st.caption(f"Pominięte instrumenty: {len(errors)}")


st.title("MarketScope PRO")
st.caption("Analityka akcji, ETF-ów i kryptowalut · sygnały ML · ryzyko · backtest")

home, stocks, etfs, crypto, radar, backtest, settings, method = st.tabs([
    "🏠 Start", "🏢 Spółki", "🧺 ETF-y", "₿ Krypto", "⭐ Sygnały", "🧪 Backtest", "⚙️ Model", "ℹ️ Metodologia",
])

with home:
    st.header("Centrum analizy rynku")
    st.write("Wybierz u góry klasę aktywów. Każdy instrument otrzyma prognozę na 1, 5 i 20 dni/sesji, ocenę wiarygodności, wykres, momentum i miary ryzyka.")
    c1, c2, c3 = st.columns(3)
    c1.markdown("<div class='pro-card'><h3>🏢 Spółki</h3><p>172 pozycje w katalogu, w tym GPW, USA, sektory i mniejsze firmy. Dostępna jest też wyszukiwarka globalna.</p></div>", unsafe_allow_html=True)
    c2.markdown("<div class='pro-card'><h3>🧺 ETF-y</h3><p>Szeroki rynek, sektory, obligacje, surowce, regiony świata, fundusze tematyczne i UCITS.</p></div>", unsafe_allow_html=True)
    c3.markdown("<div class='pro-card'><h3>₿ Krypto</h3><p>Najważniejsze kryptowaluty z osobną informacją o podwyższonym ryzyku i zmienności.</p></div>", unsafe_allow_html=True)
    st.subheader("Jak czytać prognozę?")
    st.markdown("""
    - **P(wzrost)** mówi, jak często model oczekuje ceny wyższej po danym horyzoncie.
    - **AUC** mierzy przewagę kierunkową poza próbką; około 0,50 oznacza brak przewagi.
    - **Brier** ocenia jakość prawdopodobieństw; mniej znaczy lepiej.
    - Jeśli walidacja jest słaba, aplikacja automatycznie wygasza sygnał do neutralnego.
    """)

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
    crypto_available = crypto_options()
    crypto_choice = st.selectbox("Kryptowaluta", list(crypto_available), key="crypto_choice")
    crypto_symbol = crypto_available[crypto_choice]
    st.caption(f"Wybrana para: `{crypto_symbol}`")
    analysis_action(crypto_symbol, "crypto_analysis", "crypto_analyze")

with radar:
    st.header("Automatyczny ranking rynku")
    st.write(f"Monitor śledzi **{len(default_universe())}** instrumentów z GPW, USA, ETF-ów i krypto. Po uruchomieniu aplikacji skanuje je w tle, a ranking odświeża co 12 godzin.")
    st.caption("To lista badawcza, nie automatyczna rekomendacja zakupu ani sprzedaży.")
    render_signal_dashboard()
    if st.button("Przelicz cały ranking teraz", key="signals_refresh", help="Pełny skan może potrwać kilka minut."):
        with st.spinner("Analizuję cały monitorowany rynek. Możesz pozostawić tę kartę otwartą…"):
            run_signal_scan(years=years)
        st.rerun()

with backtest:
    st.header("Backtest walk-forward")
    st.write("Model jest wielokrotnie trenowany wyłącznie na przeszłości, a następnie testowany na kolejnych, niewidzianych danych.")
    bt_symbol = st.text_input("Symbol", "SPY", help="Np. AAPL, CDR.WA, SPY, BTC-USD", key="bt_symbol").strip().upper()
    b1, b2, b3 = st.columns(3)
    bt_horizon = b1.selectbox("Horyzont", [1, 5, 20], index=1, key="bt_horizon")
    threshold = b2.slider("Minimalna pewność wejścia", 0.51, 0.70, 0.56, 0.01, key="bt_threshold")
    cost_bps = b3.number_input("Koszt transakcji (punkty bazowe)", 0.0, 100.0, 10.0, 1.0, key="bt_cost")
    if st.button("Uruchom test historyczny", type="primary", key="bt_run", use_container_width=True):
        try:
            with st.spinner("Symuluję prognozy out-of-sample…"):
                data = download_history(bt_symbol, years)
                curve, metrics = walk_forward_backtest(data, bt_horizon, threshold, cost_bps)
            st.session_state["bt_result"] = (bt_symbol, curve, metrics)
        except Exception as exc:
            st.error(str(exc))
    if "bt_result" in st.session_state and st.session_state["bt_result"][0] == bt_symbol:
        _, curve, metrics = st.session_state["bt_result"]
        cols = st.columns(6)
        values = [
            ("Łączny zwrot", pct(metrics["total_return"])), ("CAGR", pct(metrics["annual_return"])),
            ("Zmienność", pct(metrics["annual_volatility"])), ("Sharpe", f"{metrics['sharpe']:.2f}"),
            ("Max drawdown", pct(metrics["max_drawdown"])), ("Trafność", pct(metrics["hit_rate"])),
        ]
        for col, (label, value) in zip(cols, values):
            col.metric(label, value)
        st.line_chart(curve[["Equity", "BuyHold"]])
        st.caption(f"Aktywne sygnały: {metrics['trades']}. Wyniki historyczne nie gwarantują przyszłych.")

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
    s2.markdown("<div class='pro-card'><h3>2. Cechy</h3><p>Buduje momentum, trend, RSI, MACD, ATR, zmienność, wolumen i relatywną siłę.</p></div>", unsafe_allow_html=True)
    s3.markdown("<div class='pro-card'><h3>3. Walidacja</h3><p>Odkłada najnowszy fragment historii. Model nie widzi go podczas treningu oceny jakości.</p></div>", unsafe_allow_html=True)
    s4.markdown("<div class='pro-card'><h3>4. Refit</h3><p>Po ocenie jakości finalny model uczy się ponownie na całej dostępnej historii.</p></div>", unsafe_allow_html=True)

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
    st.write("Launcher uruchamia obok aplikacji lekki proces o obniżonym priorytecie. Co 12 godzin sprawdza reprezentatywne spółki GPW i USA, ETF-y oraz krypto. Wynik zapisuje lokalnie, dlatego zakładka **Sygnały** pokazuje od razu ostatni gotowy ranking i odświeża postęp bez przeładowywania całej aplikacji.")

    with st.expander("Znaczenie parametrów i skrótów"):
        st.markdown("""
        - **P(wzrost)** — skalibrowane prawdopodobieństwo dodatniego zwrotu po wybranym czasie.
        - **AUC** — 0,50 to brak przewagi; około 0,55 może oznaczać małą przewagę; 0,60+ jest interesujące, jeśli utrzymuje się w czasie.
        - **Brier** — błąd prognoz probabilistycznych; niżej jest lepiej, okolice 0,25 odpowiadają niepewności bliskiej 50/50.
        - **Zakres 90%** — szeroki przedział możliwego ruchu, a nie obietnica ceny docelowej.
        - **Benchmark** — rynek odniesienia: S&P 500, WIG20 Total Return albo Bitcoin dla altcoinów.
        - **Purge gap** — luka między treningiem i walidacją chroniąca przed podglądaniem przyszłości.
        """)

with method:
    st.header("Metodologia i ograniczenia")
    st.markdown("""
### Silnik prognostyczny

Model korzysta z ponad 30 cech: stóp zwrotu, RSI Wildera, MACD, ATR, pasm Bollingera, średnich 10–200 sesji, momentum, zmienności, luk cenowych, anomalii wolumenu oraz relatywnej siły względem rynku. Dla GPW kontekstem jest fundusz śledzący WIG20 Total Return, dla rynku amerykańskiego S&P 500, a dla altcoinów Bitcoin.

Kierunek łączy regularizowaną regresję logistyczną i gradient boosting. Ich proporcja jest dobierana osobno dla każdego instrumentu i horyzontu na wydzielonym okresie kalibracji. Oczekiwany ruch łączy Ridge z lasem losowym. Prawdopodobieństwo jest automatycznie ściągane do 50%, gdy AUC i Brier na późniejszym okresie nie potwierdzają jakości modelu. Po walidacji modele produkcyjne są ponownie trenowane na całej dostępnej historii.

### Ochrona przed fałszywie dobrym wynikiem

- cechy nie korzystają z przyszłych danych;
- trening zawsze poprzedza walidację;
- między treningiem i walidacją jest luka równa horyzontowi prognozy;
- backtest jest chronologiczny walk-forward i uwzględnia koszt transakcji;
- aplikacja pokazuje przedział niepewności oraz jakość poza próbką;
- skaner nie składa zleceń, nie korzysta z dźwigni i nie obiecuje zysku.

### Czego model nie wie

Nie zna przyszłych raportów, decyzji banków centralnych, wydarzeń geopolitycznych ani nagłych problemów z płynnością. Dane Yahoo przez yfinance są odpowiednie do badań i paper tradingu; przed użyciem realnego kapitału potrzebne są licencjonowane dane oraz niezależna kontrola ryzyka.
""")

# MarketScope

Profesjonalnie wyglądający, lokalny panel badawczy analizujący akcje, ETF-y i krypto. Generuje probabilistyczne prognozy kierunku dla 1, 5 i 20 sesji, ranking instrumentów, miary ryzyka oraz chronologiczny backtest walk-forward.

## Uruchomienie

Wymagany jest Python 3.9–3.13.

Na macOS można po prostu kliknąć dwukrotnie plik `Uruchom MarketScope.command`.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Interfejs otworzy się pod adresem `http://localhost:8501`.

Skaner można też uruchomić w terminalu:

```bash
python cli.py SPY QQQ AAPL BTC-USD ETH-USD --horizon 5
```

## Symbole

- USA: `AAPL`, `SPY`, `QQQ`
- GPW: `PKN.WA`, `PKO.WA`, `CDR.WA`
- indeksy: `WIG20.WA`, `^GSPC`
- krypto: `BTC-USD`, `ETH-USD`

## Ważne ograniczenia

To narzędzie badawcze, nie porada inwestycyjna. Model nie zna przyszłych wiadomości, zmian regulacyjnych ani zdarzeń płynnościowych. yfinance jest nieoficjalnym, społecznościowym klientem danych Yahoo i nie powinien być jedynym źródłem dla handlu realnym kapitałem. Najpierw używaj paper tradingu, a wyniki oceniaj na wielu reżimach rynku.

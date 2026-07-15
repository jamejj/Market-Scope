# MarketScope

Lokalny panel badawczy analizujący akcje, ETF-y i krypto. Generuje probabilistyczne prognozy kierunku dla 1, 5 i 20 dni/sesji, ranking instrumentów, miary ryzyka oraz chronologiczny backtest walk-forward.

## Najważniejsze funkcje

- osobne moduły spółek, ETF-ów i kryptowalut;
- ponad 170 spółek w katalogu oraz wyszukiwarka globalna;
- ensemble modeli klasyfikacyjnych i regresyjnych;
- kontekst szerokiego rynku oraz względna siła instrumentu;
- uczciwa walidacja chronologiczna z luką zapobiegającą wyciekowi danych;
- diagnostyka AUC, Brier, trafności i prostej strategii bazowej;
- ponowny trening modelu produkcyjnego na całej dostępnej historii;
- prawidłowa annualizacja 365 dni dla krypto i 252 sesji dla giełd;
- Today’s Radar z priorytetami analizy, momentum, risk/reward i Edge score;
- Setup Intelligence z rozbiciem na momentum, trend, kontrolę ryzyka, płynność, model edge i krótką tezę radaru;
- dwustopniowy skaner: szybki FAST Radar całego rynku oraz Deep ML tylko dla shortlisty;
- realistyczniejszy walk-forward backtest z tym samym pakietem wejść decyzyjnych co produkcja: probability, expected return i jakość modelu;
- wejście na następnym otwarciu, koszty oraz poślizg oddzielone od samej trafności prognozy;
- ranking rynku, analiza ryzyka i backtest walk-forward z kosztami transakcji;
- Signal Journal i Performance Lab z paper portfolio, kosztami, sizingiem, equity curve i drawdown;
- Aggregate Validation do zbiorczego sprawdzania edge według symbolu, rynku, horyzontu i folda, z osobnym holdoutem, fingerprintem danych, benchmarkami, stress testem kosztów i zapisem również odrzuconych sygnałów;
- zakładka **Model**, która tłumaczy ustawienia oraz neutralne sygnały;
- automatyczny monitor rynku zapisujący gotowy ranking domyślnie co 6 godzin;
- adaptacyjny ensemble, który dobiera udział modelu liniowego i nieliniowego na osobnym okresie kalibracji.

## Uruchomienie

Wymagany jest Python 3.9–3.13.

Na macOS można po prostu kliknąć dwukrotnie plik `Uruchom MarketScope.command`. Launcher uruchamia również monitor rynku pracujący w tle. Gdy ranking jest stary albo niepełny, zakładka **Sygnały** potrafi sama wystartować świeży skan.

Pełny ranking działa dwustopniowo. Najpierw szybki **FAST Radar** skanuje cały rynek technicznie, a następnie **Deep ML** odpala pełne modele tylko dla shortlisty najciekawszych instrumentów. W tym czasie dashboard pokazuje ranking częściowy i stopniowo zastępuje wiersze FAST wierszami ML.

Model w produkcji i backteście używa tej samej definicji targetu: kierunek close-to-close. Finalna decyzja przechodzi przez wspólny pakiet `SignalInputs`, czyli probability, expected return i jakość walidacji. Backtest i Journal nie zakładają już wejścia po tej samej cenie zamknięcia, na której powstał sygnał. Sygnał jest generowany po close dnia `t`, a wykonanie liczone jest od następnego open z kosztem i uproszczonym poślizgiem.

Moduł Aggregate Validation jest przeznaczony do cięższego egzaminu systemu: chronologicznie przechodzi po wielu instrumentach i horyzontach, zapisuje każdą obserwację decyzyjną oraz powód odrzucenia sygnału. Foldy przed holdoutem są wybierane równomiernie po historii, a nie tylko z początku danych. Raport zachowuje metadane train/calibration/test, osobne summary WALK_FORWARD i HOLDOUT, fingerprint wejściowych danych, zakres dat symboli, benchmarki always-long, buy-hold proxy, momentum i Logistic Regression, stress kosztów 1×/2×/3× oraz koncentrację wyniku. Artefakty walidacji można zapisać append-only do katalogu eksperymentów. To ma pomóc odpowiedzieć, czy przewaga utrzymuje się szerzej, a nie tylko na pojedynczym szczęśliwym tickerze.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Interfejs otworzy się pod adresem `http://localhost:8501`.

Rytm monitora można zmienić bez edycji kodu:

```bash
MARKETSCOPE_SCAN_INTERVAL_HOURS=3 python run_monitor.py
```

Autostart skanu w aplikacji można wyłączyć:

```bash
MARKETSCOPE_AUTO_SCAN=0 python -m streamlit run app.py
```

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

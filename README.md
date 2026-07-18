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
- Reality Check / Validation Lab, który z gotowych rekordów walidacji buduje konserwatywniejszy test: jedna pozycja na symbol, globalny limit pozycji, ledger gotówki i stałych slotów kapitału, dzienna krzywa kapitału, benchmark na tych samych datach/slotach oraz bootstrap przedziałów ufności;
- zamrożony **MarketScope 20D LONG Candidate v1** oraz append-only Forward Test Ledger do sprawdzania nowych sygnałów bez poprawiania historii;
- zakładka **Model**, która tłumaczy ustawienia oraz neutralne sygnały;
- automatyczny monitor rynku zapisujący gotowy ranking domyślnie co 6 godzin;
- adaptacyjny ensemble, który dobiera udział modelu liniowego i nieliniowego na osobnym okresie kalibracji.

## Uruchomienie

Wymagany jest Python 3.9–3.13.

Na macOS można po prostu kliknąć dwukrotnie plik `Uruchom MarketScope.command`. Launcher uruchamia również monitor rynku pracujący w tle. Gdy ranking jest stary albo niepełny, zakładka **Sygnały** potrafi sama wystartować świeży skan.

Pełny ranking działa dwustopniowo. Najpierw szybki **FAST Radar** skanuje cały rynek technicznie, a następnie **Deep ML** odpala pełne modele tylko dla shortlisty najciekawszych instrumentów. W tym czasie dashboard pokazuje ranking częściowy i stopniowo zastępuje wiersze FAST wierszami ML.

Model w produkcji i backteście używa tej samej definicji targetu: kierunek close-to-close. Finalna decyzja przechodzi przez wspólny pakiet `SignalInputs`, czyli probability, expected return i jakość walidacji. Backtest i Journal nie zakładają już wejścia po tej samej cenie zamknięcia, na której powstał sygnał. Sygnał jest generowany po close dnia `t`, a wykonanie liczone jest od następnego open z kosztem i uproszczonym poślizgiem.

Moduł Aggregate Validation jest przeznaczony do cięższego egzaminu systemu: chronologicznie przechodzi po wielu instrumentach i horyzontach, zapisuje każdą obserwację decyzyjną oraz powód odrzucenia sygnału. Produkcja, backtest i walidacja korzystają ze wspólnego `FittedForecastState`, więc probability, expected return, skill, quality i `SignalInputs` powstają tą samą ścieżką. Parametr `refit_every` kontroluje, jak często w foldzie model jest trenowany ponownie: `1` jest najbliżej codziennego zachowania aplikacji, większe wartości służą szybszym testom diagnostycznym. Foldy przed holdoutem są wybierane równomiernie po historii, a nie tylko z początku danych. Raport zachowuje metadane train/calibration/test, osobne summary WALK_FORWARD i HOLDOUT, fingerprint wszystkich wejść modelu — także kontekstów benchmarkowych — zakres dat symboli, benchmarki always-long, buy-hold proxy, momentum i Logistic Regression, stress kosztów 1×/2×/3× oraz koncentrację wyniku. Artefakty walidacji zapisują osobny `run_id`, checksum rekordów/raportu i append-only manifest log. To ma pomóc odpowiedzieć, czy przewaga utrzymuje się szerzej, a nie tylko na pojedynczym szczęśliwym tickerze.

Pełny diagnostyczny run można uruchomić komendą poniżej. To jest ciężki test — przy `refit_every=5` może trwać długo, bo runner ponownie trenuje cały stan modelu w wielu punktach historii. Runner ma cache danych, osobne joby `symbol × horizon`, statusy `PENDING/RUNNING/DONE/FAILED/INTERRUPTED` i job-level resume: ukończony job jest pomijany tylko wtedy, gdy zgadza się konfiguracja, fingerprint danych, checksum CSV oraz wersja pipeline’u/commit. Przerwany job startuje od początku, ale nie tracisz pozostałych ukończonych jobów. Rekordy jobów są zapisywane w `outputs/validation/jobs/...`, a cache historii w `outputs/validation/cache`.

```bash
.venv/bin/python run_validation.py --holdout-size 0 --refit-every 5 --max-folds 4
```

Szybszy smoke-test techniczny można uruchomić np. z `--refit-every 20 --max-folds 1`, ale nie należy interpretować go jako dowodu przewagi. Jeśli chcesz wymusić świeże dane, dodaj `--refresh-cache`; jeśli chcesz przeliczyć wszystko mimo gotowych jobów, dodaj `--no-resume`.

Po Aggregate Validation można odpalić konserwatywniejszy Reality Check na zapisanym CSV. Domyślnie skupia się na horyzoncie 20 dni, bo to był pierwszy kandydat na edge po wstępnym runie diagnostycznym:

```bash
.venv/bin/python run_reality_check.py --records outputs/validation/records_...csv --horizons 20
```

Ten raport nie trenuje niczego od nowa i nie optymalizuje progów. Bierze wyłącznie zapisane decyzje, wybiera pierwsze chronologiczne sygnały, odrzuca overlap na tym samym symbolu i domyślnie ogranicza portfel do 5 pozycji po 20% kapitału. Każdy slot ma własną wartość gotówki/pozycji: przy wejściu przypisywana jest konkretna kwota slotu, pozycja zmienia wartość bez codziennego rebalancingu, a po wyjściu cały slot wraca do gotówki. `portfolio_slots` określa wielkość slotu, a `max_positions` tylko maksymalną liczbę aktywnych pozycji, więc można testować np. 5 slotów po 20% przy limicie 3 pozycji. Benchmark używa tych samych dat wejścia/wyjścia, slotów kapitału i kosztów, a runner sprawdza, czy ceny `EntryPrice`/`ExitPrice` z CSV zgadzają się z cache `Open`. Przy brakach danych Reality Check domyślnie kończy się błędem, bo lepiej zatrzymać audyt niż policzyć fałszywie gładką krzywą.

Dla czystszego odczytu warto liczyć osobno rynek USA/ETF i krypto, bo mają inne kalendarze oraz annualizację:

```bash
.venv/bin/python run_reality_check.py --records outputs/validation/records_...csv --horizons 20 --markets USA,ETF --benchmark-symbol SPY
.venv/bin/python run_reality_check.py --records outputs/validation/records_...csv --horizons 20 --markets CRYPTO --benchmark-symbol BTC-USD
```

Po Reality Check Candidate v1 został zamrożony w `configs/marketscope_20d_long_candidate_v1.json`. Kontrakt v1 obejmuje: USA/ETF, tylko LONG, horyzont 20 sesji, próg `P(wzrost) >= 0.55`, dodatni expected return, jakość inną niż `NISKA — BRAK PRZEWAGI`, wejście po następnym open, 10 bps kosztu, 5 bps poślizgu, jedną pozycję na symbol, 5 slotów i maksymalnie 5 aktywnych pozycji. To jest zamrożona hipoteza badawcza — nie potwierdzony edge.

Forward Test Ledger zapisuje nowe sygnały jako append-only JSONL w `data/forward_ledger_candidate_v1.jsonl`. Stare rekordy nie są edytowane ani usuwane: każdy wpis ma `previous_event_hash` i `event_hash`, więc ręczna zmiana wcześniejszej linii jest wykrywana przy odczycie. Cykl zdarzeń to `SNAPSHOT_AUDIT`, `SIGNAL_OBSERVED`, potem `POSITION_ACCEPTED` albo `POSITION_SKIPPED`, następnie `ENTRY_FILLED`, a po 20 sesjach `POSITION_CLOSED`.

```bash
.venv/bin/python run_forward_test.py --record-snapshot --refresh
```

Pełny skan monitora dopisuje nowe sygnały Candidate v1 automatycznie po zakończeniu snapshotu. Forward ledger odrzuca snapshoty sprzed `frozen_at`, snapshoty bez zgodnego pipeline hash, brudne pliki modelu/decyzji oraz sygnały z dziennej świecy, która według czasu snapshotu nie jest jeszcze bezpiecznie zamknięta. Komenda powyżej jest przydatna do ręcznego dopisania sygnałów z ostatniego `data/signals.json` oraz do uzupełnienia wejść/wyjść, gdy nowe ceny są już dostępne.

Osobny, wcześniej nieużywany koszyk USA/ETF do kolejnego testu jest zapisany i zahashowany w `configs/unseen_usa_etf_v1.json`. Nie należy zmieniać modeli, cech ani progów Candidate v1 po zobaczeniu jego wyników. Runner unseen zapisuje preflight manifest i odpala walidację do osobnego folderu:

```bash
.venv/bin/python run_unseen_validation.py --dry-run
.venv/bin/python run_unseen_validation.py
```

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

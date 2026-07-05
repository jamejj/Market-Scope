#!/bin/zsh

cd "$(dirname "$0")" || exit 1

if [[ ! -x ".venv/bin/python" ]]; then
  echo "Pierwsze uruchomienie — przygotowuję aplikację…"
  python3 -m venv .venv || exit 1
  .venv/bin/python -m pip install -r requirements.txt || exit 1
fi

echo "Uruchamiam MarketScope PRO…"
echo "Aby zakończyć aplikację, zamknij to okno albo naciśnij Control+C."
nice -n 10 .venv/bin/python run_monitor.py &
MONITOR_PID=$!
trap 'kill "$MONITOR_PID" 2>/dev/null' EXIT INT TERM
.venv/bin/python -m streamlit run app.py

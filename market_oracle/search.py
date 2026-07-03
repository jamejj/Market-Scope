from __future__ import annotations

import yfinance as yf


ALLOWED_TYPES = {"EQUITY", "ETF", "MUTUALFUND", "CRYPTOCURRENCY", "INDEX"}


def search_assets(query: str, max_results: int = 15) -> list[dict[str, str]]:
    query = query.strip()
    if len(query) < 2:
        return []
    search = yf.Search(
        query, max_results=max_results, news_count=0, lists_count=0,
        include_cb=False, recommended=0, enable_fuzzy_query=True, timeout=15,
    )
    results = []
    seen = set()
    for quote in search.quotes:
        symbol = str(quote.get("symbol", "")).strip()
        quote_type = str(quote.get("quoteType", "")).upper()
        if not symbol or symbol in seen or quote_type not in ALLOWED_TYPES:
            continue
        seen.add(symbol)
        results.append({
            "symbol": symbol,
            "name": quote.get("longname") or quote.get("shortname") or symbol,
            "exchange": quote.get("exchDisp") or quote.get("exchange") or "—",
            "type": quote.get("typeDisp") or quote_type.title(),
        })
    return results

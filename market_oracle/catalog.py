"""Curated starter catalogue. Global search in search.py covers assets beyond it."""

CATEGORIES: dict[str, dict[str, str]] = {
    "GPW — największe spółki": {
        "PKO BP": "PKO.WA", "Orlen": "PKN.WA", "PZU": "PZU.WA", "Bank Pekao": "PEO.WA",
        "KGHM": "KGH.WA", "Dino Polska": "DNP.WA", "LPP": "LPP.WA", "Allegro": "ALE.WA",
        "CD Projekt": "CDR.WA", "Cyfrowy Polsat": "CPS.WA", "mBank": "MBK.WA",
        "Alior Bank": "ALR.WA", "ING Bank Śląski": "ING.WA", "Kruk": "KRU.WA",
        "Budimex": "BDX.WA", "Orange Polska": "OPL.WA", "Żabka Group": "ZAB.WA",
    },
    "GPW — średnie i mniejsze": {
        "XTB": "XTB.WA", "Benefit Systems": "BFT.WA", "Asseco Poland": "ACP.WA",
        "Asseco Business Solutions": "ABS.WA", "Auto Partner": "APR.WA", "AB": "ABE.WA",
        "Amica": "AMC.WA", "AmRest": "EAT.WA", "Archicom": "ARH.WA", "Atal": "1AT.WA",
        "Bank Millennium": "MIL.WA", "Bloober Team": "BLO.WA", "Celon Pharma": "CLN.WA",
        "Comarch": "CMR.WA", "Comp": "CMP.WA", "DataWalk": "DAT.WA", "Develia": "DVL.WA",
        "Dom Development": "DOM.WA", "GPW": "GPW.WA", "Grupa Kęty": "KTY.WA",
        "Huuuge": "HUG.WA", "Mabion": "MAB.WA", "Medicalgorithmics": "MDG.WA",
        "Neuca": "NEU.WA", "Rainbow Tours": "RBW.WA", "Ryvu Therapeutics": "RVU.WA",
        "Selvita": "SLV.WA", "Synektik": "SNT.WA", "Ten Square Games": "TEN.WA",
        "Text": "TXT.WA", "Vercom": "VRC.WA", "VRG": "VRG.WA", "Wirtualna Polska": "WPL.WA",
        "11 bit studios": "11B.WA",
    },
    "USA — technologia i półprzewodniki": {
        "Apple": "AAPL", "Microsoft": "MSFT", "Nvidia": "NVDA", "Alphabet": "GOOGL",
        "Meta Platforms": "META", "Amazon": "AMZN", "Tesla": "TSLA", "Broadcom": "AVGO",
        "AMD": "AMD", "Intel": "INTC", "Qualcomm": "QCOM", "Micron": "MU", "TSMC": "TSM",
        "ASML": "ASML", "Arm Holdings": "ARM", "Applied Materials": "AMAT",
        "Lam Research": "LRCX", "Marvell": "MRVL", "Palantir": "PLTR", "Oracle": "ORCL",
        "IBM": "IBM", "Cisco": "CSCO", "Adobe": "ADBE", "Salesforce": "CRM",
        "ServiceNow": "NOW", "Snowflake": "SNOW", "CrowdStrike": "CRWD",
        "Palo Alto Networks": "PANW", "Fortinet": "FTNT", "Shopify": "SHOP",
    },
    "USA — banki i finanse": {
        "JPMorgan Chase": "JPM", "Bank of America": "BAC", "Citigroup": "C", "Wells Fargo": "WFC",
        "Goldman Sachs": "GS", "Morgan Stanley": "MS", "BlackRock": "BLK", "Visa": "V",
        "Mastercard": "MA", "American Express": "AXP", "Charles Schwab": "SCHW",
        "S&P Global": "SPGI", "Coinbase": "COIN", "Robinhood": "HOOD", "SoFi": "SOFI",
    },
    "USA — zdrowie i biotechnologia": {
        "Eli Lilly": "LLY", "UnitedHealth": "UNH", "Johnson & Johnson": "JNJ", "Merck": "MRK",
        "AbbVie": "ABBV", "Pfizer": "PFE", "Amgen": "AMGN", "Gilead": "GILD",
        "Intuitive Surgical": "ISRG", "Moderna": "MRNA", "Regeneron": "REGN",
        "Vertex Pharmaceuticals": "VRTX", "CRISPR Therapeutics": "CRSP", "Hims & Hers": "HIMS",
    },
    "USA — przemysł i energia": {
        "Exxon Mobil": "XOM", "Chevron": "CVX", "ConocoPhillips": "COP", "Schlumberger": "SLB",
        "Caterpillar": "CAT", "Deere": "DE", "GE Aerospace": "GE", "RTX": "RTX",
        "Lockheed Martin": "LMT", "Northrop Grumman": "NOC", "Boeing": "BA", "Honeywell": "HON",
        "Union Pacific": "UNP", "UPS": "UPS", "FedEx": "FDX", "NextEra Energy": "NEE",
    },
    "USA — handel, media i usługi": {
        "Walmart": "WMT", "Costco": "COST", "Target": "TGT", "Home Depot": "HD",
        "McDonald's": "MCD", "Starbucks": "SBUX", "Coca-Cola": "KO", "PepsiCo": "PEP",
        "Nike": "NKE", "Disney": "DIS", "Netflix": "NFLX", "Uber": "UBER", "Airbnb": "ABNB",
        "Booking Holdings": "BKNG", "Spotify": "SPOT", "Reddit": "RDDT", "Duolingo": "DUOL",
        "Cava": "CAVA", "Celsius": "CELH",
    },
    "USA — mniejsze i spekulacyjne": {
        "Rocket Lab": "RKLB", "IonQ": "IONQ", "AST SpaceMobile": "ASTS", "Archer Aviation": "ACHR",
        "Rivian": "RIVN", "Lucid": "LCID", "Tempus AI": "TEM", "Quantum Computing": "QUBT",
        "Rigetti Computing": "RGTI", "Joby Aviation": "JOBY", "SoundHound AI": "SOUN",
        "UiPath": "PATH", "Unity Software": "U", "Aurora Innovation": "AUR",
    },
    "Świat — spółki notowane w USA": {
        "Novo Nordisk": "NVO", "AstraZeneca": "AZN", "Novartis": "NVS", "SAP": "SAP",
        "Toyota": "TM", "Sony": "SONY", "Alibaba": "BABA", "JD.com": "JD", "PDD Holdings": "PDD",
        "MercadoLibre": "MELI", "Sea Limited": "SE", "Baidu": "BIDU", "NIO": "NIO",
    },
}


def category_options(category: str) -> dict[str, str]:
    return {f"{name}  ·  {symbol}": symbol for name, symbol in CATEGORIES[category].items()}


ETF_CATEGORIES: dict[str, dict[str, str]] = {
    "Szeroki rynek USA": {
        "SPDR S&P 500": "SPY", "Vanguard S&P 500": "VOO", "iShares Core S&P 500": "IVV",
        "Vanguard Total Stock Market": "VTI", "Invesco Nasdaq 100": "QQQ",
        "iShares Russell 2000": "IWM", "SPDR Dow Jones": "DIA",
    },
    "Sektory i technologia": {
        "Technologia": "XLK", "Finanse": "XLF", "Energia": "XLE", "Zdrowie": "XLV",
        "Przemysł": "XLI", "Dobra konsumpcyjne": "XLY", "Dobra podstawowe": "XLP",
        "Utilities": "XLU", "Nieruchomości": "XLRE", "Półprzewodniki VanEck": "SMH",
        "Półprzewodniki iShares": "SOXX", "Cyberbezpieczeństwo": "CIBR",
    },
    "Świat i regiony": {
        "Rynki rozwinięte poza USA": "EFA", "Rynki wschodzące": "EEM", "Vanguard Emerging Markets": "VWO",
        "Europa": "VGK", "Japonia": "EWJ", "Indie": "INDA", "Chiny": "MCHI",
        "Brazylia": "EWZ", "Global All-World": "VT",
    },
    "Obligacje": {
        "USA 20+ lat": "TLT", "USA 7–10 lat": "IEF", "USA 1–3 lata": "SHY",
        "Cały rynek obligacji": "BND", "Obligacje korporacyjne": "LQD",
        "High Yield": "HYG", "Inflation-Protected": "TIP",
    },
    "Surowce": {
        "Złoto": "GLD", "Złoto — niższa opłata": "IAU", "Srebro": "SLV", "Ropa": "USO",
        "Surowce szeroko": "DBC", "Rolnictwo": "DBA", "Uran": "URA", "Miedź": "COPX",
    },
    "Tematyczne i wzrostowe": {
        "ARK Innovation": "ARKK", "Czysta energia": "ICLN", "Energia słoneczna": "TAN",
        "Robotyka i AI": "BOTZ", "Lit i baterie": "LIT", "Cloud Computing": "SKYY",
        "Genomika": "ARKG", "Fintech": "FINX", "Aerospace & Defense": "ITA",
    },
    "UCITS popularne w Europie": {
        "iShares Core S&P 500 UCITS": "SXR8.DE", "iShares Core MSCI World UCITS": "EUNL.DE",
        "Vanguard FTSE All-World UCITS": "VWCE.DE", "iShares Nasdaq 100 UCITS": "SXRV.DE",
        "iShares Core MSCI EM IMI UCITS": "IS3N.DE",
    },
}


CRYPTO: dict[str, str] = {
    "Bitcoin": "BTC-USD", "Ethereum": "ETH-USD", "Solana": "SOL-USD", "BNB": "BNB-USD",
    "XRP": "XRP-USD", "Cardano": "ADA-USD", "Dogecoin": "DOGE-USD", "Avalanche": "AVAX-USD",
    "Chainlink": "LINK-USD", "Polkadot": "DOT-USD", "Litecoin": "LTC-USD", "Bitcoin Cash": "BCH-USD",
    "Uniswap": "UNI-USD", "Aave": "AAVE-USD", "Cosmos": "ATOM-USD", "NEAR Protocol": "NEAR-USD",
    "Internet Computer": "ICP-USD", "Aptos": "APT-USD", "Sui": "SUI20947-USD",
    "Injective": "INJ-USD", "Arbitrum": "ARB11841-USD", "Optimism": "OP-USD",
    "Stellar": "XLM-USD", "Hedera": "HBAR-USD", "Filecoin": "FIL-USD", "Shiba Inu": "SHIB-USD",
}


def etf_options(category: str) -> dict[str, str]:
    return {f"{name}  ·  {symbol}": symbol for name, symbol in ETF_CATEGORIES[category].items()}


def crypto_options() -> dict[str, str]:
    return {f"{name}  ·  {symbol}": symbol for name, symbol in CRYPTO.items()}

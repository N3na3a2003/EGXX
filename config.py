from __future__ import annotations

from pathlib import Path


APP_NAME = "EGX Decision Support"
BASE_DIR = Path(__file__).resolve().parent
CACHE_DIR = BASE_DIR / ".cache"
CACHE_TTL_SECONDS = 10 * 60

DEFAULT_TIMEFRAME = "1y"
TIMEFRAMES = {
    "1d": {"period": "5d", "interval": "1d", "label": "1 Day"},
    "1wk": {"period": "1mo", "interval": "1wk", "label": "1 Week"},
    "1mo": {"period": "6mo", "interval": "1mo", "label": "1 Month"},
    "6mo": {"period": "6mo", "interval": "1d", "label": "6 Months"},
    "1y": {"period": "1y", "interval": "1d", "label": "1 Year"},
    "5y": {"period": "5y", "interval": "1wk", "label": "5 Years"},
}

DEFAULT_EGX_SYMBOLS = [
    "COMI",
    "HRHO",
    "TMGH",
    "EAST",
    "ABUK",
    "MFPC",
    "FWRY",
    "SWDY",
    "ORWE",
    "EKHO",
    "AUTO",
    "HELI",
    "EFID",
    "JUFO",
    "SKPC",
    "CIRA",
    "ORAS",
    "PHDC",
    "AMOC",
    "ETEL",
]

KNOWN_SYMBOL_NAMES = {
    "COMI": "Commercial International Bank",
    "HRHO": "EFG Holding",
    "TMGH": "Talaat Moustafa Group",
    "EAST": "Eastern Company",
    "ABUK": "Abou Qir Fertilizers",
    "MFPC": "Misr Fertilizers Production",
    "FWRY": "Fawry",
    "SWDY": "Elsewedy Electric",
    "ORWE": "Oriental Weavers",
    "EKHO": "Egypt Kuwait Holding",
    "AUTO": "GB Corp",
    "HELI": "Heliopolis Housing",
    "EFID": "Edita Food Industries",
    "JUFO": "Juhayna Food Industries",
    "SKPC": "Sidi Kerir Petrochemicals",
    "CIRA": "CIRA Education",
    "ORAS": "Orascom Construction",
    "PHDC": "Palm Hills Developments",
    "AMOC": "Alexandria Mineral Oils",
    "ETEL": "Telecom Egypt",
}

DEFAULT_ALERT_CONFIG = {
    "price_change_pct": 5.0,
    "rsi_low": 30.0,
    "rsi_high": 70.0,
    "volume_spike_multiple": 1.8,
}

RISK_FREE_RATE = 0.0

import requests
import pandas as pd
from datetime import datetime, timedelta, timezone
import config

GOLD_KEYWORDS = [
    'USD', 'Non-Farm', 'NFP', 'CPI', 'Inflation', 'Fed',
    'FOMC', 'Interest Rate', 'Powell', 'GDP', 'Unemployment',
    'Retail Sales', 'PPI', 'PCE', 'Treasury', 'ISM'
]

_cache_events  = []
_cache_updated = None
_CACHE_TTL_MIN = 60


def fetch_events():
    global _cache_events, _cache_updated
    now = datetime.now(timezone.utc)
    if _cache_updated and (now - _cache_updated).seconds < _CACHE_TTL_MIN * 60:
        return _cache_events
    try:
        url = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"
        r   = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=10)
        r.raise_for_status()
        raw    = r.json()
        events = []
        for item in raw:
            if item.get('impact') != 'High':
                continue
            currency = item.get('currency', '')
            title    = item.get('title', '')
            if currency != 'USD' and not any(k in title for k in GOLD_KEYWORDS):
                continue
            try:
                dt = datetime.fromisoformat(item['date'].replace('Z', '+00:00'))
            except Exception:
                continue
            events.append({'time': dt, 'title': title,
                           'currency': currency, 'impact': 'High'})
        _cache_events  = events
        _cache_updated = now
        print(f"[NewsFilter] Loaded {len(events)} high-impact events")
        return events
    except Exception as e:
        print(f"[NewsFilter] Failed to fetch: {e}")
        return _cache_events


def is_blackout(dt=None, blackout_min=None):
    if dt is None:
        dt = datetime.now(timezone.utc)
    if blackout_min is None:
        blackout_min = config.NEWS_BLACKOUT_MINUTES
    for event in fetch_events():
        diff = abs((dt - event['time']).total_seconds() / 60)
        if diff <= blackout_min:
            return True, event['title']
    return False, ""


def next_event(within_hours=4):
    now      = datetime.now(timezone.utc)
    upcoming = [
        e for e in fetch_events()
        if 0 <= (e['time'] - now).total_seconds() <= within_hours * 3600
    ]
    return min(upcoming, key=lambda e: e['time']) if upcoming else None


def get_blackout_times(lookback_days=1, forward_days=1):
    now    = datetime.now(timezone.utc)
    result = []
    for e in fetch_events():
        diff = (e['time'] - now).total_seconds() / 86400
        if -lookback_days <= diff <= forward_days:
            result.append(e['time'])
    return result

from __future__ import annotations

import re
from typing import Any


def to_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(str(value).replace(",", "."))
    except (TypeError, ValueError):
        return None


def to_int(value: Any) -> int | None:
    number = to_float(value)
    return int(number) if number is not None else None


def parse_score(score: str | None) -> tuple[int | None, int | None]:
    if not score:
        return None, None
    match = re.search(r"(\d+)\s*/\s*(\d+)", str(score))
    if not match:
        return None, None
    return int(match.group(1)), int(match.group(2))


def extract_lifetime_stats(stats_payload: dict[str, Any]) -> dict[str, float | int | None]:
    lifetime = stats_payload.get("lifetime") or {}
    matches = to_int(lifetime.get("Matches"))
    wins = to_int(lifetime.get("Wins"))
    winrate = to_float(lifetime.get("Win Rate %"))
    kd = to_float(lifetime.get("Average K/D Ratio"))
    return {
        "lifetime_matches": matches,
        "lifetime_wins": wins,
        "lifetime_winrate": winrate,
        "lifetime_kd": kd,
    }


def extract_match_round(stats_payload: dict[str, Any]) -> dict[str, Any] | None:
    rounds = stats_payload.get("rounds") or []
    if not rounds:
        return None
    return rounds[0]


def faction_key_from_team_id(match_payload: dict[str, Any], team_id: str | None) -> str | None:
    teams = match_payload.get("teams") or {}
    for faction, team in teams.items():
        if team.get("faction_id") == team_id or team.get("team_id") == team_id:
            return faction
    return None


def normalize_winner_faction(
    match_payload: dict[str, Any],
    round_payload: dict[str, Any] | None,
) -> str | None:
    results = match_payload.get("results") or {}
    winner = results.get("winner")
    if winner in {"faction1", "faction2"}:
        return winner

    if round_payload:
        winner_id = (round_payload.get("round_stats") or {}).get("Winner")
        mapped = faction_key_from_team_id(match_payload, winner_id)
        if mapped:
            return mapped

    return None


def estimate_premade_size(is_premade: bool | None, roster_size: int = 5) -> int:
    if is_premade is True:
        return roster_size
    if is_premade is False:
        return 1
    return 1


def player_game_info(player_payload: dict[str, Any], game: str) -> dict[str, Any]:
    games = player_payload.get("games") or {}
    return games.get(game) or {}


def extract_map_segment_stats(
    stats_payload: dict[str, Any],
    map_name: str | None,
) -> dict[str, float | int | None]:
    """Pull winrate / KD for a specific map from lifetime segments."""
    if not map_name:
        return {}
    target = map_name.lower().replace("de_", "")
    for segment in stats_payload.get("segments") or []:
        label = str(segment.get("label") or "").lower()
        if target not in label and map_name.lower() not in label:
            continue
        stats = segment.get("stats") or {}
        return {
            "map_matches": to_int(stats.get("Matches")),
            "map_wins": to_int(stats.get("Wins")),
            "map_winrate": to_float(stats.get("Win Rate %")),
            "map_kd": to_float(stats.get("Average K/D Ratio")),
        }
    return {}


def parse_match_id_from_url(url_or_id: str) -> str:
    """
    Accept a raw match id or a FACEIT room/match URL.

    Examples:
      1-xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx
      https://www.faceit.com/en/cs2/room/1-...
      https://www.faceit.com/en/cs2/matchroom/1-...
    """
    text = (url_or_id or "").strip()
    if not text:
        raise ValueError("Empty match URL / id.")

    patterns = [
        r"(?:room|matchroom|match)/([0-9a-fA-F-]{20,})",
        r"\b(1-[0-9a-fA-F-]{20,})\b",
        r"\b([0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12})\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(1)

    if re.fullmatch(r"[0-9a-fA-F-]{20,}", text):
        return text

    raise ValueError(f"Could not parse FACEIT match id from: {url_or_id}")


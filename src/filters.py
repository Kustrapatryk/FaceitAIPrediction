from __future__ import annotations

from typing import Any


PREMIUM_NAME_HINTS = (
    "premium",
    "super match",
    "supermatch",
    "5v5 premium",
    "premium match",
)


def competition_looks_premium(competition_name: str | None) -> bool:
    name = (competition_name or "").lower()
    return any(hint in name for hint in PREMIUM_NAME_HINTS)


def roster_premium_ratio(match_payload: dict[str, Any]) -> float:
    players = []
    for team in (match_payload.get("teams") or {}).values():
        players.extend(team.get("roster") or [])
    if not players:
        return 0.0
    premiumish = 0
    for player in players:
        membership = str(player.get("membership") or "").lower()
        if membership in {"premium", "plus", "subscription"} or "premium" in membership:
            premiumish += 1
    return premiumish / len(players)


def is_faceit_matchmaking(match_payload: dict[str, Any]) -> bool:
    if match_payload.get("competition_type") != "matchmaking":
        return False
    organizer = str(match_payload.get("organizer_id") or "").lower()
    return organizer in {"", "faceit", "faceit-com", "faceit.com"}


def team_average_elos(match_payload: dict[str, Any]) -> tuple[float | None, float | None]:
    teams = match_payload.get("teams") or {}
    out: list[float | None] = []
    for faction in ("faction1", "faction2"):
        team = teams.get(faction) or {}
        rating = (team.get("stats") or {}).get("rating")
        try:
            out.append(float(rating) if rating is not None else None)
        except (TypeError, ValueError):
            out.append(None)
    return out[0], out[1]


def passes_premium_high_elo_filters(
    match_payload: dict[str, Any],
    *,
    min_team_avg_elo: float = 2000.0,
    require_named_premium_queue: bool = False,
    min_premium_membership_ratio: float = 0.0,
) -> tuple[bool, str]:
    """
    Keep only FACEIT matchmaking lobbies where BOTH teams average Elo >= threshold.

    Note: the public Data API often labels Premium Match / Super Match still as
    'Europe 5v5 Queue'. Named-queue detection is optional; Elo gate is primary.
    """
    if not is_faceit_matchmaking(match_payload):
        return False, "not_faceit_matchmaking"

    name = match_payload.get("competition_name")
    if require_named_premium_queue and not competition_looks_premium(name):
        return False, "competition_name_not_premium"

    if min_premium_membership_ratio > 0:
        ratio = roster_premium_ratio(match_payload)
        if ratio < min_premium_membership_ratio:
            return False, f"premium_membership_ratio_{ratio:.2f}"

    elo1, elo2 = team_average_elos(match_payload)
    if elo1 is None or elo2 is None:
        return False, "missing_team_avg_elo"
    if elo1 < min_team_avg_elo or elo2 < min_team_avg_elo:
        return False, f"team_avg_elo_below_{min_team_avg_elo:g}"

    return True, "ok"

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sqlalchemy import select

from src.config import FACEIT_GAME, FORM_WINDOW, MODELS_DIR
from src.db import Match, MatchPlayer, get_session, init_db
from src.faceit_api import FaceitAPIError, FaceitClient
from src.features import FEATURE_COLUMNS, PlayerHistoryAgg, team_features_from_aggs
from src.parsing import (
    parse_match_id_from_url,
    player_game_info,
    to_float,
    to_int,
)


def load_model(model_name: str = "win_model") -> dict[str, Any]:
    path = MODELS_DIR / f"{model_name}.joblib"
    if not path.exists():
        raise FileNotFoundError(f"Model not found: {path}. Run `python main.py train` first.")
    return joblib.load(path)


def model_path(model_name: str = "win_model") -> Path:
    return MODELS_DIR / f"{model_name}.joblib"


def _predict_win_chance(payload: dict[str, Any], features: dict[str, float]) -> float:
    model = payload["model"]
    columns = payload.get("feature_columns", FEATURE_COLUMNS)
    frame = pd.DataFrame([{col: float(features.get(col, 0.0)) for col in columns}])
    value = float(model.predict(frame)[0])
    return max(0.0, min(1.0, value))


def predict_match_win_proba(
    features: dict[str, float],
    *,
    model_name: str = "win_model",
) -> dict[str, float]:
    payload = load_model(model_name)
    p_f1 = _predict_win_chance(payload, features)
    return {"faction1": p_f1, "faction2": 1.0 - p_f1}


@dataclass
class LivePrediction:
    match_id: str
    status: str | None
    map_name: str | None
    region: str | None
    faction1_name: str | None
    faction2_name: str | None
    faction1_win_chance: float
    faction2_win_chance: float
    faceit_win_prob_f1: float | None
    form_window: int
    features: dict[str, float]
    roster: dict[str, list[dict[str, Any]]]


def _elo_from_profile(client: FaceitClient, player_id: str, game: str, skill: int | None) -> float | None:
    try:
        profile = client.get_player(player_id)
        game_info = player_game_info(profile, game)
        elo = to_int(game_info.get("faceit_elo"))
        if elo is not None:
            return float(elo)
        skill = to_int(game_info.get("skill_level")) or skill
    except FaceitAPIError:
        pass
    if skill is not None:
        return float(1 + (skill - 1) * 250)
    return None


def _agg_from_db(
    player_id: str,
    *,
    form_window: int,
    current_elo: float | None = None,
) -> PlayerHistoryAgg | None:
    init_db()
    session = get_session()
    try:
        rows = session.execute(
            select(MatchPlayer, Match)
            .join(Match, Match.match_id == MatchPlayer.match_id)
            .where(MatchPlayer.player_id == player_id)
            .where(Match.winner_faction.is_not(None))
            .order_by(Match.started_at.asc())
        ).all()
        if not rows:
            return None

        agg = PlayerHistoryAgg(form_window=form_window)
        for mp, match in rows:
            won = 1 if mp.result == "win" else 0
            if mp.result is None and match.winner_faction:
                won = 1 if mp.faction == match.winner_faction else 0
            elo = float(mp.elo) if mp.elo is not None else None
            if elo is None and mp.skill_level is not None:
                elo = float(1 + (mp.skill_level - 1) * 250)
            agg.add(
                won=won,
                kd=float(mp.kd) if mp.kd is not None else None,
                elo=elo,
                map_name=match.map_name,
            )

        if current_elo is not None:
            # Attach current Elo to the newest snapshot so team Elo uses live rating.
            if agg.snapshots:
                agg.snapshots[-1].elo = current_elo
            else:
                agg.add(won=0, elo=current_elo)
        return agg
    finally:
        session.close()


def _parse_recent_stat_item(item: dict[str, Any]) -> tuple[int, float | None, str | None]:
    stats = item.get("stats") or item
    # FACEIT recent-game stats payloads vary; try common keys.
    result = stats.get("Result") or stats.get("result")
    won = 1 if str(result) in {"1", "win", "Win", "true", "True"} else 0
    if result is None:
        # Sometimes winner is encoded differently; leave as loss-ish default only if explicit.
        won = 0

    kd = to_float(stats.get("K/D Ratio") or stats.get("kd"))
    kills = to_int(stats.get("Kills"))
    deaths = to_int(stats.get("Deaths"))
    if kd is None and kills is not None and deaths is not None:
        kd = kills / deaths if deaths > 0 else float(kills)

    map_name = (
        stats.get("Map")
        or stats.get("map")
        or (item.get("c10") if isinstance(item.get("c10"), str) else None)
    )
    return won, kd, map_name


def _agg_from_api(
    client: FaceitClient,
    player_id: str,
    *,
    game: str,
    form_window: int,
    roster_skill_level: int | None = None,
) -> PlayerHistoryAgg:
    agg = PlayerHistoryAgg(form_window=form_window)
    current_elo = _elo_from_profile(client, player_id, game, roster_skill_level)

    # Best source for live form: last N matches with per-match KD.
    try:
        payload = client.get_player_recent_game_stats(
            player_id, game, limit=min(100, form_window)
        )
        items = list(payload.get("items") or [])
        # API usually returns newest first; build chronological order.
        for item in reversed(items[-form_window:]):
            won, kd, map_name = _parse_recent_stat_item(item)
            agg.add(won=won, kd=kd, elo=current_elo, map_name=map_name)
    except FaceitAPIError:
        items = []

    if agg.matches == 0:
        # Fallback: win/loss from match history only (no KD).
        try:
            history = client.get_player_history(player_id, game, limit=min(100, form_window))
            items = list(history.get("items") or [])
            for item in reversed(items[-form_window:]):
                results = item.get("results") or {}
                winner = results.get("winner")
                player_faction = None
                for faction, team in (item.get("teams") or {}).items():
                    for p in team.get("players") or []:
                        if p.get("player_id") == player_id:
                            player_faction = faction
                            break
                won = 1 if player_faction and winner and player_faction == winner else 0
                agg.add(won=won, kd=None, elo=current_elo, map_name=None)
        except FaceitAPIError:
            pass

    if agg.matches == 0 and current_elo is not None:
        agg.add(won=0, elo=current_elo)

    return agg


def _guess_map_name(match_payload: dict[str, Any]) -> str | None:
    voting = match_payload.get("voting") or {}
    map_block = voting.get("map") or {}
    pick = map_block.get("pick")
    if isinstance(pick, list) and pick:
        return str(pick[0])
    if isinstance(pick, str) and pick:
        return pick
    entities = map_block.get("entities") or []
    if entities:
        return str(
            entities[0].get("game_map_id")
            or entities[0].get("name")
            or entities[0].get("class_name")
            or ""
        )
    return None


def predict_live_match(
    url_or_id: str,
    *,
    model_name: str = "win_model",
    form_window: int = FORM_WINDOW,
) -> LivePrediction:
    """
    Live flow:
      1) read 10 players from the room
      2) for each player take stats from last `form_window` matches
      3) compare team form and regress win chance
    Never uses the current match's own box-score.
    """
    match_id = parse_match_id_from_url(url_or_id)
    client = FaceitClient()
    match_payload = client.get_match(match_id)
    teams = match_payload.get("teams") or {}
    if "faction1" not in teams or "faction2" not in teams:
        raise RuntimeError("Match payload is missing faction1/faction2 rosters.")

    map_name = _guess_map_name(match_payload)
    faceit_wp = to_float(((teams.get("faction1") or {}).get("stats") or {}).get("winProbability"))

    roster_out: dict[str, list[dict[str, Any]]] = {"faction1": [], "faction2": []}
    aggs: dict[str, list[PlayerHistoryAgg]] = {"faction1": [], "faction2": []}

    for faction in ("faction1", "faction2"):
        team = teams[faction]
        for player in team.get("roster") or []:
            player_id = player.get("player_id")
            if not player_id:
                continue
            nickname = player.get("nickname")
            skill = to_int(player.get("game_skill_level"))
            current_elo = _elo_from_profile(client, player_id, FACEIT_GAME, skill)

            # Prefer local DB form when we already crawled this player deeply.
            agg = _agg_from_db(player_id, form_window=form_window, current_elo=current_elo)
            if agg is None or agg.matches < max(5, form_window // 4):
                agg = _agg_from_api(
                    client,
                    player_id,
                    game=FACEIT_GAME,
                    form_window=form_window,
                    roster_skill_level=skill,
                )

            aggs[faction].append(agg)
            roster_out[faction].append(
                {
                    "player_id": player_id,
                    "nickname": nickname,
                    "elo": agg.elo,
                    "form_kd": agg.kd,
                    "form_wr": agg.wr,
                    "form_matches": agg.matches,
                }
            )

    features = team_features_from_aggs(
        aggs["faction1"],
        aggs["faction2"],
        map_name=map_name,
    )

    payload = load_model(model_name)
    p_f1 = _predict_win_chance(payload, features)

    return LivePrediction(
        match_id=match_id,
        status=match_payload.get("status"),
        map_name=map_name,
        region=match_payload.get("region"),
        faction1_name=(teams.get("faction1") or {}).get("name"),
        faction2_name=(teams.get("faction2") or {}).get("name"),
        faction1_win_chance=p_f1,
        faction2_win_chance=1.0 - p_f1,
        faceit_win_prob_f1=faceit_wp,
        form_window=form_window,
        features=features,
        roster=roster_out,
    )

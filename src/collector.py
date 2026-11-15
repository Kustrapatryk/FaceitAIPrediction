from __future__ import annotations

from collections import deque
from datetime import datetime, timezone
from typing import Any, Iterable

from sqlalchemy import delete, select, text
from sqlalchemy.orm import Session
from tqdm import tqdm

from src.config import FACEIT_GAME
from src.db import Match, MatchPlayer, Player, get_engine, get_session, init_db
from src.faceit_api import FaceitAPIError, FaceitClient
from src.filters import passes_premium_high_elo_filters, team_average_elos
from src.parsing import (
    extract_lifetime_stats,
    extract_match_round,
    normalize_winner_faction,
    parse_score,
    player_game_info,
    to_float,
    to_int,
)


def clear_training_data() -> dict[str, int]:
    """Wipe matches/players used for training (keeps schema)."""
    init_db()
    engine = get_engine()
    with engine.begin() as conn:
        # Order matters for FK constraints.
        deleted_mp = conn.execute(text("DELETE FROM match_players")).rowcount
        deleted_m = conn.execute(text("DELETE FROM matches")).rowcount
        deleted_p = conn.execute(text("DELETE FROM players")).rowcount
        deleted_c = conn.execute(text("DELETE FROM collection_cursors")).rowcount
    return {
        "match_players": int(deleted_mp or 0),
        "matches": int(deleted_m or 0),
        "players": int(deleted_p or 0),
        "collection_cursors": int(deleted_c or 0),
    }


class MatchCollector:
    def __init__(
        self,
        client: FaceitClient | None = None,
        session: Session | None = None,
        game: str = FACEIT_GAME,
        *,
        min_team_avg_elo: float = 2000.0,
        require_named_premium_queue: bool = False,
        min_premium_membership_ratio: float = 0.0,
    ) -> None:
        init_db()
        self.client = client or FaceitClient()
        self.session = session or get_session()
        self.game = game
        self.min_team_avg_elo = min_team_avg_elo
        self.require_named_premium_queue = require_named_premium_queue
        self.min_premium_membership_ratio = min_premium_membership_ratio
        self._owns_session = session is None
        self.stats = {
            "matches_stored": 0,
            "matches_skipped_filter": 0,
            "matches_skipped_other": 0,
            "players_visited": 0,
        }

    def close(self) -> None:
        if self._owns_session:
            self.session.close()

    def upsert_player_from_profile(self, player_id: str, *, fetch_stats: bool = True) -> Player:
        profile = self.client.get_player(player_id)
        game_info = player_game_info(profile, self.game)

        player = self.session.get(Player, player_id) or Player(player_id=player_id)
        player.nickname = profile.get("nickname")
        player.country = profile.get("country")
        player.elo = to_int(game_info.get("faceit_elo"))
        player.skill_level = to_int(game_info.get("skill_level"))
        player.faceit_url = profile.get("faceit_url")
        player.updated_at = datetime.now(timezone.utc)

        if fetch_stats:
            try:
                stats = self.client.get_player_stats(player_id, self.game)
                lifetime = extract_lifetime_stats(stats)
                player.lifetime_matches = lifetime["lifetime_matches"]  # type: ignore[assignment]
                player.lifetime_wins = lifetime["lifetime_wins"]  # type: ignore[assignment]
                player.lifetime_winrate = lifetime["lifetime_winrate"]  # type: ignore[assignment]
                player.lifetime_kd = lifetime["lifetime_kd"]  # type: ignore[assignment]
            except FaceitAPIError:
                pass

        self.session.merge(player)
        self.session.commit()
        return player

    def seed_from_nickname(self, nickname: str) -> str:
        profile = self.client.get_player_by_nickname(nickname)
        player_id = profile["player_id"]
        self.upsert_player_from_profile(player_id, fetch_stats=True)
        return player_id

    def seed_from_rankings(
        self,
        region: str = "EU",
        limit: int = 100,
        *,
        min_player_elo: int = 2000,
    ) -> list[str]:
        """Seed crawler with high-Elo ranked players (better Premium/high-Elo yield)."""
        player_ids: list[str] = []
        offset = 0
        # FACEIT requires offset to be a multiple of limit.
        page_size = 100
        while len(player_ids) < limit:
            payload = self.client.get_rankings(
                self.game, region, offset=offset, limit=page_size
            )
            items = payload.get("items") or []
            if not items:
                break
            for item in items:
                player_id = item.get("player_id")
                elo = to_int(item.get("faceit_elo"))
                if not player_id or elo is None or elo < min_player_elo:
                    continue
                player = self.session.get(Player, player_id) or Player(player_id=player_id)
                player.nickname = item.get("nickname")
                player.country = item.get("country")
                player.elo = elo
                player.skill_level = to_int(item.get("game_skill_level"))
                player.updated_at = datetime.now(timezone.utc)
                self.session.merge(player)
                player_ids.append(player_id)
                if len(player_ids) >= limit:
                    break
            self.session.commit()
            offset += page_size
            if len(items) < page_size:
                break
        return player_ids

    def collect_player_history_match_ids(
        self,
        player_id: str,
        *,
        max_matches: int = 100,
    ) -> list[str]:
        match_ids: list[str] = []
        offset = 0
        page_size = 20

        while len(match_ids) < max_matches:
            limit = min(page_size, max_matches - len(match_ids))
            payload = self.client.get_player_history(
                player_id, self.game, offset=offset, limit=limit
            )
            items = payload.get("items") or []
            if not items:
                break
            for item in items:
                # Cheap prefilter from history payload when possible.
                if item.get("competition_type") and item.get("competition_type") != "matchmaking":
                    continue
                match_id = item.get("match_id")
                if match_id and match_id not in match_ids:
                    match_ids.append(match_id)
            offset += len(items)
            if len(items) < limit:
                break
        return match_ids

    def store_match(self, match_id: str, *, force: bool = False) -> Match | None:
        existing = self.session.get(Match, match_id)
        if existing and existing.winner_faction and not force:
            return existing

        try:
            match_payload = self.client.get_match(match_id)
        except FaceitAPIError as exc:
            if exc.status_code == 404:
                return None
            raise

        if match_payload.get("game") not in {None, self.game}:
            if match_payload.get("game") and match_payload.get("game") != self.game:
                self.stats["matches_skipped_other"] += 1
                return None

        status = match_payload.get("status")
        if status and status.lower() not in {"finished", "closed"}:
            self.stats["matches_skipped_other"] += 1
            return None

        ok, reason = passes_premium_high_elo_filters(
            match_payload,
            min_team_avg_elo=self.min_team_avg_elo,
            require_named_premium_queue=self.require_named_premium_queue,
            min_premium_membership_ratio=self.min_premium_membership_ratio,
        )
        if not ok:
            self.stats["matches_skipped_filter"] += 1
            return None

        try:
            stats_payload = self.client.get_match_stats(match_id)
        except FaceitAPIError as exc:
            if exc.status_code in {404, 403}:
                stats_payload = {}
            else:
                raise

        round_payload = extract_match_round(stats_payload)
        round_stats = (round_payload or {}).get("round_stats") or {}
        score_f1, score_f2 = parse_score(round_stats.get("Score"))
        winner = normalize_winner_faction(match_payload, round_payload)
        team1_avg_elo, team2_avg_elo = team_average_elos(match_payload)
        faceit_wp_f1 = to_float(
            (((match_payload.get("teams") or {}).get("faction1") or {}).get("stats") or {}).get(
                "winProbability"
            )
        )

        match = existing or Match(match_id=match_id)
        match.game = match_payload.get("game") or self.game
        match.region = match_payload.get("region")
        match.map_name = round_stats.get("Map")
        match.competition_type = match_payload.get("competition_type")
        match.competition_name = match_payload.get("competition_name")
        match.started_at = to_int(match_payload.get("started_at"))
        match.finished_at = to_int(match_payload.get("finished_at"))
        match.status = status
        match.winner_faction = winner
        match.score_faction1 = score_f1
        match.score_faction2 = score_f2
        # Premade intentionally unused for this dataset.
        match.premade_faction1 = None
        match.premade_faction2 = None
        match.premade_size_faction1 = None
        match.premade_size_faction2 = None
        match.team1_avg_elo = team1_avg_elo
        match.team2_avg_elo = team2_avg_elo
        match.faceit_win_prob_f1 = faceit_wp_f1
        match.collected_at = datetime.now(timezone.utc)
        self.session.merge(match)

        self.session.execute(delete(MatchPlayer).where(MatchPlayer.match_id == match_id))
        stats_by_player = self._index_player_stats(round_payload)
        teams = match_payload.get("teams") or {}

        for faction, team in teams.items():
            for roster_player in team.get("roster") or []:
                player_id = roster_player.get("player_id")
                if not player_id:
                    continue
                player = self.session.get(Player, player_id) or Player(player_id=player_id)
                player.nickname = roster_player.get("nickname") or player.nickname
                player.skill_level = to_int(roster_player.get("game_skill_level")) or player.skill_level
                player.updated_at = datetime.now(timezone.utc)
                self.session.merge(player)

                pstats = stats_by_player.get(player_id, {})
                kills = to_int(pstats.get("Kills"))
                deaths = to_int(pstats.get("Deaths"))
                assists = to_int(pstats.get("Assists"))
                kd = to_float(pstats.get("K/D Ratio"))
                if kd is None and kills is not None and deaths is not None:
                    kd = kills / deaths if deaths > 0 else float(kills)

                result = None
                if winner:
                    result = "win" if faction == winner else "loss"

                self.session.add(
                    MatchPlayer(
                        match_id=match_id,
                        player_id=player_id,
                        faction=faction,
                        nickname=roster_player.get("nickname"),
                        skill_level=to_int(roster_player.get("game_skill_level")),
                        elo=None,
                        kills=kills,
                        deaths=deaths,
                        assists=assists,
                        kd=kd,
                        adr=to_float(pstats.get("ADR") or pstats.get("Average Damage per Round")),
                        headshots_pct=to_float(pstats.get("Headshots %")),
                        mvps=to_int(pstats.get("MVPs")),
                        result=result,
                    )
                )

        self.session.commit()
        return match

    @staticmethod
    def _index_player_stats(round_payload: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
        indexed: dict[str, dict[str, Any]] = {}
        if not round_payload:
            return indexed
        for team in round_payload.get("teams") or []:
            for player in team.get("players") or []:
                player_id = player.get("player_id")
                if player_id:
                    indexed[player_id] = player.get("player_stats") or {}
        return indexed

    def crawl(
        self,
        seed_player_ids: Iterable[str],
        *,
        max_matches: int = 500,
        history_per_player: int = 40,
        max_players: int = 200,
        refresh_player_stats: bool = False,
    ) -> dict[str, int]:
        queue: deque[str] = deque(dict.fromkeys(seed_player_ids))
        seen_players: set[str] = set()
        seen_matches: set[str] = set(self.session.scalars(select(Match.match_id)).all())
        stored = 0

        progress = tqdm(total=max_matches, desc="Collecting premium/high-Elo matches")
        while queue and stored < max_matches and len(seen_players) < max_players:
            player_id = queue.popleft()
            if player_id in seen_players:
                continue
            seen_players.add(player_id)

            if refresh_player_stats or self.session.get(Player, player_id) is None:
                try:
                    self.upsert_player_from_profile(player_id, fetch_stats=True)
                except FaceitAPIError:
                    continue

            try:
                match_ids = self.collect_player_history_match_ids(
                    player_id, max_matches=history_per_player
                )
            except FaceitAPIError:
                continue

            for match_id in match_ids:
                if match_id in seen_matches:
                    continue
                seen_matches.add(match_id)
                try:
                    match = self.store_match(match_id)
                except FaceitAPIError:
                    self.stats["matches_skipped_other"] += 1
                    continue

                if match is None or not match.winner_faction:
                    # store_match already counted filter/other skips when None
                    if match is not None and not match.winner_faction:
                        self.stats["matches_skipped_other"] += 1
                    continue

                stored += 1
                self.stats["matches_stored"] += 1
                progress.update(1)

                for mp in self.session.scalars(
                    select(MatchPlayer.player_id).where(MatchPlayer.match_id == match_id)
                ):
                    if mp not in seen_players:
                        queue.append(mp)

                if stored >= max_matches:
                    break

        progress.close()
        self.stats["players_visited"] = len(seen_players)
        self.stats["queue_remaining"] = len(queue)
        return dict(self.stats)

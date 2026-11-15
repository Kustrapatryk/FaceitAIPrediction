from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session

from src.config import FORM_WINDOW
from src.db import Match, MatchPlayer, get_session, init_db

FEATURE_COLUMNS = [
    "avg_elo_f1",
    "avg_elo_f2",
    "delta_avg_elo",
    "avg_kd_f1",
    "avg_kd_f2",
    "delta_avg_kd",
    "avg_wr_f1",
    "avg_wr_f2",
    "delta_avg_wr",
    "avg_map_wr_f1",
    "avg_map_wr_f2",
    "delta_map_wr",
    "avg_recent_wr_f1",
    "avg_recent_wr_f2",
    "delta_recent_wr",
    "elo_std_f1",
    "elo_std_f2",
    "games_played_f1",
    "games_played_f2",
]


@dataclass
class MatchSnapshot:
    """One finished match for a player — raw history fuel for form windows."""

    won: int
    kd: float | None = None
    elo: float | None = None
    map_name: str | None = None


@dataclass
class PlayerHistoryAgg:
    """
    Rolling player form built ONLY from previous matches.

    Stats used by the model (kd/wr/map_wr/...) come from the last
    `form_window` matches, never from the match being predicted.
    """

    form_window: int = FORM_WINDOW
    snapshots: list[MatchSnapshot] = field(default_factory=list)

    def add(
        self,
        *,
        won: int,
        kd: float | None = None,
        elo: float | None = None,
        map_name: str | None = None,
    ) -> None:
        self.snapshots.append(MatchSnapshot(won=won, kd=kd, elo=elo, map_name=map_name))

    @property
    def matches(self) -> int:
        return len(self.form_snapshots)

    @property
    def form_snapshots(self) -> list[MatchSnapshot]:
        return self.snapshots[-self.form_window :]

    @property
    def wr(self) -> float:
        window = self.form_snapshots
        if not window:
            return 0.5
        return sum(s.won for s in window) / len(window)

    @property
    def kd(self) -> float:
        values = [s.kd for s in self.form_snapshots if s.kd is not None]
        return float(np.mean(values)) if values else 1.0

    @property
    def elo(self) -> float:
        # Prefer most recent known Elo/skill proxy in the form window, else earlier history.
        for snap in reversed(self.snapshots):
            if snap.elo is not None:
                return float(snap.elo)
        return 1500.0

    def recent_wr(self, n: int | None = None) -> float:
        size = n or min(10, self.form_window)
        window = self.snapshots[-size:]
        if not window:
            return 0.5
        return sum(s.won for s in window) / len(window)

    def map_wr(self, map_name: str | None) -> float:
        if not map_name:
            return self.wr
        on_map = [s for s in self.form_snapshots if s.map_name == map_name]
        if not on_map:
            return self.wr
        return sum(s.won for s in on_map) / len(on_map)

    # Compatibility helpers used by older predict blending code.
    @property
    def recent_results(self) -> list[int]:
        return [s.won for s in self.snapshots]

    @recent_results.setter
    def recent_results(self, values: list[int]) -> None:
        # Keep Elo/KD when replacing only W/L form from another source.
        existing = list(self.snapshots)
        self.snapshots = []
        for i, won in enumerate(values):
            kd = existing[i].kd if i < len(existing) else None
            elo = existing[i].elo if i < len(existing) else None
            map_name = existing[i].map_name if i < len(existing) else None
            self.add(won=int(won), kd=kd, elo=elo, map_name=map_name)

    @property
    def map_matches(self) -> dict[str, int]:
        counts: dict[str, int] = defaultdict(int)
        for snap in self.form_snapshots:
            if snap.map_name:
                counts[snap.map_name] += 1
        return counts

    @property
    def map_wins(self) -> dict[str, int]:
        counts: dict[str, int] = defaultdict(int)
        for snap in self.form_snapshots:
            if snap.map_name and snap.won:
                counts[snap.map_name] += 1
        return counts


def _safe_mean(values: list[float], default: float) -> float:
    return float(np.mean(values)) if values else default


def _safe_std(values: list[float]) -> float:
    return float(np.std(values)) if len(values) > 1 else 0.0


def team_features_from_aggs(
    f1: list[PlayerHistoryAgg],
    f2: list[PlayerHistoryAgg],
    *,
    map_name: str | None = None,
    recent_window: int = 10,
) -> dict[str, float]:
    """Compare team form from each player's last N matches."""

    def side_stats(players: list[PlayerHistoryAgg]) -> dict[str, float]:
        elos = [p.elo for p in players]
        kds = [p.kd for p in players]
        wrs = [p.wr for p in players]
        map_wrs = [p.map_wr(map_name) for p in players]
        recent = [p.recent_wr(recent_window) for p in players]
        games = [float(p.matches) for p in players]
        return {
            "avg_elo": _safe_mean(elos, 1500.0),
            "elo_std": _safe_std(elos),
            "avg_kd": _safe_mean(kds, 1.0),
            "avg_wr": _safe_mean(wrs, 0.5),
            "avg_map_wr": _safe_mean(map_wrs, 0.5),
            "avg_recent_wr": _safe_mean(recent, 0.5),
            "games_played": _safe_mean(games, 0.0),
        }

    s1 = side_stats(f1)
    s2 = side_stats(f2)
    return {
        "avg_elo_f1": s1["avg_elo"],
        "avg_elo_f2": s2["avg_elo"],
        "delta_avg_elo": s1["avg_elo"] - s2["avg_elo"],
        "avg_kd_f1": s1["avg_kd"],
        "avg_kd_f2": s2["avg_kd"],
        "delta_avg_kd": s1["avg_kd"] - s2["avg_kd"],
        "avg_wr_f1": s1["avg_wr"],
        "avg_wr_f2": s2["avg_wr"],
        "delta_avg_wr": s1["avg_wr"] - s2["avg_wr"],
        "avg_map_wr_f1": s1["avg_map_wr"],
        "avg_map_wr_f2": s2["avg_map_wr"],
        "delta_map_wr": s1["avg_map_wr"] - s2["avg_map_wr"],
        "avg_recent_wr_f1": s1["avg_recent_wr"],
        "avg_recent_wr_f2": s2["avg_recent_wr"],
        "delta_recent_wr": s1["avg_recent_wr"] - s2["avg_recent_wr"],
        "elo_std_f1": s1["elo_std"],
        "elo_std_f2": s2["elo_std"],
        "games_played_f1": s1["games_played"],
        "games_played_f2": s2["games_played"],
    }


class FeatureBuilder:
    """Builds point-in-time team features from historical matches."""

    def __init__(
        self,
        session: Session | None = None,
        form_window: int = FORM_WINDOW,
        recent_window: int = 10,
    ) -> None:
        init_db()
        self.session = session or get_session()
        self.form_window = form_window
        self.recent_window = recent_window
        self._owns_session = session is None

    def close(self) -> None:
        if self._owns_session:
            self.session.close()

    def build_dataset(self, min_prior_matches: int = 5) -> pd.DataFrame:
        matches = self.session.scalars(
            select(Match)
            .where(Match.winner_faction.is_not(None))
            .where(Match.started_at.is_not(None))
            .order_by(Match.started_at.asc())
        ).all()

        match_players = self.session.scalars(select(MatchPlayer)).all()
        players_by_match: dict[str, list[MatchPlayer]] = defaultdict(list)
        for row in match_players:
            players_by_match[row.match_id].append(row)

        history: dict[str, PlayerHistoryAgg] = defaultdict(
            lambda: PlayerHistoryAgg(form_window=self.form_window)
        )
        rows: list[dict[str, float | int | str | None]] = []

        for match in matches:
            roster = players_by_match.get(match.match_id, [])
            f1 = [p for p in roster if p.faction == "faction1"]
            f2 = [p for p in roster if p.faction == "faction2"]
            if len(f1) < 5 or len(f2) < 5:
                if len(f1) < 3 or len(f2) < 3:
                    self._update_history(history, match, roster)
                    continue

            # Require enough prior form samples across the lobby.
            prior_counts = [history[p.player_id].matches for p in f1 + f2]
            if sum(1 for c in prior_counts if c >= 1) < min_prior_matches:
                self._update_history(history, match, roster)
                continue

            # CRITICAL: features first, THEN append this match to history.
            features = self._team_features(history, f1, f2, match)
            features["match_id"] = match.match_id
            features["started_at"] = match.started_at
            features["map_name"] = match.map_name
            features["region"] = match.region
            features["label"] = 1 if match.winner_faction == "faction1" else 0
            features["faceit_win_prob_f1"] = match.faceit_win_prob_f1
            rows.append(features)

            self._update_history(history, match, roster)

        if not rows:
            return pd.DataFrame(columns=[*FEATURE_COLUMNS, "match_id", "label"])

        return pd.DataFrame(rows)

    def _team_features(
        self,
        history: dict[str, PlayerHistoryAgg],
        f1: list[MatchPlayer],
        f2: list[MatchPlayer],
        match: Match,
    ) -> dict[str, float]:
        return team_features_from_aggs(
            [history[p.player_id] for p in f1],
            [history[p.player_id] for p in f2],
            map_name=match.map_name,
            recent_window=self.recent_window,
        )

    def _update_history(
        self,
        history: dict[str, PlayerHistoryAgg],
        match: Match,
        roster: list[MatchPlayer],
    ) -> None:
        for player in roster:
            agg = history[player.player_id]
            won = 1 if player.result == "win" else 0
            if player.result is None and match.winner_faction:
                won = 1 if player.faction == match.winner_faction else 0

            elo = float(player.elo) if player.elo is not None else None
            if elo is None and player.skill_level is not None:
                elo = float(1 + (player.skill_level - 1) * 250)

            agg.add(
                won=won,
                kd=float(player.kd) if player.kd is not None else None,
                elo=elo,
                map_name=match.map_name,
            )

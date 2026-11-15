from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sqlalchemy import func, select

from src.config import MODELS_DIR
from src.db import Match, MatchPlayer, Player, get_session, init_db
from src.faceit_api import FaceitClient
from src.predict import predict_live_match


def main() -> None:
    init_db()
    session = get_session()

    n_matches = session.scalar(select(func.count()).select_from(Match)) or 0
    n_players = session.scalar(select(func.count()).select_from(Player)) or 0
    n_mp = session.scalar(select(func.count()).select_from(MatchPlayer)) or 0
    below = session.scalar(
        select(func.count()).select_from(Match).where(
            (Match.team1_avg_elo < 2000) | (Match.team2_avg_elo < 2000)
        )
    ) or 0
    non_mm = session.scalar(
        select(func.count()).select_from(Match).where(Match.competition_type != "matchmaking")
    ) or 0

    print("=== DB sanity ===")
    print(f"matches={n_matches} players={n_players} match_players={n_mp}")
    print(f"matches_with_team_elo_below_2000={below}")
    print(f"non_matchmaking_matches={non_mm}")
    assert below == 0, "Found matches below Elo filter"
    assert non_mm == 0, "Found non-matchmaking matches"
    assert n_matches >= 50, f"Need more matches for meaningful eval, got {n_matches}"

    meta_path = MODELS_DIR / "win_model.meta.json"
    assert meta_path.exists(), "Missing trained model meta"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    print("=== Train holdout metrics ===")
    print(json.dumps(meta, indent=2))

    # Live predict on a few newest stored matches (API + model path).
    newest = session.scalars(
        select(Match).order_by(Match.started_at.desc()).limit(3)
    ).all()
    print("=== Live predict smoke tests ===")
    for match in newest:
        result = predict_live_match(match.match_id, form_window=int(meta.get("form_window", 20)))
        print(
            f"{match.match_id} map={result.map_name} "
            f"model={result.faction1_win_chance:.3f}/{result.faction2_win_chance:.3f} "
            f"faceit={result.faceit_win_prob_f1} actual_winner={match.winner_faction}"
        )
        assert 0.0 <= result.faction1_win_chance <= 1.0
        assert abs(result.faction1_win_chance + result.faction2_win_chance - 1.0) < 1e-6

    # One fresh match from current rankings (may still be finished).
    client = FaceitClient()
    top = client.get_rankings("cs2", "EU", limit=1)["items"][0]
    hist = client.get_player_history(top["player_id"], "cs2", limit=1)
    live_id = hist["items"][0]["match_id"]
    live = predict_live_match(live_id, form_window=int(meta.get("form_window", 20)))
    print(
        f"fresh:{live_id} status={live.status} "
        f"model={live.faction1_win_chance:.3f}/{live.faction2_win_chance:.3f}"
    )
    print("ALL TESTS PASSED")


if __name__ == "__main__":
    main()

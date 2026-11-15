from __future__ import annotations

import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.db import Match, MatchPlayer, Player, get_session, init_db
from src.train import train_model


def main() -> None:
    init_db()
    session = get_session()

    for mid in [f"smoke-{i}" for i in range(120)]:
        match = session.get(Match, mid)
        if match:
            session.delete(match)
    session.commit()

    players: list[str] = []
    for i in range(40):
        pid = f"p{i}"
        players.append(pid)
        session.merge(
            Player(
                player_id=pid,
                nickname=f"Player{i}",
                elo=1000 + i * 20,
                skill_level=min(10, 1 + i // 4),
            )
        )
    session.commit()

    random.seed(0)
    t0 = 1_700_000_000
    for i in range(120):
        mid = f"smoke-{i}"
        f1 = random.sample(players[:20], 5)
        f2 = random.sample(players[20:], 5)
        strength1 = sum(int(p[1:]) for p in f1)
        strength2 = sum(int(p[1:]) for p in f2)
        winner = "faction1" if strength1 + random.randint(-30, 30) >= strength2 else "faction2"
        session.merge(
            Match(
                match_id=mid,
                game="cs2",
                region="EU",
                map_name=random.choice(["de_mirage", "de_inferno", "de_dust2"]),
                started_at=t0 + i * 3600,
                finished_at=t0 + i * 3600 + 3000,
                status="finished",
                winner_faction=winner,
                score_faction1=16 if winner == "faction1" else random.randint(5, 14),
                score_faction2=16 if winner == "faction2" else random.randint(5, 14),
                premade_faction1=random.random() < 0.3,
                premade_faction2=random.random() < 0.3,
                premade_size_faction1=5 if random.random() < 0.3 else 1,
                premade_size_faction2=5 if random.random() < 0.3 else 1,
            )
        )
        for faction, ids in (("faction1", f1), ("faction2", f2)):
            for pid in ids:
                won = faction == winner
                session.add(
                    MatchPlayer(
                        match_id=mid,
                        player_id=pid,
                        faction=faction,
                        nickname=pid,
                        skill_level=5,
                        elo=1200 + int(pid[1:]) * 20,
                        kills=random.randint(10, 30),
                        deaths=random.randint(10, 30),
                        assists=random.randint(0, 10),
                        kd=round(random.uniform(0.7, 1.5), 2),
                        result="win" if won else "loss",
                    )
                )
    session.commit()
    session.close()

    path, metrics = train_model(
        min_prior_matches=3,
        test_size=0.25,
        calibrate=False,
        model_name="smoke_model",
    )
    print(path)
    print(metrics)


if __name__ == "__main__":
    main()

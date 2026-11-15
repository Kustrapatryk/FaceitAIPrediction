from __future__ import annotations

from typing import Optional

import typer
from rich import print

from src.collector import MatchCollector, clear_training_data
from src.config import FORM_WINDOW, ensure_dirs
from src.db import init_db
from src.predict import predict_live_match
from src.train import train_model

app = typer.Typer(help="FACEIT match win-probability pipeline", no_args_is_help=True)


@app.command("init-db")
def init_db_cmd() -> None:
    """Create SQLite database and tables."""
    ensure_dirs()
    init_db()
    print("[green]Database initialized.[/green]")


@app.command("reset-db")
def reset_db_cmd(
    yes: bool = typer.Option(False, "--yes", help="Confirm wiping all collected training rows."),
) -> None:
    """Delete all players/matches/match_players used for training."""
    if not yes:
        raise typer.BadParameter("Refusing to wipe DB without --yes")
    ensure_dirs()
    deleted = clear_training_data()
    print("[yellow]Cleared training data:[/yellow]", deleted)


@app.command("collect")
def collect_cmd(
    nickname: Optional[str] = typer.Option(
        None, help="Optional seed nickname in addition to rankings."
    ),
    region: str = typer.Option("EU", help="Ranking region seed: EU, NA, ..."),
    ranking_limit: int = typer.Option(80, help="How many high-Elo ranking players to seed."),
    min_team_avg_elo: float = typer.Option(
        2000.0, help="Keep match only if BOTH teams have avg Elo >= this value."
    ),
    min_player_elo: int = typer.Option(2000, help="Seed rankings only with players above this Elo."),
    require_named_premium_queue: bool = typer.Option(
        False,
        "--require-named-premium-queue",
        help="Also require competition_name to contain Premium/Super (often missing in API).",
    ),
    min_premium_membership_ratio: float = typer.Option(
        0.0,
        help="Optional: fraction of roster with premium/plus membership (0 disables).",
    ),
    max_matches: int = typer.Option(300, help="Target number of finished matches to store."),
    history_per_player: int = typer.Option(50, help="Match history depth per visited player."),
    max_players: int = typer.Option(200, help="Max unique players to visit."),
) -> None:
    """
    Collect FACEIT matchmaking matches where both teams average Elo >= threshold.

    Premade size is ignored. Public API rarely exposes a distinct 'Premium Match'
    queue name, so Elo gate (+ optional membership/name filters) is the practical filter.
    """
    ensure_dirs()
    collector = MatchCollector(
        min_team_avg_elo=min_team_avg_elo,
        require_named_premium_queue=require_named_premium_queue,
        min_premium_membership_ratio=min_premium_membership_ratio,
    )
    try:
        seeds: list[str] = []
        if nickname:
            player_id = collector.seed_from_nickname(nickname)
            seeds.append(player_id)
            print(f"Seeded nickname [cyan]{nickname}[/cyan] -> {player_id}")

        ranked = collector.seed_from_rankings(
            region=region,
            limit=ranking_limit,
            min_player_elo=min_player_elo,
        )
        seeds.extend(ranked)
        print(
            f"Seeded [cyan]{len(ranked)}[/cyan] players from {region} rankings "
            f"(Elo >= {min_player_elo})."
        )

        seeds = list(dict.fromkeys(seeds))
        stats = collector.crawl(
            seeds,
            max_matches=max_matches,
            history_per_player=history_per_player,
            max_players=max_players,
        )
        print(stats)
    finally:
        collector.close()


@app.command("train")
def train_cmd(
    min_prior_matches: int = typer.Option(5, help="Min prior player-match samples before using a row."),
    test_size: float = typer.Option(0.2, help="Holdout fraction (time-based)."),
    model_name: str = typer.Option("win_model", help="Output model basename in models/."),
    form_window: int = typer.Option(FORM_WINDOW, help="Use each player's last N matches as form."),
    no_calibrate: bool = typer.Option(False, help="Disable probability calibration."),
) -> None:
    """Train a regression model on pre-match form from each player's last N games."""
    path, metrics = train_model(
        min_prior_matches=min_prior_matches,
        test_size=test_size,
        calibrate=not no_calibrate,
        model_name=model_name,
        form_window=form_window,
    )
    print(f"[green]Saved regression model:[/green] {path}")
    print(metrics)


@app.command("predict")
def predict_cmd(
    match: str = typer.Argument(..., help="FACEIT match URL or match id."),
    model_name: str = typer.Option("win_model", help="Model basename in models/."),
    form_window: int = typer.Option(FORM_WINDOW, help="Last N matches used as player form."),
    show_features: bool = typer.Option(False, "--features", help="Print feature vector."),
    show_roster: bool = typer.Option(False, "--roster", help="Print per-player form stats used."),
) -> None:
    """Predict win chances from pre-match form (last N matches), never from the live box-score."""
    result = predict_live_match(match, model_name=model_name, form_window=form_window)
    print(f"[bold]Match:[/bold] {result.match_id}")
    print(
        f"[bold]Status:[/bold] {result.status}  "
        f"[bold]Region:[/bold] {result.region}  "
        f"[bold]Map:[/bold] {result.map_name}  "
        f"[bold]Form window:[/bold] last {result.form_window}"
    )
    print(
        f"[cyan]{result.faction1_name or 'faction1'}[/cyan]: "
        f"[green]{result.faction1_win_chance * 100:.1f}%[/green]"
    )
    print(
        f"[magenta]{result.faction2_name or 'faction2'}[/magenta]: "
        f"[green]{result.faction2_win_chance * 100:.1f}%[/green]"
    )
    if result.faceit_win_prob_f1 is not None:
        print(
            f"FACEIT winProbability faction1: {result.faceit_win_prob_f1 * 100:.1f}%"
        )
    if show_features:
        print(result.features)
    if show_roster:
        print(result.roster)


if __name__ == "__main__":
    app()

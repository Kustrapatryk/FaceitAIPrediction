from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
    create_engine,
    event,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, relationship, sessionmaker

from src.config import DATABASE_URL, ensure_dirs


class Base(DeclarativeBase):
    pass


class Player(Base):
    __tablename__ = "players"

    player_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    nickname: Mapped[str | None] = mapped_column(String(128))
    country: Mapped[str | None] = mapped_column(String(8))
    elo: Mapped[int | None] = mapped_column(Integer)
    skill_level: Mapped[int | None] = mapped_column(Integer)
    faceit_url: Mapped[str | None] = mapped_column(Text)
    lifetime_matches: Mapped[int | None] = mapped_column(Integer)
    lifetime_wins: Mapped[int | None] = mapped_column(Integer)
    lifetime_winrate: Mapped[float | None] = mapped_column(Float)
    lifetime_kd: Mapped[float | None] = mapped_column(Float)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
    )

    match_appearances: Mapped[list[MatchPlayer]] = relationship(back_populates="player")


class Match(Base):
    __tablename__ = "matches"

    match_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    game: Mapped[str] = mapped_column(String(16), default="cs2")
    region: Mapped[str | None] = mapped_column(String(16))
    map_name: Mapped[str | None] = mapped_column(String(64))
    competition_type: Mapped[str | None] = mapped_column(String(64))
    competition_name: Mapped[str | None] = mapped_column(String(128))
    started_at: Mapped[int | None] = mapped_column(Integer)
    finished_at: Mapped[int | None] = mapped_column(Integer)
    status: Mapped[str | None] = mapped_column(String(32))
    winner_faction: Mapped[str | None] = mapped_column(String(16))
    score_faction1: Mapped[int | None] = mapped_column(Integer)
    score_faction2: Mapped[int | None] = mapped_column(Integer)
    premade_faction1: Mapped[bool | None] = mapped_column(Boolean)
    premade_faction2: Mapped[bool | None] = mapped_column(Boolean)
    premade_size_faction1: Mapped[int | None] = mapped_column(Integer)
    premade_size_faction2: Mapped[int | None] = mapped_column(Integer)
    team1_avg_elo: Mapped[float | None] = mapped_column(Float)
    team2_avg_elo: Mapped[float | None] = mapped_column(Float)
    faceit_win_prob_f1: Mapped[float | None] = mapped_column(Float)
    collected_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
    )

    players: Mapped[list[MatchPlayer]] = relationship(
        back_populates="match",
        cascade="all, delete-orphan",
    )


class MatchPlayer(Base):
    __tablename__ = "match_players"
    __table_args__ = (UniqueConstraint("match_id", "player_id", name="uq_match_player"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    match_id: Mapped[str] = mapped_column(ForeignKey("matches.match_id"), index=True)
    player_id: Mapped[str] = mapped_column(ForeignKey("players.player_id"), index=True)
    faction: Mapped[str] = mapped_column(String(16))
    nickname: Mapped[str | None] = mapped_column(String(128))
    skill_level: Mapped[int | None] = mapped_column(Integer)
    elo: Mapped[int | None] = mapped_column(Integer)
    kills: Mapped[int | None] = mapped_column(Integer)
    deaths: Mapped[int | None] = mapped_column(Integer)
    assists: Mapped[int | None] = mapped_column(Integer)
    kd: Mapped[float | None] = mapped_column(Float)
    adr: Mapped[float | None] = mapped_column(Float)
    headshots_pct: Mapped[float | None] = mapped_column(Float)
    mvps: Mapped[int | None] = mapped_column(Integer)
    result: Mapped[str | None] = mapped_column(String(8))

    match: Mapped[Match] = relationship(back_populates="players")
    player: Mapped[Player] = relationship(back_populates="match_appearances")


class CollectionCursor(Base):
    __tablename__ = "collection_cursors"

    key: Mapped[str] = mapped_column(String(128), primary_key=True)
    value: Mapped[str] = mapped_column(Text)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
    )


def get_engine():
    ensure_dirs()
    connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}
    engine = create_engine(DATABASE_URL, future=True, connect_args=connect_args)

    if DATABASE_URL.startswith("sqlite"):

        @event.listens_for(engine, "connect")
        def set_sqlite_pragma(dbapi_connection, _connection_record) -> None:
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.close()

    return engine


def init_db() -> None:
    engine = get_engine()
    Base.metadata.create_all(engine)


def get_session() -> Session:
    engine = get_engine()
    return sessionmaker(bind=engine, expire_on_commit=False, future=True)()

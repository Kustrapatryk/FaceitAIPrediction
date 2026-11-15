from __future__ import annotations

import time
from typing import Any

import requests

from src.config import FACEIT_API_KEY, FACEIT_BASE_URL, MAX_RETRIES, REQUEST_DELAY_SECONDS


class FaceitAPIError(RuntimeError):
    def __init__(self, status_code: int, message: str) -> None:
        super().__init__(f"FACEIT API {status_code}: {message}")
        self.status_code = status_code


class FaceitClient:
    """Thin wrapper around FACEIT Data API v4."""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str = FACEIT_BASE_URL,
        delay_seconds: float = REQUEST_DELAY_SECONDS,
        max_retries: int = MAX_RETRIES,
    ) -> None:
        self.api_key = api_key or FACEIT_API_KEY
        if not self.api_key:
            raise ValueError(
                "Missing FACEIT_API_KEY. Copy .env.example to .env and set your key."
            )
        self.base_url = base_url.rstrip("/")
        self.delay_seconds = delay_seconds
        self.max_retries = max_retries
        self.session = requests.Session()
        self.session.headers.update(
            {
                "Authorization": f"Bearer {self.api_key}",
                "Accept": "application/json",
            }
        )

    def _request(self, method: str, path: str, params: dict[str, Any] | None = None) -> Any:
        url = f"{self.base_url}{path}"
        last_error: Exception | None = None

        for attempt in range(1, self.max_retries + 1):
            time.sleep(self.delay_seconds)
            try:
                response = self.session.request(method, url, params=params, timeout=30)
            except requests.RequestException as exc:
                last_error = exc
                time.sleep(min(2**attempt, 30))
                continue

            if response.status_code == 429:
                retry_after = float(response.headers.get("Retry-After", min(2**attempt, 60)))
                time.sleep(retry_after)
                continue

            if response.status_code >= 500:
                time.sleep(min(2**attempt, 30))
                continue

            if response.status_code >= 400:
                raise FaceitAPIError(response.status_code, response.text[:500])

            if response.status_code == 204 or not response.content:
                return {}
            return response.json()

        raise RuntimeError(f"FACEIT request failed after retries: {url}") from last_error

    def get_player_by_nickname(self, nickname: str) -> dict[str, Any]:
        return self._request("GET", "/players", params={"nickname": nickname})

    def get_player(self, player_id: str) -> dict[str, Any]:
        return self._request("GET", f"/players/{player_id}")

    def get_player_stats(self, player_id: str, game: str) -> dict[str, Any]:
        return self._request("GET", f"/players/{player_id}/stats/{game}")

    def get_player_recent_game_stats(
        self,
        player_id: str,
        game: str,
        *,
        offset: int = 0,
        limit: int = 20,
    ) -> dict[str, Any]:
        """Per-match stats for the player's recent games (form window source)."""
        return self._request(
            "GET",
            f"/players/{player_id}/games/{game}/stats",
            params={"offset": offset, "limit": limit},
        )

    def get_player_history(
        self,
        player_id: str,
        game: str,
        *,
        offset: int = 0,
        limit: int = 20,
        from_ts: int | None = None,
        to_ts: int | None = None,
    ) -> dict[str, Any]:
        params: dict[str, Any] = {"game": game, "offset": offset, "limit": limit}
        if from_ts is not None:
            params["from"] = from_ts
        if to_ts is not None:
            params["to"] = to_ts
        return self._request("GET", f"/players/{player_id}/history", params=params)

    def get_match(self, match_id: str) -> dict[str, Any]:
        return self._request("GET", f"/matches/{match_id}")

    def get_match_stats(self, match_id: str) -> dict[str, Any]:
        return self._request("GET", f"/matches/{match_id}/stats")

    def get_rankings(
        self,
        game: str,
        region: str,
        *,
        offset: int = 0,
        limit: int = 20,
    ) -> dict[str, Any]:
        return self._request(
            "GET",
            f"/rankings/games/{game}/regions/{region}",
            params={"offset": offset, "limit": limit},
        )

"""Common connector interface plus a network-free mock collector for tests."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import pandas as pd

from src.schemas import ensure_schema


@dataclass
class ConnectorStatus:
    name: str
    available: bool
    message: str  # never contains secret values


@dataclass
class CollectionResult:
    df: pd.DataFrame
    per_source: dict[str, int] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)

    @property
    def fetched(self) -> int:
        return len(self.df)


class BaseCollector(ABC):
    """Every authorized source implements this interface."""

    name: str = "base"

    @abstractmethod
    def status(self) -> ConnectorStatus:
        """Report whether the connector can run, without exposing credentials."""

    @abstractmethod
    def collect(self, **kwargs) -> CollectionResult:
        """Collect comments. Must raise ConnectorError subclasses with safe messages."""


class MockCollector(BaseCollector):
    """Returns provided records without any network access (tests / offline demos)."""

    name = "mock"

    def __init__(self, records: list[dict] | None = None):
        self._records = records or []

    def status(self) -> ConnectorStatus:
        return ConnectorStatus(self.name, True, "Mock collector - no network access.")

    def collect(self, **kwargs) -> CollectionResult:
        df = ensure_schema(pd.DataFrame(self._records), default_source="mock",
                           collection_method="mock")
        return CollectionResult(df=df, per_source={"mock": len(df)})

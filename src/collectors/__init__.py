"""Data collectors. Each source sits behind the BaseCollector interface."""
from src.collectors.base import BaseCollector, CollectionResult, ConnectorStatus, MockCollector

__all__ = ["BaseCollector", "CollectionResult", "ConnectorStatus", "MockCollector"]

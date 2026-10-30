"""TikTok Research API connector - NOT AVAILABLE.

The TikTok Research API is restricted to approved, eligible researchers (see the
eligibility page) and is not generally available to commercial apps. This module
reports status only and does not collect data. If you have approved credentials
and eligible use, a collector can be added behind BaseCollector; otherwise use
CSV upload with comments you are permitted to export.
Docs: https://developers.tiktok.com/products/research-api/
      https://developers.tiktok.com/docs/en/research-api-faq
      https://developers.tiktok.com/docs/en/research-api-specs-query-video-comments
"""
from __future__ import annotations

from src.collectors.base import BaseCollector, CollectionResult, ConnectorStatus
from src.schemas import ConnectorNotConfigured

REQUIREMENTS = (
    "TikTok comments require approved Research API access (eligibility-restricted, not generally "
    "available to commercial apps). Use CSV upload with data you are permitted to export.")


class TikTokResearchCollector(BaseCollector):
    name = "tiktok_research"

    def status(self) -> ConnectorStatus:
        return ConnectorStatus(self.name, False, "Not available in this version. " + REQUIREMENTS)

    def collect(self, **kwargs) -> CollectionResult:
        raise ConnectorNotConfigured(
            "The TikTok connector is not implemented in this version. " + REQUIREMENTS)

"""Meta (Facebook Page / Instagram professional account) connector - NOT AVAILABLE.

This module reports status and requirements honestly. It does not collect data.
Meta's Graph API only returns comments for content the authenticated user or app
is authorized to access (for example a Page you manage, or an Instagram
professional account you own), typically needing a Page/User access token with
the relevant permissions, a linked professional account and, for production use
by other people's accounts, App Review / advanced access. It cannot be used to
collect arbitrary competitor comments.

Until such access exists, export your own comments (Meta Business Suite) and use
the CSV import, which this app fully supports.
Docs: https://developers.facebook.com/docs/graph-api/reference/object/comments/
      https://developers.facebook.com/docs/instagram-platform/
"""
from __future__ import annotations

from src.collectors.base import BaseCollector, CollectionResult, ConnectorStatus
from src.schemas import ConnectorNotConfigured

REQUIREMENTS = (
    "Facebook / Instagram comments need official Meta API access to an account you own or manage "
    "(access token + permissions, possibly App Review). Arbitrary public brand pages are not "
    "collectable. Use a comment export from Meta Business Suite with CSV upload instead.")


class MetaCollector(BaseCollector):
    name = "meta"

    def status(self) -> ConnectorStatus:
        return ConnectorStatus(self.name, False, "Not available in this version. " + REQUIREMENTS)

    def collect(self, **kwargs) -> CollectionResult:
        raise ConnectorNotConfigured(
            "The Meta connector is not implemented in this version. " + REQUIREMENTS)

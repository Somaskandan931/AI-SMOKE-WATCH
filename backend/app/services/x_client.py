"""
Real HTTP client for publishing to X (Twitter).

Two calls are made, both signed with OAuth 1.0a user-context credentials:

1. POST https://upload.twitter.com/1.1/media/upload.json
   X API v2 still has no media-upload endpoint of its own -- this v1.1
   endpoint is the only way to attach an image to a post, for any account
   tier. This is not a workaround; it's documented, current behaviour.

2. POST https://api.x.com/2/tweets
   The v2 tweet-creation endpoint accepts OAuth 1.0a user-context auth
   (it doesn't require OAuth 2.0). Using OAuth 1.0a for both calls means
   one fixed credential set (API key/secret + access token/secret)
   generated once in the X developer portal -- no login/consent screen
   flow needed. That's the right tradeoff for a single reporting/demo
   account; a version where each citizen posts from their own X account
   would need a 3-legged OAuth 2.0 PKCE flow instead (out of scope here).

Every function in this module either returns real data from a real X API
response or raises XApiError -- nothing here silently pretends to
succeed. Mock-mode fallback (for demoing before you have credentials)
lives one layer up, in report_service.publish_report_to_x.
"""

from __future__ import annotations

import requests
from requests_oauthlib import OAuth1

from app import config

MEDIA_UPLOAD_URL = "https://upload.twitter.com/1.1/media/upload.json"
TWEET_CREATE_URL = "https://api.x.com/2/tweets"

# X's simple (non-chunked) media upload tops out at 5MB for images.
# Smartphone photos can exceed this; the caller should compress first.
MAX_SIMPLE_UPLOAD_BYTES = 5 * 1024 * 1024

REQUEST_TIMEOUT_SECONDS = 30


class XApiError(RuntimeError):
    """Raised for any failed or malformed X API interaction (network
    failure, non-2xx response, or an unexpected response shape)."""


def _auth() -> OAuth1:
    if not config.x_credentials_configured():
        raise XApiError(
            "X API credentials are not configured. Set X_API_KEY, X_API_SECRET, "
            "X_ACCESS_TOKEN and X_ACCESS_SECRET in .env."
        )
    return OAuth1(
        client_key=config.X_API_KEY,
        client_secret=config.X_API_SECRET,
        resource_owner_key=config.X_ACCESS_TOKEN,
        resource_owner_secret=config.X_ACCESS_SECRET,
    )


def upload_media(image_bytes: bytes, filename: str = "evidence.jpg") -> str:
    """
    Uploads the evidence image and returns the media_id_string to attach
    to a tweet. Raises XApiError on any failure.
    """
    if len(image_bytes) > MAX_SIMPLE_UPLOAD_BYTES:
        raise XApiError(
            f"Evidence image is {len(image_bytes) / 1_048_576:.1f}MB, over X's "
            f"{MAX_SIMPLE_UPLOAD_BYTES // 1_048_576}MB simple-upload limit. "
            "Compress the image before publishing."
        )

    try:
        response = requests.post(
            MEDIA_UPLOAD_URL,
            auth=_auth(),
            files={"media": (filename, image_bytes)},
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
    except requests.RequestException as exc:
        raise XApiError(f"Network error contacting X media upload: {exc}") from exc

    if response.status_code != 200:
        raise XApiError(f"X media upload failed ({response.status_code}): {response.text}")

    try:
        media_id = response.json()["media_id_string"]
    except (ValueError, KeyError) as exc:
        raise XApiError(f"X media upload response missing media_id_string: {response.text}") from exc

    return media_id


def post_tweet(text: str, media_id: str | None = None) -> dict:
    """
    Creates a tweet via X API v2. Returns the parsed JSON response
    (contains {"data": {"id": ..., "text": ...}} on success). Raises
    XApiError on any failure.
    """
    body: dict = {"text": text}
    if media_id:
        body["media"] = {"media_ids": [media_id]}

    try:
        response = requests.post(
            TWEET_CREATE_URL,
            auth=_auth(),
            json=body,
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
    except requests.RequestException as exc:
        raise XApiError(f"Network error contacting X tweet-create endpoint: {exc}") from exc

    if response.status_code not in (200, 201):
        raise XApiError(f"X tweet creation failed ({response.status_code}): {response.text}")

    try:
        return response.json()
    except ValueError as exc:
        raise XApiError(f"X tweet creation returned a non-JSON response: {response.text}") from exc


def tweet_url_from_response(response: dict) -> str | None:
    """Builds a permalink from a post_tweet() response. Uses the
    'i/web/status/<id>' form, which resolves correctly regardless of the
    posting account's handle -- so this doesn't need to know or store it."""
    tweet_id = response.get("data", {}).get("id")
    if not tweet_id:
        return None
    return f"https://x.com/i/web/status/{tweet_id}"

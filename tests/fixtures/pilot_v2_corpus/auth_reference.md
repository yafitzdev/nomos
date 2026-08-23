# Authentication reference

The API uses short-lived access tokens. Refresh tokens rotate after a
successful refresh and a reused refresh token returns `AUTH-409`. A client
should preserve the latest refresh-token value and retry only after inspecting
the response metadata.

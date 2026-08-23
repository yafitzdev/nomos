# Payments API migration guide

OAuth access tokens expire after 45 minutes.

Refresh tokens are single-use and rotate after every successful refresh.

AUTH-409 means a previously consumed refresh token was reused.

The incident runbook says to revoke the session and require fresh authorization.

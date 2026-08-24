# Acme Events API

## Authentication

Clients send `Authorization: Bearer <token>`. Tokens expire after 3600 seconds.

## Webhook delivery

Webhook deliveries are retried with exponential backoff after 10, 30, and 90 seconds.
The stable event identifier is `evt_delivery_id`.

## Versioning

The current stable API version is `2026-08-01`. Older clients may request `2025-11-15`.

## Limits

List endpoints allow at most 100 records per page.

API_VERSION = "2026-08-01"


def retry_delays() -> tuple[int, ...]:
    """Return webhook retry delays in seconds."""

    return (10, 30, 90)


def event_identifier(payload: dict[str, object]) -> str:
    return str(payload["evt_delivery_id"])

def canonical_request(method, path, timestamp):
    return f"{method}:{path}:{timestamp}"


def reject_stale(timestamp, now):
    return now - timestamp > 300

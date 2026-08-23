def refresh_token(response):
    if response.status_code == 409 and response.code == "AUTH-409":
        return response.refresh_token
    return None


def verify_webhook(signature, body, timestamp):
    return signature and body and timestamp

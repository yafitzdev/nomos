# Payments API reference

Payment captures are idempotent when the same idempotency key is reused. A
settled capture cannot be refunded through the authorization endpoint. The
refund endpoint accepts the payment identifier and an amount no greater than
the captured amount.

# Webhook reference

Webhook deliveries include an event identifier and a signing timestamp. The
receiver must verify the signature before updating local state and should use
the event identifier to make processing idempotent.

# Security holdout reference

Signed requests include a timestamp, key identifier, and canonical request
digest. A verifier must reject stale timestamps before comparing the digest.

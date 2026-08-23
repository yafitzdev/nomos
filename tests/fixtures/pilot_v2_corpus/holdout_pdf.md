# Signed request appendix

The canonical request digest includes the HTTP method, normalized path, query
parameters, body digest, and timestamp. Verification must reject stale
requests before evaluating the signature.

"""Small local-only OpenAI-compatible proxy that rewrites the served model ID."""

from __future__ import annotations

import argparse
import json
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Sequence


class AliasProxyHandler(BaseHTTPRequestHandler):
    target_base_url = "http://127.0.0.1:19003/v1"
    target_model = "qwen3.8-27b-nvfp4"

    def log_message(self, _format: str, *_args: object) -> None:
        return

    def _forward(self, body: bytes | None = None) -> None:
        target = f"{self.target_base_url.rstrip('/')}{self.path}"
        headers = {"Content-Type": self.headers.get("Content-Type", "application/json")}
        if body is not None:
            try:
                payload = json.loads(body)
                if isinstance(payload, dict):
                    payload["model"] = self.target_model
                    body = json.dumps(payload).encode("utf-8")
            except json.JSONDecodeError:
                pass
            headers["Content-Length"] = str(len(body))
        request = urllib.request.Request(target, data=body, headers=headers, method=self.command)
        try:
            with urllib.request.urlopen(request, timeout=300) as response:
                response_body = response.read()
                self.send_response(response.status)
                self.send_header("Content-Type", response.headers.get("Content-Type", "application/json"))
                self.send_header("Content-Length", str(len(response_body)))
                self.end_headers()
                self.wfile.write(response_body)
        except urllib.error.HTTPError as exc:
            response_body = exc.read()
            self.send_response(exc.code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(response_body)))
            self.end_headers()
            self.wfile.write(response_body)
        except (urllib.error.URLError, TimeoutError) as exc:
            response_body = json.dumps({"error": str(exc)}).encode("utf-8")
            self.send_response(502)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(response_body)))
            self.end_headers()
            self.wfile.write(response_body)

    def do_GET(self) -> None:  # noqa: N802
        self._forward()

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length", "0"))
        self._forward(self.rfile.read(length))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--listen-host", default="127.0.0.1")
    parser.add_argument("--listen-port", type=int, default=19004)
    parser.add_argument("--target-url", default="http://127.0.0.1:19003/v1")
    parser.add_argument("--target-model", default="qwen3.8-27b-nvfp4")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    AliasProxyHandler.target_base_url = args.target_url
    AliasProxyHandler.target_model = args.target_model
    server = ThreadingHTTPServer((args.listen_host, args.listen_port), AliasProxyHandler)
    print(f"proxy listening on http://{args.listen_host}:{args.listen_port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        return 0
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

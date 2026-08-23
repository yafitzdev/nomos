"""Bridge an OpenAI-compatible local agent endpoint through Nomos.

The proxy is intentionally outside Fitz-Sage V2. It raises the output budget
for tool calls, converts parseable Qwen XML/JSON tool text into the OpenAI
``tool_calls`` shape when a backend omits native calls, and optionally uses a
Nomos artifact to choose one tool from the external agent's visible legal set.
"""

from __future__ import annotations

import argparse
import copy
import json
import re
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Mapping, Sequence

from fitz_tool.adapters.fitz_sage_v2 import build_runner_request_from_openai
from fitz_tool.router_v2 import load_router_v2, rank_tools_v2
from fitz_tool.tool_registry import ToolRegistry


XML_CALL_RE = re.compile(
    r"<tool_call>\s*<function=([A-Za-z_][A-Za-z0-9_]*)>(.*?)</function>\s*</tool_call>",
    flags=re.IGNORECASE | re.DOTALL,
)
PARAMETER_RE = re.compile(
    r"<parameter=([A-Za-z_][A-Za-z0-9_]*)>\s*(.*?)\s*</parameter>",
    flags=re.IGNORECASE | re.DOTALL,
)


@dataclass
class ProxyConfig:
    target_url: str
    min_max_tokens: int
    retry_max_tokens: int
    max_steps: int
    source_modality: str
    model: Any | None = None
    metadata: Mapping[str, Any] | None = None
    registry: ToolRegistry | None = None
    trace_path: Path | None = None


def _target_url(target_base_url: str, request_path: str) -> str:
    base = target_base_url.rstrip("/")
    if request_path.startswith("/v1/"):
        return f"{base}/{request_path.split('/v1/', 1)[1]}"
    if request_path == "/v1":
        return base
    return f"{base}{request_path}"


def _tool_names(body: Mapping[str, Any]) -> list[str]:
    names: list[str] = []
    raw_tools = body.get("tools")
    if not isinstance(raw_tools, list):
        return names
    for item in raw_tools:
        if not isinstance(item, Mapping):
            continue
        function = item.get("function")
        name = function.get("name") if isinstance(function, Mapping) else None
        if isinstance(name, str) and name and name not in names:
            names.append(name)
    return names


def _json_action(text: str, allowed: set[str]) -> tuple[str, dict[str, Any]] | None:
    candidate = text.strip()
    candidate = re.sub(r"^```(?:json)?\s*|\s*```$", "", candidate, flags=re.IGNORECASE | re.DOTALL).strip()
    decoder = json.JSONDecoder()
    values: list[Any] = []
    try:
        values.append(json.loads(candidate))
    except json.JSONDecodeError:
        pass
    for index, char in enumerate(candidate):
        if char != "{":
            continue
        try:
            value, _end = decoder.raw_decode(candidate[index:])
        except json.JSONDecodeError:
            continue
        values.append(value)
        break
    for value in values:
        if not isinstance(value, Mapping):
            continue
        action = value.get("action")
        name = value.get("name")
        if action == "tool" and isinstance(name, str):
            arguments = value.get("arguments")
        elif isinstance(action, str) and action in allowed:
            name = action
            arguments = value.get("arguments")
            if not isinstance(arguments, Mapping):
                arguments = {
                    key: item for key, item in value.items() if key not in {"action", "name"}
                }
        else:
            continue
        if name not in allowed:
            continue
        return name, dict(arguments) if isinstance(arguments, Mapping) else {}
    return None


def _xml_action(text: str, allowed: set[str]) -> tuple[str, dict[str, Any]] | None:
    match = XML_CALL_RE.search(text)
    if match is None or match.group(1) not in allowed:
        return None
    arguments: dict[str, Any] = {}
    for parameter in PARAMETER_RE.finditer(match.group(2)):
        raw_value = parameter.group(2).strip()
        try:
            arguments[parameter.group(1)] = json.loads(raw_value)
        except json.JSONDecodeError:
            arguments[parameter.group(1)] = raw_value
    return match.group(1), arguments


def parse_tool_text(text: str, allowed_names: Sequence[str]) -> tuple[str, dict[str, Any]] | None:
    """Parse only complete, allowlisted model tool calls."""

    allowed = set(allowed_names)
    return _xml_action(text, allowed) or _json_action(text, allowed)


def repair_completion_payload(
    payload: Mapping[str, Any],
    *,
    allowed_names: Sequence[str],
    call_id: str = "nomos-repaired-call",
    preferred_name: str | None = None,
    request_body: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], bool]:
    """Normalize a text-encoded call into the OpenAI response shape."""

    output = copy.deepcopy(dict(payload))
    choices = output.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], Mapping):
        return output, False
    choice = dict(choices[0])
    message = dict(choice.get("message") or {})
    expected_name = preferred_name or (allowed_names[0] if len(allowed_names) == 1 else None)
    existing_call = _first_tool_call(message.get("tool_calls"))
    if existing_call is not None:
        existing_name, existing_arguments = existing_call
        if existing_name in allowed_names and (expected_name is None or existing_name == expected_name):
            if _valid_tool_calls(message.get("tool_calls"), allowed_names):
                return output, False
        if expected_name is not None:
            return _set_tool_call(
                output,
                choice,
                expected_name,
                _coerce_arguments(request_body, expected_name, existing_arguments),
                call_id,
            ), True
    content = message.get("content")
    parsed = parse_tool_text(content, allowed_names) if isinstance(content, str) else None
    if parsed is not None:
        name, arguments = parsed
        if expected_name is None:
            expected_name = name
        return _set_tool_call(
            output,
            choice,
            expected_name,
            _coerce_arguments(request_body, expected_name, arguments),
            call_id,
        ), True
    if expected_name is not None:
        return _set_tool_call(
            output,
            choice,
            expected_name,
            _coerce_arguments(request_body, expected_name, {}),
            call_id,
        ), True
    return output, False


def _first_tool_call(value: Any) -> tuple[str, dict[str, Any]] | None:
    if not isinstance(value, list) or not value or not isinstance(value[0], Mapping):
        return None
    function = value[0].get("function")
    if not isinstance(function, Mapping) or not isinstance(function.get("name"), str):
        return None
    arguments = function.get("arguments", {})
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            arguments = {}
    return str(function["name"]), dict(arguments) if isinstance(arguments, Mapping) else {}


def _set_tool_call(
    output: dict[str, Any],
    choice: Mapping[str, Any],
    name: str,
    arguments: Mapping[str, Any],
    call_id: str,
) -> dict[str, Any]:
    updated_choice = dict(choice)
    updated_choice["message"] = {
        **dict(choice.get("message") or {}),
        "content": None,
        "tool_calls": [
            {
                "id": call_id,
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": json.dumps(dict(arguments), ensure_ascii=False),
                },
            }
        ],
    }
    updated_choice["finish_reason"] = "tool_calls"
    choices = output.get("choices") or []
    output["choices"] = [updated_choice, *choices[1:]]
    return output


def _question_from_body(body: Mapping[str, Any] | None) -> str:
    if not isinstance(body, Mapping) or not isinstance(body.get("messages"), list):
        return ""
    for message in reversed(body["messages"]):
        if isinstance(message, Mapping) and message.get("role") == "user":
            content = message.get("content")
            if isinstance(content, str) and content.strip():
                return content.strip()
    return ""


def _coerce_arguments(
    body: Mapping[str, Any] | None,
    name: str,
    raw_arguments: Mapping[str, Any],
) -> dict[str, Any]:
    """Keep model arguments compatible with the selected tool schema."""

    properties: Mapping[str, Any] = {}
    required: list[str] = []
    matched_schema = False
    if isinstance(body, Mapping) and isinstance(body.get("tools"), list):
        for item in body["tools"]:
            if not isinstance(item, Mapping) or not isinstance(item.get("function"), Mapping):
                continue
            function = item["function"]
            if function.get("name") != name:
                continue
            matched_schema = True
            parameters = function.get("parameters")
            if isinstance(parameters, Mapping):
                properties = parameters.get("properties") if isinstance(parameters.get("properties"), Mapping) else {}
                required = [str(value) for value in parameters.get("required", []) if isinstance(value, str)]
            break
    if not matched_schema:
        return dict(raw_arguments)
    arguments = {key: value for key, value in raw_arguments.items() if key in properties}
    question = _question_from_body(body)
    for key in required:
        if key in arguments:
            continue
        if key in {"objective", "query", "pattern"}:
            arguments[key] = question
        elif key == "filters":
            arguments[key] = {}
        elif key in {"evidence_ids", "source_scope", "pages"}:
            arguments[key] = []
        else:
            arguments[key] = ""
    if "scope" in properties and "scope" not in arguments:
        arguments["scope"] = str(raw_arguments.get("query") or "")
    return arguments


def _valid_tool_calls(value: Any, allowed_names: Sequence[str]) -> bool:
    if not isinstance(value, list) or not value:
        return False
    allowed = set(allowed_names)
    for item in value:
        if not isinstance(item, Mapping):
            return False
        function = item.get("function")
        if not isinstance(function, Mapping) or function.get("name") not in allowed:
            return False
        arguments = function.get("arguments", {})
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                return False
        if not isinstance(arguments, Mapping):
            return False
    return True


def _has_tool_call(payload: Mapping[str, Any], allowed_names: Sequence[str]) -> bool:
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], Mapping):
        return False
    message = choices[0].get("message")
    return isinstance(message, Mapping) and _valid_tool_calls(message.get("tool_calls"), allowed_names)


def _looks_like_tool_call(payload: Mapping[str, Any]) -> bool:
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], Mapping):
        return False
    message = choices[0].get("message")
    content = message.get("content") if isinstance(message, Mapping) else None
    return isinstance(content, str) and ("<tool_call>" in content or '"action"' in content)


def _selected_tool(body: Mapping[str, Any], config: ProxyConfig) -> str | None:
    if config.model is None or config.metadata is None or config.registry is None:
        return None
    names = _tool_names(body)
    if len(names) < 2:
        return None
    request = build_runner_request_from_openai(
        body,
        registry=config.registry,
        max_steps=config.max_steps,
        source_modality=config.source_modality,
    )
    rankable = [
        tool_id
        for tool_id in names
        if _candidate_compatible_with_source(request, tool_id, config.source_modality)
    ]
    ranking_request = dict(request)
    if rankable:
        ranking_request["legal_candidate_ids"] = rankable
    ranked = rank_tools_v2(config.model, config.metadata, ranking_request, top_k=len(rankable or names))
    return str(ranked[0]["tool_id"]) if ranked else None


def _candidate_compatible_with_source(
    request: Mapping[str, Any],
    tool_id: str,
    source_modality: str,
) -> bool:
    if source_modality == "mixed":
        return True
    registry = ToolRegistry.from_dict(request["tool_registry"])
    tool = registry.require(tool_id)
    inputs = set(tool.input_modalities)
    capabilities = set(tool.capabilities)
    if source_modality != "pdf" and "discover_pdf_sources" in capabilities:
        return False
    if source_modality not in {"csv", "excel", "sqlite", "mixed"} and "discover_structured_sources" in capabilities:
        return False
    # Discovery remains legal after evidence exists when the external runner
    # exposes it. A named source constraint (for example, "according to the
    # payments guide") can still be unresolved after retrieval, so hiding
    # discovery here would force the router to repeat the same search. The
    # external runner's visible candidate set remains the source of truth.
    if source_modality in inputs or "metadata" in inputs or "agent_state" in inputs:
        return True
    if request.get("observed_evidence") and inputs.intersection(
        {"evidence_candidates", "evidence", "governance_state"}
    ):
        return True
    return False


def _append_tool_constraint(body: dict[str, Any], name: str) -> None:
    instruction = (
        "\n\nRUNTIME TOOL CONSTRAINT: On this turn you must call exactly the function "
        f"{name}. Do not call any other function, even if another function name "
        "appears in earlier instructions or results. Return one tool call only "
        "with arguments matching that function's supplied schema."
    )
    messages = body.get("messages")
    if not isinstance(messages, list):
        body["messages"] = [{"role": "system", "content": instruction.strip()}]
        return
    for index, message in enumerate(messages):
        if not isinstance(message, Mapping) or message.get("role") != "system":
            continue
        updated = dict(message)
        content = updated.get("content")
        updated["content"] = (content + instruction) if isinstance(content, str) else instruction.strip()
        messages[index] = updated
        return
    messages.insert(0, {"role": "system", "content": instruction.strip()})


def prepare_request(body: Mapping[str, Any], config: ProxyConfig) -> tuple[dict[str, Any], str | None]:
    output = copy.deepcopy(dict(body))
    selected = _selected_tool(output, config)
    names = _tool_names(output)
    enforced = selected or (names[0] if len(names) == 1 else None)
    if enforced:
        output["tool_choice"] = {"type": "function", "function": {"name": enforced}}
        _append_tool_constraint(output, enforced)
    current_tokens = output.get("max_tokens")
    try:
        current = int(current_tokens)
    except (TypeError, ValueError):
        current = 0
    output["max_tokens"] = max(current, config.min_max_tokens)
    return output, selected


def _forward(
    request_path: str,
    body: bytes | None,
    *,
    target_url: str,
) -> tuple[int, dict[str, str], bytes]:
    headers = {"Content-Type": "application/json"}
    request = urllib.request.Request(
        _target_url(target_url, request_path),
        data=body,
        headers=headers,
        method="POST" if body is not None else "GET",
    )
    try:
        with urllib.request.urlopen(request, timeout=300) as response:
            return response.status, {
                "Content-Type": response.headers.get("Content-Type", "application/json")
            }, response.read()
    except urllib.error.HTTPError as exc:
        return exc.code, {"Content-Type": "application/json"}, exc.read()


class NomosProxyHandler(BaseHTTPRequestHandler):
    config: ProxyConfig
    route_count = 0
    repair_count = 0

    def log_message(self, _format: str, *_args: object) -> None:
        return

    def _send(self, status: int, headers: Mapping[str, str], content: bytes, *, selected: str | None = None) -> None:
        self.send_response(status)
        for key, value in headers.items():
            self.send_header(key, value)
        if selected:
            self.send_header("X-Nomos-Selected-Tool", selected)
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def _trace_response(
        self,
        *,
        body: Mapping[str, Any],
        payload: Mapping[str, Any] | None,
        selected: str | None,
        retried: bool,
        repaired: bool,
    ) -> None:
        path = self.config.trace_path
        if path is None:
            return
        choices = payload.get("choices") if isinstance(payload, Mapping) else None
        choice = choices[0] if isinstance(choices, list) and choices and isinstance(choices[0], Mapping) else {}
        message = choice.get("message") if isinstance(choice, Mapping) else {}
        record = {
            "request_id": body.get("user"),
            "candidate_names": _tool_names(body),
            "tool_choice": body.get("tool_choice"),
            "max_tokens": body.get("max_tokens"),
            "selected_tool": selected,
            "retried": retried,
            "repaired": repaired,
            "finish_reason": choice.get("finish_reason") if isinstance(choice, Mapping) else None,
            "message_content": message.get("content") if isinstance(message, Mapping) else None,
            "tool_calls": message.get("tool_calls") if isinstance(message, Mapping) else None,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")

    def do_GET(self) -> None:  # noqa: N802
        status, headers, content = _forward(self.path, None, target_url=self.config.target_url)
        self._send(status, headers, content)

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length", "0"))
        raw_body = self.rfile.read(length)
        if not self.path.endswith("/chat/completions"):
            status, headers, content = _forward(self.path, raw_body, target_url=self.config.target_url)
            self._send(status, headers, content)
            return
        try:
            body = json.loads(raw_body)
            if not isinstance(body, Mapping):
                raise ValueError("request body must be an object")
            prepared, selected = prepare_request(body, self.config)
            if selected:
                type(self).route_count += 1
            request_body = json.dumps(prepared, ensure_ascii=False).encode("utf-8")
        except (json.JSONDecodeError, TypeError, ValueError, KeyError) as exc:
            self._send(
                400,
                {"Content-Type": "application/json"},
                json.dumps({"error": str(exc)}).encode("utf-8"),
            )
            return

        status, headers, content = _forward(
            self.path,
            request_body,
            target_url=self.config.target_url,
        )
        selected_for_header = selected
        retried = False
        repaired = False
        try:
            payload = json.loads(content)
            allowed_names = _tool_names(prepared)
            if isinstance(payload, Mapping) and status < 400 and not _has_tool_call(payload, allowed_names):
                max_tokens = int(prepared.get("max_tokens") or 0)
                if max_tokens < self.config.retry_max_tokens:
                    retry = dict(prepared)
                    retry["max_tokens"] = self.config.retry_max_tokens
                    status, headers, content = _forward(
                        self.path,
                        json.dumps(retry, ensure_ascii=False).encode("utf-8"),
                        target_url=self.config.target_url,
                    )
                    payload = json.loads(content)
                    retried = True
            if isinstance(payload, Mapping) and status < 400:
                if not _has_tool_call(payload, allowed_names):
                    payload, repaired = repair_completion_payload(
                        payload,
                        allowed_names=allowed_names,
                        call_id="nomos-repaired-call",
                        preferred_name=selected_for_header,
                        request_body=prepared,
                    )
                    if repaired:
                        type(self).repair_count += 1
                        content = json.dumps(payload, ensure_ascii=False).encode("utf-8")
                self._trace_response(
                    body=prepared,
                    payload=payload,
                    selected=selected_for_header,
                    retried=retried,
                    repaired=repaired,
                )
        except (ValueError, TypeError, json.JSONDecodeError):
            pass
        self._send(status, headers, content, selected=selected_for_header)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--listen-host", default="127.0.0.1")
    parser.add_argument("--listen-port", type=int, default=19004)
    parser.add_argument("--target-url", default="http://127.0.0.1:19003/v1")
    parser.add_argument("--artifact", type=Path)
    parser.add_argument("--min-max-tokens", type=int, default=512)
    parser.add_argument("--retry-max-tokens", type=int, default=1024)
    parser.add_argument("--max-steps", type=int, default=8)
    parser.add_argument(
        "--source-modality",
        choices=("text", "pdf", "csv", "excel", "sqlite", "code", "mixed"),
        default="text",
    )
    parser.add_argument("--trace-output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.min_max_tokens < 1 or args.retry_max_tokens < args.min_max_tokens:
        raise SystemExit("retry-max-tokens must be at least min-max-tokens, both positive")
    model = metadata = registry = None
    if args.artifact:
        model, metadata = load_router_v2(str(args.artifact))
        from fitz_tool.adapters.fitz_sage_v2 import load_fitz_sage_v2_registry

        registry = load_fitz_sage_v2_registry()
    config = ProxyConfig(
        target_url=args.target_url,
        min_max_tokens=args.min_max_tokens,
        retry_max_tokens=args.retry_max_tokens,
        max_steps=args.max_steps,
        source_modality=args.source_modality,
        model=model,
        metadata=metadata,
        registry=registry,
        trace_path=args.trace_output,
    )
    NomosProxyHandler.config = config
    server = ThreadingHTTPServer((args.listen_host, args.listen_port), NomosProxyHandler)
    print(
        f"nomos proxy listening on http://{args.listen_host}:{args.listen_port}/v1 "
        f"router={'on' if model is not None else 'off'} min_max_tokens={args.min_max_tokens}",
        file=sys.stderr,
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        return 0
    finally:
        server.server_close()
        print(
            f"nomos proxy stopped routes={NomosProxyHandler.route_count} "
            f"repairs={NomosProxyHandler.repair_count}",
            file=sys.stderr,
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

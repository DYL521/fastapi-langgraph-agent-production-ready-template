#!/usr/bin/env python3
"""End-to-end API smoke test for the LangGraph FastAPI agent.

Exercises the relational-DB-backed and checkpointer-backed endpoints against a
running server, so it works the same whether the backend is PostgreSQL or
MySQL (`DB_DIALECT`). Useful for validating multi-database compatibility.

Flow: health -> register -> login -> create session -> list sessions ->
update name -> get messages -> clear history (checkpointer delete) ->
delete session. The chat endpoint is exercised only when RUN_CHAT=1 (it needs a
working LLM key).

Usage:
    python scripts/smoke_test_api.py                 # against http://localhost:8000
    BASE_URL=http://localhost:8000 RUN_CHAT=1 python scripts/smoke_test_api.py

Exits non-zero if any step fails. No third-party dependencies (stdlib only).
"""

from __future__ import annotations

import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid

BASE_URL = os.environ.get("BASE_URL", "http://localhost:8000").rstrip("/")
API = f"{BASE_URL}/api/v1"
RUN_CHAT = os.environ.get("RUN_CHAT", "0") == "1"

PASSED = 0
FAILED = 0


def _request(method: str, url: str, *, json_body=None, form_body=None, token=None, timeout=30):
    """Make an HTTP request; return (status_code, parsed_json_or_text)."""
    headers = {}
    data = None
    if json_body is not None:
        data = json.dumps(json_body).encode()
        headers["Content-Type"] = "application/json"
    elif form_body is not None:
        data = urllib.parse.urlencode(form_body).encode()
        headers["Content-Type"] = "application/x-www-form-urlencoded"
    if token:
        headers["Authorization"] = f"Bearer {token}"

    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode()
            status = resp.status
    except urllib.error.HTTPError as e:
        body = e.read().decode()
        status = e.code
    except urllib.error.URLError as e:
        return 0, str(e)

    try:
        return status, json.loads(body)
    except json.JSONDecodeError:
        return status, body


def check(name: str, ok: bool, detail: str = "") -> bool:
    """Record and print a step result."""
    global PASSED, FAILED
    if ok:
        PASSED += 1
        print(f"  [PASS] {name}")
    else:
        FAILED += 1
        print(f"  [FAIL] {name} -- {detail}")
    return ok


def wait_for_health(retries: int = 30, delay: float = 2.0) -> bool:
    """Poll the health endpoint until the server is reachable."""
    for _ in range(retries):
        status, body = _request("GET", f"{BASE_URL}/health")
        if status in (200, 503):
            return True
        time.sleep(delay)
    return False


def main() -> int:
    """Run the smoke test sequence."""
    print(f"== API smoke test against {BASE_URL} ==")

    if not wait_for_health():
        check("server reachable", False, "health endpoint never responded")
        return 1
    status, body = _request("GET", f"{BASE_URL}/health")
    check("GET /health", status in (200, 503), f"status={status} body={body}")
    if isinstance(body, dict):
        print(f"         backend health: {body.get('components', body.get('status'))}")

    # --- register ---------------------------------------------------------
    email = f"smoke_{uuid.uuid4().hex[:10]}@example.com"
    password = "SmokeTest123!"
    status, body = _request(
        "POST", f"{API}/auth/register", json_body={"email": email, "password": password, "username": "smoke"}
    )
    ok = check("POST /auth/register", status == 200 and isinstance(body, dict) and "token" in body, f"status={status} body={body}")
    if not ok:
        return _finish()
    user_token = body["token"]["access_token"]

    # --- login ------------------------------------------------------------
    status, body = _request(
        "POST", f"{API}/auth/login", form_body={"email": email, "password": password, "grant_type": "password"}
    )
    check("POST /auth/login", status == 200 and isinstance(body, dict) and "access_token" in body, f"status={status} body={body}")
    if status == 200 and isinstance(body, dict):
        user_token = body["access_token"]

    # --- create session ---------------------------------------------------
    status, body = _request("POST", f"{API}/auth/session", token=user_token)
    ok = check("POST /auth/session", status == 200 and isinstance(body, dict) and "session_id" in body, f"status={status} body={body}")
    if not ok:
        return _finish()
    session_id = body["session_id"]
    session_token = body["token"]["access_token"]

    # --- list sessions ----------------------------------------------------
    status, body = _request("GET", f"{API}/auth/sessions", token=user_token)
    found = isinstance(body, list) and any(s.get("session_id") == session_id for s in body)
    check("GET /auth/sessions", status == 200 and found, f"status={status} found={found}")

    # --- update session name ---------------------------------------------
    status, body = _request(
        "PATCH", f"{API}/auth/session/{session_id}/name", form_body={"name": "renamed-by-smoke"}, token=session_token
    )
    renamed = isinstance(body, dict) and body.get("name") == "renamed-by-smoke"
    check("PATCH /auth/session/{id}/name", status == 200 and renamed, f"status={status} body={body}")

    # --- get messages (reads from checkpointer) --------------------------
    status, body = _request("GET", f"{API}/chatbot/messages", token=session_token)
    check("GET /chatbot/messages", status == 200, f"status={status} body={body}")

    # --- optional chat ----------------------------------------------------
    if RUN_CHAT:
        status, body = _request(
            "POST",
            f"{API}/chatbot/chat",
            json_body={"messages": [{"role": "user", "content": "Say hello in one word."}]},
            token=session_token,
            timeout=120,
        )
        check("POST /chatbot/chat", status == 200 and isinstance(body, dict) and body.get("messages"), f"status={status} body={body}")
    else:
        print("  [SKIP] POST /chatbot/chat (set RUN_CHAT=1 with a valid LLM key to enable)")

    # --- clear history (checkpointer DELETE — key MySQL/PG path) ----------
    status, body = _request("DELETE", f"{API}/chatbot/messages", token=session_token)
    check("DELETE /chatbot/messages (clear checkpoints)", status == 200, f"status={status} body={body}")

    # --- delete session (cleanup) ----------------------------------------
    status, body = _request("DELETE", f"{API}/auth/session/{session_id}", token=session_token)
    check("DELETE /auth/session/{id}", status in (200, 204), f"status={status} body={body}")

    return _finish()


def _finish() -> int:
    """Print the summary and return an exit code."""
    print(f"\n== Result: {PASSED} passed, {FAILED} failed ==")
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())

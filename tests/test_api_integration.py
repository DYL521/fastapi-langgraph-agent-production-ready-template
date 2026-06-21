"""End-to-end API integration tests over an in-memory SQLite database."""

PASSWORD = "TestPass123!"
API = "/api/v1"


async def _register(client, email, username="tester"):
    return await client.post(f"{API}/auth/register", json={"email": email, "password": PASSWORD, "username": username})


async def _user_headers(client, email):
    token = (await _register(client, email)).json()["token"]["access_token"]
    return {"Authorization": f"Bearer {token}"}


async def test_register_and_duplicate(client):
    resp = await _register(client, "a@example.com")
    assert resp.status_code == 200
    body = resp.json()
    assert body["email"] == "a@example.com"
    assert body["token"]["access_token"]

    duplicate = await _register(client, "a@example.com")
    assert duplicate.status_code == 400


async def test_login_flow(client):
    await _register(client, "b@example.com")

    ok = await client.post(
        f"{API}/auth/login", data={"email": "b@example.com", "password": PASSWORD, "grant_type": "password"}
    )
    assert ok.status_code == 200
    assert ok.json()["access_token"]

    bad = await client.post(
        f"{API}/auth/login", data={"email": "b@example.com", "password": "WrongPass1!", "grant_type": "password"}
    )
    assert bad.status_code == 401


async def test_session_lifecycle(client):
    headers = await _user_headers(client, "c@example.com")

    created = await client.post(f"{API}/auth/session", headers=headers)
    assert created.status_code == 200
    session = created.json()
    session_id = session["session_id"]
    session_headers = {"Authorization": f"Bearer {session['token']['access_token']}"}

    listed = await client.get(f"{API}/auth/sessions", headers=headers)
    assert listed.status_code == 200
    assert any(s["session_id"] == session_id for s in listed.json())

    renamed = await client.patch(
        f"{API}/auth/session/{session_id}/name", data={"name": "renamed"}, headers=session_headers
    )
    assert renamed.status_code == 200
    assert renamed.json()["name"] == "renamed"

    deleted = await client.delete(f"{API}/auth/session/{session_id}", headers=session_headers)
    assert deleted.status_code == 200

    empty = await client.get(f"{API}/auth/sessions", headers=headers)
    assert empty.json() == []


async def test_session_pagination(client):
    headers = await _user_headers(client, "d@example.com")
    for _ in range(5):
        await client.post(f"{API}/auth/session", headers=headers)

    first_page = await client.get(f"{API}/auth/sessions?limit=2&offset=0", headers=headers)
    assert first_page.status_code == 200
    assert len(first_page.json()) == 2

    last_page = await client.get(f"{API}/auth/sessions?limit=2&offset=4", headers=headers)
    assert len(last_page.json()) == 1


async def test_pagination_limit_bounds(client):
    headers = await _user_headers(client, "e@example.com")
    too_large = await client.get(f"{API}/auth/sessions?limit=1000", headers=headers)
    assert too_large.status_code == 422  # limit le=100


async def test_auth_required_error_envelope(client):
    resp = await client.get(f"{API}/auth/sessions")
    assert resp.status_code in (401, 403)
    body = resp.json()
    assert body["error"]["code"]
    assert "request_id" in body["meta"]


async def test_cannot_modify_other_session(client):
    # Two users; user-1 must not rename a session they don't own.
    h1 = await _user_headers(client, "f1@example.com")
    h2 = await _user_headers(client, "f2@example.com")
    s2 = (await client.post(f"{API}/auth/session", headers=h2)).json()
    other_id = s2["session_id"]

    # user-1 creates their own session token but targets user-2's session id.
    s1 = (await client.post(f"{API}/auth/session", headers=h1)).json()
    s1_headers = {"Authorization": f"Bearer {s1['token']['access_token']}"}
    resp = await client.patch(f"{API}/auth/session/{other_id}/name", data={"name": "hijack"}, headers=s1_headers)
    assert resp.status_code == 403

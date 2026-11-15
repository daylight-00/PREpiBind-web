"""The SNU wscan conditions that the app itself answers: the security headers on every response, robots.txt, and no cookie. No GPU and no server: the middleware is driven through the ASGI interface."""
import asyncio
import tomllib
from pathlib import Path

import security_config as sc

ROOT = Path(__file__).resolve().parents[1]
APP_HEADERS = [(b"server", b"uvicorn"), (b"x-frame-options", b"ALLOWALL"), (b"content-type", b"text/html")]


def drive(path="/prepibind/", method="GET", kind="http", base_path="/prepibind"):
    """Send one request through the middleware around a small app that sets its own Server and X-Frame-Options; return (the messages sent to the client, whether the app was called)."""
    sent, called = [], []

    async def app(scope, receive, send):
        called.append(scope["type"])
        await send({"type": "http.response.start", "status": 200, "headers": APP_HEADERS})
        await send({"type": "http.response.body", "body": b"<html>"})

    async def send(message):
        sent.append(message)

    scope = {"type": kind, "method": method, "path": path, "headers": []}
    asyncio.run(sc.SecurityHeaders(app, base_path=base_path)(scope, None, send))
    return sent, bool(called)


def test_every_response_gets_each_header_once_and_replaces_the_apps_own():
    messages, called = drive()
    assert called
    headers = messages[0]["headers"]
    for name, value in sc.HEADERS:
        assert [v for k, v in headers if k == name] == [value]
    assert (b"x-frame-options", b"SAMEORIGIN") in headers and (b"x-content-type-options", b"nosniff") in headers and (b"x-xss-protection", b"1; mode=block") in headers
    assert (b"content-type", b"text/html") in headers and (b"server", b"uvicorn") in headers  # the middleware adds; it does not touch the rest
    assert messages[0]["status"] == 200 and messages[1]["body"] == b"<html>"


def test_a_response_without_headers_still_gets_them():
    async def bare(scope, receive, send):
        await send({"type": "http.response.start", "status": 204})
        await send({"type": "http.response.body", "body": b""})

    sent = []

    async def send(message):
        sent.append(message)

    asyncio.run(sc.SecurityHeaders(bare)({"type": "http", "method": "GET", "path": "/x"}, None, send))
    assert {k for k, _ in sent[0]["headers"]} == {name for name, _ in sc.HEADERS}


def test_websocket_and_lifespan_scopes_pass_through_untouched():
    for kind in ("websocket", "lifespan"):
        messages, called = drive(kind=kind)
        assert called and messages[0]["headers"] == APP_HEADERS


def test_robots_txt_is_answered_at_the_root_and_under_the_base_path_without_calling_the_app():
    for path in ("/robots.txt", "/prepibind/robots.txt"):
        for method in ("GET", "HEAD"):
            messages, called = drive(path=path, method=method)
            headers = messages[0]["headers"]
            assert not called
            assert messages[0]["status"] == 200 and (b"content-type", b"text/plain; charset=utf-8") in headers
            assert messages[1]["body"] == (ROOT / "robots.txt").read_bytes() == b"User-agent: *\nDisallow: /\n"
            assert [v for k, v in headers if k == b"x-content-type-options"] == [b"nosniff"]


def test_other_paths_and_methods_reach_the_app():
    for path, method in (("/prepibind/robots.txt.bak", "GET"), ("/other/robots.txt", "GET"), ("/prepibind/robots.txt", "POST"), ("/prepibind/", "GET")):
        _, called = drive(path=path, method=method)
        assert called, (path, method)


def test_robots_txt_paths_follow_the_base_path():
    assert sc.SecurityHeaders(None, base_path="").robots_paths() == {"/robots.txt"}
    assert sc.SecurityHeaders(None, base_path="prepibind").robots_paths() == {"/robots.txt", "/prepibind/robots.txt"}
    assert sc.SecurityHeaders(None, base_path="/prepibind/").robots_paths() == {"/robots.txt", "/prepibind/robots.txt"}


def test_csp_keeps_streamlit_working_and_everything_else_on_the_own_origin():
    csp = {d.split()[0]: d.split()[1:] for d in sc.CSP.split("; ")}
    assert csp["default-src"] == ["'self'"]
    assert {"'self'", "'unsafe-inline'", "'wasm-unsafe-eval'"} <= set(csp["script-src"]) and {"'self'", "'unsafe-inline'"} <= set(csp["style-src"])
    assert "'unsafe-eval'" not in csp["script-src"]  # JavaScript eval was not needed on any page
    assert {"ws:", "wss:"} <= set(csp["connect-src"])
    assert csp["object-src"] == ["'none'"] and csp["frame-ancestors"] == ["'self'"] and csp["base-uri"] == ["'self'"]
    assert not any(token.startswith(("http:", "https:", "*")) for tokens in csp.values() for token in tokens)


def test_streamlit_run_finds_the_asgi_app_in_serve_py():
    from streamlit.web.server.app_discovery import discover_asgi_app

    found = discover_asgi_app(ROOT / "serve.py")
    assert found.is_asgi_app and found.app_name == "app"


def test_the_xsrf_cookie_stays_off():
    """The wscan scan flags the cookie without HttpOnly and HttpOnly would break uploads, so the app sets none (see security_config.py)."""
    config = tomllib.loads((ROOT / ".streamlit" / "config.toml").read_text())
    assert config["server"]["enableXsrfProtection"] is False

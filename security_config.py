"""
HTTP security headers and robots.txt, as the SNU information-security office asks for them (wscan scan and
service request SR2507-08607, July 2025).

Current Streamlit serves through Starlette and uvicorn, not Tornado, so the headers are added by an ASGI middleware.
`serve.py` gives it to `st.App`: start the server with `streamlit run serve.py` (`streamlit run app.py` still works
but sends none of this).

Every HTTP response (pages, static files, health and host-config routes, uploads) gets, replacing what the app set:
  Content-Security-Policy, X-Frame-Options: SAMEORIGIN, X-Content-Type-Options: nosniff,
  X-XSS-Protection: 1; mode=block (obsolete, browsers ignore it; the scan asks for it), Referrer-Policy
`GET /robots.txt` and `GET <baseUrlPath>/robots.txt` answer with robots.txt of this directory ("User-agent: *", "Disallow: /").

Deliberately not here:
  Server header  uvicorn sends `Server: uvicorn` (no version) and the app cannot remove it; nginx's
                 `more_clear_headers Server` does on the public route.
  cookie flags   the app sets no cookie. Streamlit's only cookie is the XSRF token: its frontend reads it from
                 JavaScript to send it with uploads, so it can never be HttpOnly, and without a login it protects
                 nothing. `.streamlit/config.toml` turns it off; tests/test_security.py keeps it off.
  HSTS           TLS ends in nginx, which sends it.

The CSP allows inline scripts and styles, which Streamlit's frontend needs (an inline script in index.html,
emotion's style tags) and the SNU office accepts, and WebAssembly (its protobuf library tries a wasm module for
64-bit integers and falls back to JavaScript when blocked; `'wasm-unsafe-eval'` does not allow JavaScript eval);
everything else is limited to the own origin and websockets. Checked in Chromium on every page, an upload, a
prediction with its plots, a scan, an evaluation, a download: no violation.
"""
from pathlib import Path

CSP = (
    "default-src 'self'; "
    "script-src 'self' 'unsafe-inline' 'wasm-unsafe-eval'; "
    "style-src 'self' 'unsafe-inline'; "
    "img-src 'self' data: blob:; "
    "font-src 'self' data:; "
    "connect-src 'self' ws: wss:; "
    "media-src 'self' data: blob:; "
    "worker-src 'self' blob:; "
    "object-src 'none'; "
    "base-uri 'self'; "
    "form-action 'self'; "
    "frame-ancestors 'self'"
)
HEADERS = [
    (b"content-security-policy", CSP.encode()),
    (b"x-frame-options", b"SAMEORIGIN"),
    (b"x-content-type-options", b"nosniff"),
    (b"x-xss-protection", b"1; mode=block"),
    (b"referrer-policy", b"strict-origin-when-cross-origin"),
]
ROBOTS_TXT = (Path(__file__).resolve().parent / "robots.txt").read_bytes()


class SecurityHeaders:
    """Pure ASGI middleware: HEADERS on every HTTP response, robots.txt answered here. `base_path` is Streamlit's server.baseUrlPath (read from its config on the first request when not given)."""

    def __init__(self, app, base_path=None):
        self.app = app
        self.base_path = base_path

    def robots_paths(self):
        if self.base_path is None:
            from streamlit import config
            self.base_path = config.get_option("server.baseUrlPath") or ""
        base = self.base_path.strip("/")
        return {"/robots.txt", f"/{base}/robots.txt"} if base else {"/robots.txt"}

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        names = {name for name, _ in HEADERS}

        async def send_with_headers(message):
            if message["type"] == "http.response.start":
                kept = [(k, v) for k, v in message.get("headers", []) if k.lower() not in names]
                message = {**message, "headers": kept + HEADERS}
            await send(message)

        if scope["method"] in ("GET", "HEAD") and scope["path"] in self.robots_paths():
            await send_with_headers({"type": "http.response.start", "status": 200, "headers": [(b"content-type", b"text/plain; charset=utf-8"), (b"content-length", str(len(ROBOTS_TXT)).encode())]})
            await send_with_headers({"type": "http.response.body", "body": ROBOTS_TXT})
            return
        await self.app(scope, receive, send_with_headers)

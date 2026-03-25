"""
Security header configuration for the PREpiBind Streamlit application.

Applied headers:
  - X-Content-Type-Options: nosniff
  - X-Frame-Options: SAMEORIGIN
  - X-XSS-Protection: 1; mode=block
  - Referrer-Policy: strict-origin-when-cross-origin
  - HttpOnly + SameSite=Strict on all cookies

Note: Content-Security-Policy is intentionally omitted. Streamlit injects
inline scripts and styles that cannot be reconciled with a strict CSP without
'unsafe-inline', which would negate its protective value. This limitation is
inherent to the Streamlit framework.

HSTS and Secure cookie flag are commented out; enable them if the server
is deployed behind TLS termination.
"""

import logging
import streamlit as st

logger = logging.getLogger(__name__)


def setup_robots_txt():
    """Serve robots.txt content via Streamlit query parameter fallback."""
    @st.cache_data
    def get_robots_txt():
        return "User-agent: *\nDisallow: /"

    query_params = st.query_params
    if query_params.get("robots") is not None:
        st.text(get_robots_txt())
        st.stop()


def setup_security_headers():
    """Patch Tornado RequestHandler to inject security headers on every response."""
    try:
        import gc
        import tornado.web

        class SecureRobotsHandler(tornado.web.RequestHandler):
            def set_default_headers(self):
                self._apply_security_headers()

            def _apply_security_headers(self):
                self.clear_header("Server")
                self.set_header("X-Content-Type-Options", "nosniff")
                self.set_header("X-Frame-Options", "SAMEORIGIN")
                self.set_header("X-XSS-Protection", "1; mode=block")
                self.set_header("Referrer-Policy", "strict-origin-when-cross-origin")
                # self.set_header("Strict-Transport-Security", "max-age=31536000; includeSubDomains")

            def get(self):
                self.set_header("Content-Type", "text/plain")
                self.write("User-agent: *\nDisallow: /")

        # Monkey-patch Tornado to apply headers to all handlers
        original_init = tornado.web.RequestHandler.__init__
        original_set_cookie = tornado.web.RequestHandler.set_cookie

        def secure_init(self, *args, **kwargs):
            original_init(self, *args, **kwargs)
            try:
                self.clear_header("Server")
                self.set_header("X-Content-Type-Options", "nosniff")
                self.set_header("X-Frame-Options", "SAMEORIGIN")
                self.set_header("X-XSS-Protection", "1; mode=block")
                self.set_header("Referrer-Policy", "strict-origin-when-cross-origin")
            except Exception:
                pass  # Handler not yet attached to a request context

        def secure_set_cookie(self, name, value, **kwargs):
            kwargs.setdefault("httponly", True)
            kwargs.setdefault("samesite", "Strict")
            # kwargs.setdefault("secure", True)  # Enable under TLS
            return original_set_cookie(self, name, value, **kwargs)

        tornado.web.RequestHandler.__init__ = secure_init
        tornado.web.RequestHandler.set_cookie = secure_set_cookie

        # Register robots.txt handler with the running Tornado application
        _registered = False
        for obj in gc.get_objects():
            if hasattr(obj, "add_handlers") and "tornado" in str(type(obj)):
                try:
                    obj.add_handlers(r".*", [(r"/robots\.txt", SecureRobotsHandler)])
                    _registered = True
                    break
                except Exception:
                    continue

        if not _registered:
            logger.debug("security_config: robots.txt handler not registered (Tornado app not yet running)")

    except Exception as e:
        logger.warning("security_config: setup failed: %s", e)


def apply_security():
    """Apply security headers and robots.txt handler."""
    setup_security_headers()
    setup_robots_txt()


if __name__ == "__main__":
    apply_security()

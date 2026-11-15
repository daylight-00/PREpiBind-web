"""ASGI entry point: `streamlit run serve.py` (with the usual --server.* flags) serves app.py with the security headers of security_config.py."""
import streamlit as st
from starlette.middleware import Middleware
from streamlit.runtime.pages_manager import PagesManager

from security_config import SecurityHeaders

PagesManager.uses_pages_directory = False  # before the first request: its legacy mode for a `pages/` directory would run the first session without the navigation of app.py (see app.py)
app = st.App("app.py", middleware=[Middleware(SecurityHeaders)])

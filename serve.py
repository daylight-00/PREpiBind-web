"""ASGI entry point: `streamlit run serve.py` (with the usual --server.* flags) serves app.py with the security headers of security_config.py."""
import streamlit as st
from starlette.middleware import Middleware

from security_config import SecurityHeaders

app = st.App("app.py", middleware=[Middleware(SecurityHeaders)])

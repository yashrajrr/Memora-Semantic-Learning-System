"""Production WSGI entry point for Gunicorn and Render."""

from app import app

__all__ = ["app"]

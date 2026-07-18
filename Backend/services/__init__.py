"""
Service layer package initializer.

Having this file ensures that `services` is treated as a proper
Python package so imports like `from services.db import init_db`
work correctly when running `uvicorn main:app` from the Backend
directory.
"""


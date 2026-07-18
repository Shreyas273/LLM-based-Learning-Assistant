import uvicorn
import sys
import traceback

print("Attempting to import app from main...")
try:
    from main import app
    print("Successfully imported app.")
except Exception:
    print("Failed to import app:")
    traceback.print_exc()
    sys.exit(1)

print("Starting server programmatically...")
try:
    uvicorn.run(app, host="0.0.0.0", port=8000)
except Exception:
    print("Failed to start server:")
    traceback.print_exc()

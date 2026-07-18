import shutil
import os
from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from api.routes import router

# Load environment variables
load_dotenv()

from services.limiter import limiter
from slowapi import _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

app = FastAPI(
    docs_url="/docs",
    redoc_url="/redoc",
    swagger_ui_parameters={"tryItOutEnabled": True}
)
# Setup Rate Limiter
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # Allow all origins for development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)

@app.get("/")
def home():
    return {"message": "Hello, World!"}

@app.post("/uploadfiles/")
def upload_files(file: UploadFile = File(...)):
    file_path = f"uploads/{file.filename}"
    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    return {"status": "file uploaded successfully", "filename": file.filename}

# LLM provider is configured via env (default: Ollama).
print(f"LLM Provider: {os.getenv('LLM_PROVIDER', 'ollama')}")
from pathlib import Path

import uvicorn
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from research_agent.api import router
from research_agent.database import init_db
from research_agent.paths import STATIC_DIR


app = FastAPI(title="Research Copilot", version="0.1.0")
app.include_router(router, prefix="/api")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.on_event("startup")
def startup() -> None:
    init_db()


@app.get("/")
def index() -> FileResponse:
    return FileResponse(Path(STATIC_DIR) / "index.html")


if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000)

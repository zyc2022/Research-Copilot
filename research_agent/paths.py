from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
FILES_DIR = DATA_DIR / "files"
STATIC_DIR = ROOT_DIR / "research_agent" / "static"
DB_PATH = DATA_DIR / "app.db"


def ensure_dirs() -> None:
    DATA_DIR.mkdir(exist_ok=True)
    FILES_DIR.mkdir(exist_ok=True)

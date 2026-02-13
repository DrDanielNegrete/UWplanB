# debug_snapshot.py
from __future__ import annotations

import zipfile
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parent

# Carpeta donde generamos el snapshot (hay que EXCLUIRLA del scan)
SNAP_DIRNAME = "_debug_snapshot"

INCLUDE_EXT = {
    ".py", ".toml", ".txt", ".md", ".yaml", ".yml", ".json", ".ini", ".cfg", ".lock"
}

IGNORE_DIRS = {
    ".git", ".github", "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache",
    ".venv", "venv", "env", ".env", "node_modules", "dist", "build", ".streamlit",
    ".idea", ".vscode",
    SNAP_DIRNAME,  # <- CRÍTICO
}

IGNORE_FILES = {
    ".env", ".env.local", ".env.prod", ".env.production", "secrets.toml",
    "debug_snapshot.py",  # opcional: no hace falta incluirlo
}

def should_ignore(path: Path) -> bool:
    # Ignora cualquier cosa dentro de carpetas ignoradas
    for part in path.parts:
        if part in IGNORE_DIRS:
            return True
    if path.name in IGNORE_FILES:
        return True
    # Ignora snapshots previos (zip) para evitar ruido
    if path.name.startswith("snapshot_") and path.suffix.lower() == ".zip":
        return True
    return False

def build_tree(root: Path) -> str:
    lines = []
    for p in sorted(root.rglob("*")):
        if should_ignore(p) or p.is_dir():
            continue
        lines.append(str(p.relative_to(root)))
    return "\n".join(lines)

def main() -> None:
    out_dir = ROOT / SNAP_DIRNAME
    files_dir = out_dir / "files"
    out_dir.mkdir(exist_ok=True)
    files_dir.mkdir(parents=True, exist_ok=True)

    # 1) tree.txt
    (out_dir / "tree.txt").write_text(build_tree(ROOT), encoding="utf-8")

    # 2) Copiar archivos relevantes
    copied = 0
    for p in ROOT.rglob("*"):
        if should_ignore(p) or p.is_dir():
            continue
        if p.suffix.lower() not in INCLUDE_EXT:
            continue

        rel = p.relative_to(ROOT)
        dest = files_dir / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        try:
            dest.write_bytes(p.read_bytes())
            copied += 1
        except Exception as e:
            (out_dir / "copy_errors.txt").open("a", encoding="utf-8").write(
                f"{rel}: {e}\n"
            )

    # 3) Crear zip
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    zip_path = ROOT / f"snapshot_{stamp}.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for p in out_dir.rglob("*"):
            if p.is_dir():
                continue
            z.write(p, p.relative_to(out_dir))

    print(f"OK ✅ Archivos copiados: {copied}")
    print(f"Generado: {zip_path}")

if __name__ == "__main__":
    main()

# debug_snapshot.py
from __future__ import annotations

import os
import zipfile
from pathlib import Path
from datetime import datetime

# Ajusta si tu proyecto vive en otra ruta
ROOT = Path(__file__).resolve().parent

# Qué incluir (extensiones típicas de Streamlit/Python)
INCLUDE_EXT = {
    ".py", ".toml", ".txt", ".md", ".yaml", ".yml", ".json", ".ini", ".cfg", ".lock"
}

# Carpetas a ignorar (ruido)
IGNORE_DIRS = {
    ".git", ".github", "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache",
    ".venv", "venv", "env", ".env", "node_modules", "dist", "build", ".streamlit",
    ".idea", ".vscode"
}

# Archivos a ignorar por nombre
IGNORE_FILES = {
    ".env", ".env.local", ".env.prod", ".env.production", "secrets.toml"
}

def should_ignore(path: Path) -> bool:
    parts = set(path.parts)
    if parts & IGNORE_DIRS:
        return True
    if path.name in IGNORE_FILES:
        return True
    return False

def build_tree(root: Path) -> str:
    lines = []
    for p in sorted(root.rglob("*")):
        if should_ignore(p):
            continue
        rel = p.relative_to(root)
        if p.is_dir():
            continue
        lines.append(str(rel))
    return "\n".join(lines)

def main() -> None:
    out_dir = ROOT / "_debug_snapshot"
    files_dir = out_dir / "files"
    out_dir.mkdir(exist_ok=True)
    files_dir.mkdir(exist_ok=True)

    # 1) tree.txt
    tree_txt = build_tree(ROOT)
    (out_dir / "tree.txt").write_text(tree_txt, encoding="utf-8")

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
            # Si algún archivo falla, no detengas todo
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

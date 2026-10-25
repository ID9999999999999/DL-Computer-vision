"""Install into a project-local environment, choose images and run Lab 06."""
import os
from pathlib import Path
import subprocess
import sys
import venv

ROOT = Path(__file__).resolve().parent


def main():
    if not (3, 10) <= sys.version_info[:2] <= (3, 12):
        raise SystemExit("Please install Python 3.11 (recommended), or 3.10/3.12, then launch again.")
    print("Lab 06 — choose the folder containing metadata.jsonl, good/ and bad/.", flush=True)
    try:
        import tkinter as tk
        from tkinter.filedialog import askdirectory
        window = tk.Tk()
        window.withdraw()
        location = askdirectory(title="Select demo_quality_dataset (metadata.jsonl + good + bad)")
        window.destroy()
    except Exception:
        location = input("Dataset folder path: ").strip().strip('"')
    if not location:
        raise SystemExit("No folder selected; nothing was trained.")
    dataset = Path(location).expanduser().resolve()
    if not all((dataset / item).exists() for item in ("metadata.jsonl", "good", "bad")):
        raise SystemExit("Wrong folder: select demo_quality_dataset itself, containing metadata.jsonl, good/ and bad/.")
    environment = ROOT / ".venv"
    python = environment / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    if not python.exists():
        print("Creating the Python environment...", flush=True)
        venv.EnvBuilder(with_pip=True).create(environment)
    marker = environment / "lab06_requirements_installed.txt"
    requirements = (ROOT / "requirements.txt").read_text(encoding="utf-8")
    if not marker.exists() or marker.read_text(encoding="utf-8") != requirements:
        print("Installing training libraries (Internet required; this can take several minutes)...", flush=True)
        subprocess.run([str(python), "-m", "pip", "install", "--upgrade", "pip"], check=True)
        subprocess.run([str(python), "-m", "pip", "install", "-r", str(ROOT / "requirements.txt")], check=True)
        marker.write_text(requirements, encoding="utf-8")
    subprocess.run([str(python), str(ROOT / "lab06.py"), "run", "--dataset", str(dataset)], cwd=ROOT, check=True)
    print("Finished. Open results/Final_Report.md and results/Presentation_5min.md.", flush=True)


if __name__ == "__main__":
    try:
        main()
    except (Exception, KeyboardInterrupt) as exc:
        print("Stopped:", exc, flush=True)
        print("The work is not complete. Keep the error message and send it for correction.", flush=True)
        sys.exit(1)

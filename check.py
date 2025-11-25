from pathlib import Path
import sys

txt = Path(sys.argv[1])

missing = False

for line in txt.read_text().splitlines():
    path = Path(line.strip())
    if not line.strip():
        continue
    if not path.is_file():
        print("[MISSING]", path)
        missing = True

if not missing:
    print("All exist.")

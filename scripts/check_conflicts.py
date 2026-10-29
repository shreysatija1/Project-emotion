import re
from pathlib import Path

def search_file(path, keywords):
    print(f"\n=== Searching {path} ===")
    text = Path(path).read_text(encoding="utf-8")
    for kw in keywords:
        matches = [m.start() for m in re.finditer(re.escape(kw), text, re.IGNORECASE)]
        print(f"Keyword '{kw}': {len(matches)} occurrences")
        for m in matches:
            start = max(0, m - 60)
            end = min(len(text), m + len(kw) + 60)
            snippet = text[start:end].replace('\n', ' ')
            print(f"  [{m}] ...{snippet}...")

keywords = [
    "proves", "proven", "dominant hidden factor", "rely on speaker overlap",
    "98.5%", "95.9%", "Att-HACK", "12% to 21%", "12-21", "-1.5%", "+3.5%",
    "15-25", "12-25", "-7 to +11", "-2 to +5", "48-55%", "48-55"
]

search_file("WEEK4_FINDINGS_NOTE.md", keywords)
search_file("scripts/generate_week4_deck.py", keywords)

from pathlib import Path
import re

note_text = Path("WEEK4_FINDINGS_NOTE.md").read_text(encoding="utf-8")
deck_text = Path("scripts/generate_week4_deck.py").read_text(encoding="utf-8")

# Let's inspect slide by slide in generate_week4_deck.py
slide_blocks = re.split(r"# SLIDE \d+:", deck_text)
print(f"Found {len(slide_blocks)} slide blocks")
for idx, sb in enumerate(slide_blocks[1:], start=1):
    print(f"\n--- SLIDE {idx} ---")
    # print lines that contain numbers or claims
    for line in sb.splitlines()[:35]:
        if any(term in line.lower() for term in ["drop", "shift", "leak", "aesdd", "att-hack", "prove", "dominant", "overlap", "48-", "55%"]):
            print("  ", line.strip())

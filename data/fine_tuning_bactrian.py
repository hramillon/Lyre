import json
import re
import random

def is_clean_text(text):
    if not text:
        return True
    # Rejette si contient des caractères non-latins (CJK, arabe, cyrillique, etc.)
    non_latin = re.findall(r'[^\x00-\x7F\u00C0-\u024F]', text)
    ratio = len(non_latin) / max(len(text), 1)
    if ratio > 0.01:
        return False
    # Rejette les phonétiques IPA /ˈɛskəbɑːr/
    if re.search(r'/[ˈˌæɛɪɔʊəɑɒʌɜːˑ̃].{1,30}/', text):
        return False
    return True

def convert_bactrian(input_file="fr.json", output_file="bactrian_fr.jsonl"):
    print("Lecture...")
    with open(input_file, "r", encoding="utf-8", errors="replace") as f:
        data = json.load(f)

    print(f"{len(data)} exemples trouvés")
    kept, skipped = 0, 0

    with open(output_file, "w", encoding="utf-8") as f:
        for example in data:
            instruction = (example.get("instruction") or "").strip()
            input_text  = (example.get("input") or "").strip()
            output_text = (example.get("output") or "").strip()

            user      = instruction
            context   = input_text if input_text else None
            assistant = output_text

            # Filtres qualité
            if not user or not assistant:
                skipped += 1; continue
            if len(assistant) < 10:
                skipped += 1; continue
            if not is_clean_text(user) or not is_clean_text(assistant) or not is_clean_text(context):
                skipped += 1; continue

            # Garde seulement 1/10 des exemples sans contexte
            if context is None and random.random() > 0.1:
                skipped += 1; continue

            entry = {
                "context": context,
                "user": user,
                "assistant": assistant
            }
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            kept += 1

    print(f"Gardés : {kept} | Ignorés : {skipped} → {output_file}")

random.seed(42)
convert_bactrian()
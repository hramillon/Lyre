import json
import os
from datasets import load_dataset

MAX_BYTES = 12 * 1024 * 1024 * 1024  # 12 Go en octets
output_file = "culturax_fr.jsonl"
count = 0

dataset = load_dataset("uonlp/CulturaX", "fr", split="train", streaming=True)

print("Début de l'extraction de CulturaX-fr (limite : 12 Go)...")

with open(output_file, "w", encoding="utf-8") as f:
    for example in dataset:
        entry = {
            "context": example.get("text", ""),
            "user": "",
            "assistant": ""
        }
        
        line = json.dumps(entry, ensure_ascii=False) + "\n"
        f.write(line)
        
        count += 1
        if count % 50000 == 0:
            current_size = os.path.getsize(output_file)
            print(f"{count} lignes écrites | Taille : {current_size / (1024**3):.2f} Go")
            
            if current_size >= MAX_BYTES:
                print("Limite de 12 Go atteinte. Arrêt.")
                break

print(f"Terminé ! {count} exemples extraits. Taille finale : {os.path.getsize(output_file) / (1024**3):.2f} Go")

import requests
import os
import json
import math
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics.pairwise import cosine_similarity


OLLAMA_URL = "http://localhost:11434/api/embed"
MODEL_NAME = "bge-m3"
BATCH_SIZE = 8 

def is_valid_text(text: str) -> bool:
    if not text:
        return False
    text = text.strip()
    if len(text) < 10:     
        return False
    if text.count('.') > len(text) * 0.5:
        return False
    return True


def create_embedding(text_list):
    r = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL_NAME,
            "input": text_list
        },
        timeout=120
    )

    if r.status_code != 200:
        raise RuntimeError(r.text)

    data = r.json()
    return data["embeddings"]


jsons = os.listdir("outputs")
my_dict = []
chunk_id = 0

for json_file in jsons:
    with open(os.path.join("outputs", json_file), encoding="utf-8") as f:
        content = json.load(f)

    valid_chunks = []
    texts = []

    for c in content["segments"]:
        t = c.get("text", "")
        if is_valid_text(t):
            c["text"] = t.strip()
            valid_chunks.append(c)
            texts.append(c["text"])
        else:
            print(" ", repr(t))

    for i in range(0, len(texts), BATCH_SIZE):
        batch_texts = texts[i:i+BATCH_SIZE]
        batch_chunks = valid_chunks[i:i+BATCH_SIZE]

        try:
            embeddings = create_embedding(batch_texts)
        except RuntimeError as e:
            continue

        for chunk, emb in zip(batch_chunks, embeddings):
            if any(math.isnan(x) for x in emb):
                continue

            chunk["chunk_id"] = chunk_id
            chunk["embedding"] = emb
            my_dict.append(chunk)

            chunk_id += 1

    break

df= pd.DataFrame.from_records(my_dict)
# print(df)
joblib.dump(df,"embeddings.joblib")


 
import requests
import os
import json
import math
import numpy as np
import pandas as pd
import joblib
from read_chunks import create_embedding
from sklearn.metrics.pairwise import cosine_similarity

OLLAMA_URL = "http://localhost:11434/api/embed"
MODEL_NAME = "bge-m3"


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

def inference(prompt):
    r = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "llama3.2",
            "prompt": prompt,
            "stream":False
        },
    )
    response=r.json()
    print(response)
    return response

df=joblib.load('embeddings.joblib')

incoming_query= input("Ask a Question: ")
question_embedding = create_embedding([incoming_query])[0]


similarities= cosine_similarity(np.vstack(df['embedding']),[question_embedding]).flatten()
top_results=5
max_idx=similarities.argsort()[::-1][0:top_results]

print(max_idx)
new_df=df.loc[max_idx]

prompt=f'''I have a downloaded course related to web development
Here are the video subtitle chunks containing video title, 
video number, start time in seconds, end time in seconds, the text at that time:
{new_df[["id","start", "end","text"]].to_json(orient="records")}
----------------------------------------------
"{incoming_query}"
User asked this question related to the video chunks, you have to answer where and how much content is taught where 
is taught in which video (in which video at what timestamp) and guide the user to go to that particular video.
If user asks an unrelated question, tell him you can answer only according to the course.
Just answer the question concisely, precisely with accuracy don't give follow up questions
'''
with open("prompt.txt","w")as f:
    f.write(prompt)

response= inference(prompt)["response"]
print(response)

with open("response.txt","w")as f:
    f.write(response)
# for index, item in new_df.iterrows():
#     print(index, item["id"],item["start"], item["end"], item["text"])
# main.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import sqlite3
import os
import uuid
from typing import List, Optional

# Embedding & NLP
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import numpy as np
import faiss

DB_PATH = 'monasteries.db'
VEC_DIM = 384  # dimensionality of all-MiniLM-L6-v2
FAISS_INDEX_PATH = 'faiss.index'

app = FastAPI(title="Monastery360 AI Search API")

# ✅ Enable CORS so your website frontend can call the backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 🔒 in production replace "*" with ["https://monastery360.com"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load models once
embed_model = SentenceTransformer('all-MiniLM-L6-v2')
classifier = pipeline('zero-shot-classification', model='facebook/bart-large-mnli')

# Initialize DB
def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS monasteries (
                 id TEXT PRIMARY KEY,
                 name TEXT,
                 location TEXT,
                 lat REAL,
                 lon REAL,
                 description TEXT,
                 year_founded INTEGER,
                 metadata_json TEXT
                 )''')
    conn.commit()
    conn.close()

# Initialize or load FAISS
def init_faiss():
    if os.path.exists(FAISS_INDEX_PATH):
        idx = faiss.read_index(FAISS_INDEX_PATH)
    else:
        idx = faiss.IndexFlatIP(VEC_DIM)  # inner product on normalized vectors
    return idx

init_db()
faiss_index = init_faiss()

# Keep an in-memory mapping of row index -> monastery id
id_map = []
if os.path.exists('id_map.txt'):
    with open('id_map.txt','r') as f:
        id_map = [line.strip() for line in f.readlines()]

def save_id_map():
    with open('id_map.txt','w') as f:
        f.write('\n'.join(id_map))

class MonasteryIn(BaseModel):
    name: str
    location: Optional[str] = None
    lat: Optional[float] = None
    lon: Optional[float] = None
    description: Optional[str] = ''
    year_founded: Optional[int] = None
    metadata_json: Optional[str] = None

@app.post('/ingest')
async def ingest(mon: MonasteryIn):
    mid = str(uuid.uuid4())
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('INSERT INTO monasteries (id,name,location,lat,lon,description,year_founded,metadata_json) VALUES (?,?,?,?,?,?,?,?)',
              (mid, mon.name, mon.location, mon.lat, mon.lon, mon.description, mon.year_founded, mon.metadata_json))
    conn.commit()
    conn.close()

    # compute embedding and add to FAISS
    text = (mon.name or '') + '. ' + (mon.description or '')
    emb = embed_model.encode([text], convert_to_numpy=True)
    emb_norm = emb / np.linalg.norm(emb, axis=1, keepdims=True)
    faiss_index.add(emb_norm)
    id_map.append(mid)
    save_id_map()
    faiss.write_index(faiss_index, FAISS_INDEX_PATH)
    return {"id": mid, "status": "ingested"}

class SearchReq(BaseModel):
    query: str
    k: Optional[int] = 5

@app.post('/search')
async def search(req: SearchReq):
    q_emb = embed_model.encode([req.query], convert_to_numpy=True)
    q_emb = q_emb / np.linalg.norm(q_emb, axis=1, keepdims=True)
    D, I = faiss_index.search(q_emb, req.k)
    results = []
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    for score, idx in zip(D[0], I[0]):
        if idx < 0 or idx >= len(id_map):
            continue
        mid = id_map[idx]
        c.execute('SELECT id,name,location,lat,lon,description,year_founded,metadata_json FROM monasteries WHERE id=?', (mid,))
        row = c.fetchone()
        if not row:
            continue
        # zero-shot categorize
        text = row[5] or ''
        candidate_labels = ["Historic","Pilgrimage site","Tourist-friendly","Festival site",
                            "Architectural marvel","Small monastery","Large monastery",
                            "17th century","18th century"]
        cat = classifier(text or row[1], candidate_labels, multi_label=False)
        results.append({
            "id": row[0],
            "name": row[1],
            "location": row[2],
            "lat": row[3],
            "lon": row[4],
            "description": row[5],
            "year_founded": row[6],
            "metadata": row[7],
            "score": float(score),
            "top_label": cat['labels'][0] if cat['labels'] else "",
            "label_score": float(cat['scores'][0]) if cat['scores'] else 0.0
        })
    conn.close()
    return {"query": req.query, "results": results}

@app.get('/health')
async def health():
    return {"status":"ok"}

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8000)

# Install dependencies
# pip install fastapi uvicorn sentence-transformers transformers numpy faiss-cpu

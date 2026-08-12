# Backend

Reusable inference and API code lives in `backend/app/`.

Run the local inference smoke test from the project root after generating
`pipeline/artifacts/`:

```bash
python3 pipeline/training/export_metadata_artifacts.py
```

```bash
python3 -m backend.app.inference
```

You can also pass explicit symptom IDs:

```bash
python3 -m backend.app.inference SMTS00012 SMTS00420 SMTS01081
```

## Local API

Run the FastAPI server from the project root:

```bash
uvicorn backend.app.main:app --reload
```

Then call:

```bash
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/symptoms
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"symptom_ids":["SMTS00012","SMTS00420","SMTS01081"],"top_syndromes":5,"top_herbs":5}'
```

Local CORS allows `http://localhost:5173` and `http://localhost:3000`.
Set `BACKEND_CORS_ORIGINS` to a comma-separated origin list for production.

# TCMNet Frontend

Expo React Native app for TCMNet. It supports local development and Expo Web
deployment.

## Setup

Install frontend dependencies:

```bash
cd frontend
npm install
```

## Browser-Side Inference

The deployed web app runs inference in the browser from bundled frozen
artifacts. A hosted FastAPI backend is no longer required for the frontend.

If the frozen artifacts in `pipeline/artifacts/` change, regenerate the browser
bundle from the project root:

```bash
python3 tools/export_frontend_artifacts.py
```

## Run Expo

```bash
cd frontend
npm run start
```

For web:

```bash
npm run web
```

## Deploy To Vercel

Deploy the `frontend` folder as the Vercel project root.

Use:

```bash
npm run build
```

Output directory:

```bash
dist
```

No backend URL environment variable is needed for the web deployment.

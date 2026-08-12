# TCMNet Frontend

Expo React Native app for TCMNet. It supports local development and Expo Web
deployment.

## Setup

Install frontend dependencies:

```bash
cd frontend
npm install
```

## Backend

From the project root, run the FastAPI backend:

```bash
uvicorn backend.app.main:app --reload
```

The frontend defaults to:

```bash
http://localhost:8000
```

Override it when needed:

```bash
EXPO_PUBLIC_API_BASE_URL=http://localhost:8000 npm run web
```

For testing on a physical phone, use your computer's LAN IP instead of
`localhost`, for example `http://192.168.1.25:8000`.

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

Set this environment variable in Vercel:

```bash
EXPO_PUBLIC_API_BASE_URL=https://your-backend-url
```

The Python/FastAPI backend should be deployed separately, then its deployed URL
should be used as `EXPO_PUBLIC_API_BASE_URL`.

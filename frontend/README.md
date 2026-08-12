# TCMNet Frontend

Expo React Native app for the TCMNet research prototype. It supports native local
development and Expo Web for a future web deployment.

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

For a future Vercel deployment with Expo Web, set `EXPO_PUBLIC_API_BASE_URL` to
the deployed backend URL during the build.

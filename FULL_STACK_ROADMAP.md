# Full Stack Roadmap

## Goal
Turn the current TCMNet research project into a full stack web app with:

- `pipeline/` for offline data processing, synthetic data generation, training, evaluation, and artifacts.
- `backend/` for API endpoints and model inference.
- `frontend/` for the user-facing symptom picker and prediction results.

## Step-By-Step Plan

1. Freeze the pipeline.
   - Create reusable training artifacts.
   - Save trained model weights.
   - Save stable mappings for symptoms, syndromes, herbs, and concepts.
   - Save model configuration such as input size, concept count, and syndrome count.

2. Create an inference module.
   - Load saved model artifacts.
   - Convert selected symptom IDs into the exact model input vector.
   - Return top syndrome predictions, concept scores, and herb recommendations.

3. Build the backend.
   - Use FastAPI.
   - Add endpoints such as `/health`, `/symptoms`, and `/predict`.
   - Keep model loading centralized so requests do not retrain the model.

4. Add app metadata.
   - Provide frontend-readable symptom, syndrome, herb, and concept metadata.
   - Avoid hard-coding model IDs in the frontend.

5. Build the frontend.
   - Add searchable symptom selection.
   - Show selected symptoms.
   - Display syndrome predictions, concept chart, herb recommendations, and explanations.

6. Connect frontend to backend.
   - Fetch symptoms from the backend.
   - Send selected symptom IDs to `/predict`.
   - Render structured prediction results.

7. Add explanation logic.
   - Show matching symptoms.
   - Show concept alignment.
   - Show known syndrome-herb associations.
   - Explain herb ranking scores.

8. Add safety guardrails.
   - Clearly label the app as educational/research use only.
   - Avoid presenting output as medical advice.

9. Add persistence if needed.
   - Optional database for saved cases, feedback, prediction history, or admin review.

10. Deploy.
    - Deploy the frontend to Vercel.
    - Choose how to host model inference.
    - Include model artifacts with whichever service runs inference.

## Vercel Deployment Plan

Vercel should be the primary home for the frontend. The model-serving backend needs an explicit choice because Python ML inference is heavier than a normal static web app.

Recommended architecture:

- `frontend/`: React or Next.js app deployed on Vercel.
- `backend/`: FastAPI service deployed separately on a Python-friendly host.
- `pipeline/`: offline training/export workflow, not deployed as a public app.

Good backend hosting options:

- Render, Railway, Fly.io, or Google Cloud Run for the FastAPI inference service.
- Vercel frontend calls the backend through an environment variable such as `VITE_API_BASE_URL` or `NEXT_PUBLIC_API_BASE_URL`.

Alternative architecture:

- Use Next.js on Vercel for the frontend and lightweight API routes.
- Keep heavy PyTorch inference outside Vercel unless the model is converted to a smaller format and fits serverless limits.

Deployment steps:

1. Build and export pipeline artifacts locally.
   - `pipeline/artifacts/tcmnet.pt`
   - model config
   - symptom mappings
   - syndrome mappings
   - herb mappings
   - concept order

2. Build the FastAPI backend.
   - Load artifacts at server startup.
   - Expose `/health`, `/symptoms`, and `/predict`.
   - Add CORS so the Vercel frontend can call it.

3. Deploy backend to a Python-friendly host.
   - Include artifacts with the backend deployment.
   - Set any needed environment variables.
   - Confirm `/health` and `/predict` work from a public URL.

4. Build the frontend for Vercel.
   - Use React/Vite or Next.js.
   - Store backend URL in Vercel environment variables.
   - Call `/symptoms` and `/predict` from the deployed backend.

5. Deploy frontend to Vercel.
   - Connect the GitHub repo.
   - Set the frontend root directory.
   - Set the backend API URL environment variable.
   - Verify the deployed frontend can reach the deployed backend.

6. Add production guardrails.
   - Educational/research disclaimer.
   - Error states when backend is unavailable.
   - Basic request validation.
   - Avoid storing user health information unless persistence is intentionally added.

## Current Next Step
Start with Step 1: freeze the pipeline by creating saved model artifacts and stable mappings.

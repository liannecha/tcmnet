// API helpers for the React Native app.
// These functions connect to the FastAPI backend by building HTTP requests
// from API_BASE_URL, sending/receiving JSON, and surfacing backend errors.
import { API_BASE_URL } from './config';
import type { MetadataRecord, PredictRequest, PredictionResponse } from './types';

async function requestJson<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE_URL}${path}`, {
    ...init,
    headers: {
      'Content-Type': 'application/json',
      ...init?.headers,
    },
  });

  if (!response.ok) {
    const body = await response.text();
    throw new Error(`${response.status} ${response.statusText}${body ? `: ${body}` : ''}`);
  }

  return response.json() as Promise<T>;
}

export async function fetchSymptoms(): Promise<MetadataRecord[]> {
  const payload = await requestJson<{ symptoms: MetadataRecord[] }>('/symptoms');
  return payload.symptoms;
}

export async function predict(payload: PredictRequest): Promise<PredictionResponse> {
  return requestJson<PredictionResponse>('/predict', {
    method: 'POST',
    body: JSON.stringify(payload),
  });
}

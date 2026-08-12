// API-shaped helpers backed by local browser inference.
import type { MetadataRecord, PredictRequest, PredictionResponse } from './types';
import { fetchLocalSymptoms, predictLocal } from './localInference';

export async function fetchSymptoms(): Promise<MetadataRecord[]> {
  return fetchLocalSymptoms();
}

export async function predict(payload: PredictRequest): Promise<PredictionResponse> {
  return predictLocal(payload);
}

import type { BodyRegion } from './bodyMapRegions';

export type MetadataRecord = {
  id: string;
  label: string;
  english_name?: string;
  chinese_name?: string;
};

export type PatientIntakeBasics = {
  sexGender: string;
  age: number | null;
  mainConcern: string;
  onsetDate: string;
  onsetUnknown: boolean;
  severity: number | null;
};

export type PatientIntakeSymptom = {
  key: string;
  label: string;
  ids: string[];
};

export type PatientIntake = PatientIntakeBasics & {
  selectedBodyRegionIds: string[];
  selectedBodyRegions: BodyRegion[];
  selectedSymptoms: PatientIntakeSymptom[];
  plainLanguageSymptoms?: string[];
};

export type PredictRequest = {
  symptom_ids: string[];
  top_syndromes: number;
  top_herbs: number;
};

export type SyndromePrediction = {
  index: number;
  syndrome_id: string;
  label: string;
  english_name: string;
  chinese_name: string;
  description: string;
  confidence: number;
};

export type ConceptScore = {
  id: string;
  label: string;
  score: number;
};

export type HerbRecommendation = {
  herb_id: string;
  label: string;
  english_name: string;
  chinese_name: string;
  description: string;
  target_concepts: string[];
  score: number;
  concept_similarity: number;
  syndrome_prior: number;
  known_for_predicted_syndrome: boolean;
};

export type Explanation = {
  matching_symptoms: MetadataRecord[];
  concept_alignment: ConceptScore[];
  syndrome_herb_associations: {
    syndrome_id: string;
    label: string;
    english_name: string;
    chinese_name: string;
    associated_herbs: MetadataRecord[];
    total_associated_herbs: number;
  };
  herb_ranking: {
    formula: string;
    alpha: number;
    items: Array<{
      herb_id: string;
      label: string;
      english_name: string;
      chinese_name: string;
      description: string;
      target_concepts: string[];
      concept_similarity: number;
      syndrome_prior: number;
      score: number;
      known_for_predicted_syndrome: boolean;
    }>;
  };
};

export type PredictionResponse = {
  input: {
    requested_symptom_ids: string[];
    known_symptom_ids: string[];
    unknown_symptom_ids: string[];
  };
  syndromes: SyndromePrediction[];
  concepts: ConceptScore[];
  herbs: HerbRecommendation[];
  explanation: Explanation;
};

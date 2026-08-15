import artifactsPayload from './generated/tcmnet-artifacts.json';
import type {
  ConceptScore,
  HerbRecommendation,
  MetadataRecord,
  PredictRequest,
  PredictionResponse,
  SyndromePrediction,
} from './types';

type Artifacts = {
  model_config: {
    num_symptoms: number;
    num_concepts: number;
    num_syndromes: number;
    num_herbs?: number;
  };
  symptom_mapping: {
    columns: string[];
    symptom_id_to_index: Record<string, number>;
  };
  syndrome_mapping: {
    index_to_id: string[];
  };
  concept_mapping: {
    labels: string[];
  };
  herb_mapping: {
    herb_ids: string[];
    recommendation_alpha_default?: number;
  };
  symptoms_metadata: MetadataRecord[];
  syndromes_metadata: Array<MetadataRecord & { description?: string }>;
  herbs_metadata: Array<MetadataRecord & { description?: string; target_concepts?: string[] }>;
  concepts_metadata: MetadataRecord[];
  herb_concept_matrix: number[][];
  syndrome_herb_prior: number[][];
  weights: {
    shared_weight: number[][];
    shared_bias: number[];
    concept_weight: number[][];
    concept_bias: number[];
    syndrome_hidden_weight: number[][];
    syndrome_hidden_bias: number[];
    syndrome_output_weight: number[][];
    syndrome_output_bias: number[];
    herb_hidden_weight?: number[][];
    herb_hidden_bias?: number[];
    herb_output_weight?: number[][];
    herb_output_bias?: number[];
  };
};

const artifacts = artifactsPayload as Artifacts;

const symptomColumns = artifacts.symptom_mapping.columns.map(String);
const symptomIdToIndex = artifacts.symptom_mapping.symptom_id_to_index;
const syndromeIndexToId = artifacts.syndrome_mapping.index_to_id;
const conceptLabels = artifacts.concept_mapping.labels;
const herbIds = artifacts.herb_mapping.herb_ids;
const recommendationAlpha = artifacts.herb_mapping.recommendation_alpha_default ?? 0.7;

const symptomLabelById = metadataLabelMap(artifacts.symptoms_metadata);
const syndromeMetadataById = metadataRecordMap(artifacts.syndromes_metadata);
const herbMetadataById = metadataRecordMap(artifacts.herbs_metadata);

export async function fetchLocalSymptoms(): Promise<MetadataRecord[]> {
  return artifacts.symptoms_metadata;
}

export async function predictLocal(request: PredictRequest): Promise<PredictionResponse> {
  const requestedSymptoms = request.symptom_ids.map(String);
  const { inputVector, knownSymptoms, unknownSymptoms } = vectorize(requestedSymptoms);

  const sharedFeatures = relu(
    linear(inputVector, artifacts.weights.shared_weight, artifacts.weights.shared_bias),
  );
  const conceptScores = sigmoidVector(
    linear(sharedFeatures, artifacts.weights.concept_weight, artifacts.weights.concept_bias),
  );
  const syndromeInput = sharedFeatures.concat(conceptScores);
  const syndromeHidden = relu(
    linear(
      syndromeInput,
      artifacts.weights.syndrome_hidden_weight,
      artifacts.weights.syndrome_hidden_bias,
    ),
  );
  const syndromeLogits = linear(
    syndromeHidden,
    artifacts.weights.syndrome_output_weight,
    artifacts.weights.syndrome_output_bias,
  );
  const herbWeights = getHerbHeadWeights();
  const herbInput = sharedFeatures.concat(conceptScores, syndromeLogits);
  const herbHidden = relu(
    linear(herbInput, herbWeights.herbHiddenWeight, herbWeights.herbHiddenBias),
  );
  const herbScores = linear(
    herbHidden,
    herbWeights.herbOutputWeight,
    herbWeights.herbOutputBias,
  );
  const syndromeProbabilities = softmax(syndromeLogits);

  const syndromes = topSyndromes(syndromeProbabilities, request.top_syndromes);
  const predSyndromeIdx = syndromes[0]?.index ?? 0;
  const herbs = recommendHerbs(
    conceptScores,
    herbScores,
    predSyndromeIdx,
    request.top_herbs,
  );
  const concepts = conceptLabels.map((label, index) => ({
    id: label,
    label,
    score: conceptScores[index],
  }));

  return {
    input: {
      requested_symptom_ids: requestedSymptoms,
      known_symptom_ids: knownSymptoms,
      unknown_symptom_ids: unknownSymptoms,
    },
    syndromes,
    concepts,
    herbs,
    explanation: buildExplanation(
      knownSymptoms,
      concepts,
      predSyndromeIdx,
      herbs,
      recommendationAlpha,
    ),
  };
}

function linear(input: number[], weight: number[][], bias: number[]) {
  return weight.map((row, rowIndex) => {
    let sum = bias[rowIndex] ?? 0;
    for (let index = 0; index < input.length; index += 1) {
      sum += row[index] * input[index];
    }
    return sum;
  });
}

function relu(values: number[]) {
  return values.map((value) => Math.max(0, value));
}

function sigmoidVector(values: number[]) {
  return values.map(sigmoidValue);
}

function sigmoidValue(value: number) {
  return 1 / (1 + Math.exp(-value));
}

function softmax(values: number[]) {
  const maxValue = Math.max(...values);
  const expValues = values.map((value) => Math.exp(value - maxValue));
  const total = expValues.reduce((sum, value) => sum + value, 0);
  return expValues.map((value) => value / total);
}

function getHerbHeadWeights() {
  const {
    herb_hidden_weight: herbHiddenWeight,
    herb_hidden_bias: herbHiddenBias,
    herb_output_weight: herbOutputWeight,
    herb_output_bias: herbOutputBias,
  } = artifacts.weights;

  if (!herbHiddenWeight || !herbHiddenBias || !herbOutputWeight || !herbOutputBias) {
    throw new Error(
      'Local TCMNet artifacts do not include herb_head weights. Regenerate frontend artifacts before using neural herb recommendations.',
    );
  }

  return {
    herbHiddenWeight,
    herbHiddenBias,
    herbOutputWeight,
    herbOutputBias,
  };
}

function vectorize(symptomIds: string[]) {
  const inputVector = Array.from({ length: symptomColumns.length }, () => 0);
  const knownSymptoms: string[] = [];
  const unknownSymptoms: string[] = [];

  for (const symptomId of symptomIds) {
    const lookupKey = lookupSymptomKey(symptomId);
    if (lookupKey === null) {
      unknownSymptoms.push(symptomId);
      continue;
    }
    inputVector[symptomIdToIndex[lookupKey]] = 1;
    knownSymptoms.push(symptomId);
  }

  return { inputVector, knownSymptoms, unknownSymptoms };
}

function lookupSymptomKey(symptomId: string) {
  const raw = symptomId.trim();
  const candidates = [raw];
  const upper = raw.toUpperCase();

  if (upper.startsWith('SMTS')) {
    const numeric = upper.replace(/^SMTS/, '').replace(/^0+/, '') || '0';
    candidates.push(numeric, `SMTS${Number(numeric).toString().padStart(5, '0')}`);
  } else if (/^\d+$/.test(raw)) {
    const numeric = raw.replace(/^0+/, '') || '0';
    candidates.push(numeric, `SMTS${Number(numeric).toString().padStart(5, '0')}`);
  }

  return candidates.find((candidate) => candidate in symptomIdToIndex) ?? null;
}

function displaySymptomId(symptomId: string) {
  const lookupKey = lookupSymptomKey(symptomId);
  if (lookupKey === null) {
    return symptomId;
  }
  return /^\d+$/.test(lookupKey)
    ? `SMTS${Number(lookupKey).toString().padStart(5, '0')}`
    : lookupKey;
}

function topSyndromes(syndromeProbabilities: number[], topK: number): SyndromePrediction[] {
  const limit = Math.min(topK, syndromeIndexToId.length);
  return topIndices(syndromeProbabilities, limit).map((index) => {
    const syndromeId = syndromeIndexToId[index];
    const metadata = syndromeMetadataById[syndromeId] ?? {};
    return {
      index,
      syndrome_id: syndromeId,
      label: metadata.label ?? syndromeId,
      english_name: metadata.english_name ?? metadata.label ?? syndromeId,
      chinese_name: metadata.chinese_name ?? syndromeId,
      description: metadata.description ?? '',
      confidence: syndromeProbabilities[index],
    };
  });
}

function recommendHerbs(
  conceptScores: number[],
  herbScores: number[],
  predSyndromeIdx: number,
  topK: number,
): HerbRecommendation[] {
  const conceptSimilarity = artifacts.herb_concept_matrix.map(
    (row) =>
      row.reduce((sum, value, index) => sum + value * conceptScores[index], 0) /
      Math.max(conceptScores.length, 1),
  );
  const prior = artifacts.syndrome_herb_prior[predSyndromeIdx];

  return herbIds
    .map((_, index) => index)
    .sort((a, b) => herbScores[b] - herbScores[a] || b - a)
    .slice(0, topK)
    .map((index) => {
      const herbId = herbIds[index];
      const metadata = herbMetadataById[herbId] ?? {};
      return {
        herb_id: herbId,
        label: metadata.label ?? herbId,
        english_name: metadata.english_name ?? metadata.label ?? herbId,
        chinese_name: metadata.chinese_name ?? herbId,
        description: metadata.description ?? '',
        target_concepts: metadata.target_concepts ?? [],
        score: sigmoidValue(herbScores[index]),
        concept_similarity: conceptSimilarity[index],
        syndrome_prior: prior[index],
        known_for_predicted_syndrome: prior[index] > 0,
      };
    });
}

function buildExplanation(
  knownSymptoms: string[],
  concepts: ConceptScore[],
  predSyndromeIdx: number,
  herbRecommendations: HerbRecommendation[],
  alpha: number,
): PredictionResponse['explanation'] {
  const matchingSymptoms = knownSymptoms.map((symptomId) => {
    const displayId = displaySymptomId(symptomId);
    return {
      id: displayId,
      label: symptomLabelById[displayId] ?? displayId,
    };
  });
  const topConcepts = concepts.slice().sort((a, b) => b.score - a.score).slice(0, 5);
  const syndromeId = syndromeIndexToId[predSyndromeIdx];
  const prior = artifacts.syndrome_herb_prior[predSyndromeIdx];
  const associatedIndices = prior
    .map((value, index) => (value > 0 ? index : -1))
    .filter((index) => index >= 0);
  const syndromeMetadata = syndromeMetadataById[syndromeId] ?? {};
  const associatedHerbs = associatedIndices.slice(0, 12).map((index) => {
    const herbId = herbIds[index];
    const metadata = herbMetadataById[herbId] ?? {};
    return {
      id: herbId,
      label: metadata.label ?? herbId,
      english_name: metadata.english_name ?? metadata.label ?? herbId,
      chinese_name: metadata.chinese_name ?? herbId,
    };
  });

  return {
    matching_symptoms: matchingSymptoms,
    concept_alignment: topConcepts,
    syndrome_herb_associations: {
      syndrome_id: syndromeId,
      label: syndromeMetadata.label ?? syndromeId,
      english_name: syndromeMetadata.english_name ?? syndromeMetadata.label ?? syndromeId,
      chinese_name: syndromeMetadata.chinese_name ?? syndromeId,
      associated_herbs: associatedHerbs,
      total_associated_herbs: associatedIndices.length,
    },
    herb_ranking: {
      formula: 'ranking = descending neural herb_head logit; score = sigmoid(neural herb_head logit)',
      alpha,
      items: herbRecommendations.map((herb) => ({
        herb_id: herb.herb_id,
        label: herb.label,
        english_name: herb.english_name,
        chinese_name: herb.chinese_name,
        description: herb.description,
        target_concepts: herb.target_concepts,
        concept_similarity: herb.concept_similarity,
        syndrome_prior: herb.syndrome_prior,
        score: herb.score,
        known_for_predicted_syndrome: herb.known_for_predicted_syndrome,
      })),
    },
  };
}

function topIndices(values: number[], limit: number) {
  return values
    .map((value, index) => ({ value, index }))
    .sort((a, b) => b.value - a.value || b.index - a.index)
    .slice(0, limit)
    .map((item) => item.index);
}

function metadataLabelMap(records: MetadataRecord[]) {
  return Object.fromEntries(records.map((record) => [record.id, record.label || record.id]));
}

function metadataRecordMap<T extends MetadataRecord>(records: T[]) {
  return Object.fromEntries(records.map((record) => [record.id, record]));
}

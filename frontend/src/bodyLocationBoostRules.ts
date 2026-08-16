import symptomBodyLocationPayload from './generated/symptom-body-location-map.json';

export type SymptomBodyLocationCategory =
  keyof typeof symptomBodyLocationPayload.categoryRegionIds;

export type SymptomBodyLocationMapping = {
  label: string;
  categories: SymptomBodyLocationCategory[];
  primaryCategory: SymptomBodyLocationCategory;
  bodyRegionIds: string[];
  matchedTerms: string[];
};

export type SymptomBodyLocationBoost = {
  symptomId: string;
  score: number;
  categories: SymptomBodyLocationCategory[];
  directRegionMatches: string[];
  categoryMatches: SymptomBodyLocationCategory[];
};

export const BODY_LOCATION_CATEGORY_REGION_IDS =
  symptomBodyLocationPayload.categoryRegionIds as Record<
    SymptomBodyLocationCategory,
    string[]
  >;

export const SYMPTOM_BODY_LOCATION_MAP =
  symptomBodyLocationPayload.mappings as Record<
    string,
    SymptomBodyLocationMapping
  >;

export function getBodyLocationMappingForSymptom(symptomId: string) {
  return SYMPTOM_BODY_LOCATION_MAP[normalizeSymptomId(symptomId)] ?? null;
}

export function getCategoriesForSelectedBodyRegions(selectedRegionIds: string[]) {
  const selected = new Set(selectedRegionIds);
  const categories = new Set<SymptomBodyLocationCategory>();

  for (const [category, regionIds] of Object.entries(
    BODY_LOCATION_CATEGORY_REGION_IDS,
  ) as Array<[SymptomBodyLocationCategory, string[]]>) {
    if (regionIds.some((regionId) => selected.has(regionId))) {
      categories.add(category);
    }
  }

  return [...categories];
}

export function getBodyLocationBoostForSymptom(
  symptomId: string,
  selectedRegionIds: string[],
): SymptomBodyLocationBoost {
  const normalizedSymptomId = normalizeSymptomId(symptomId);
  const mapping = SYMPTOM_BODY_LOCATION_MAP[normalizedSymptomId];

  if (!mapping || selectedRegionIds.length === 0) {
    return {
      symptomId: normalizedSymptomId,
      score: 0,
      categories: mapping?.categories ?? [],
      directRegionMatches: [],
      categoryMatches: [],
    };
  }

  const selected = new Set(selectedRegionIds);
  const selectedCategories = new Set(
    getCategoriesForSelectedBodyRegions(selectedRegionIds),
  );
  const directRegionMatches = mapping.bodyRegionIds.filter((regionId) =>
    selected.has(regionId),
  );
  const categoryMatches = mapping.categories.filter((category) =>
    selectedCategories.has(category),
  );

  return {
    symptomId: normalizedSymptomId,
    score: directRegionMatches.length * 2 + categoryMatches.length,
    categories: mapping.categories,
    directRegionMatches,
    categoryMatches,
  };
}

export function getBoostedSymptomIdsForBodyRegions(selectedRegionIds: string[]) {
  return Object.keys(SYMPTOM_BODY_LOCATION_MAP)
    .map((symptomId) =>
      getBodyLocationBoostForSymptom(symptomId, selectedRegionIds),
    )
    .filter((boost) => boost.score > 0)
    .sort((left, right) => right.score - left.score);
}

function normalizeSymptomId(symptomId: string) {
  const trimmed = symptomId.trim();
  const upper = trimmed.toUpperCase();

  if (upper.startsWith('SMTS')) {
    const numeric = upper.replace(/^SMTS/, '').replace(/^0+/, '') || '0';
    return `SMTS${Number(numeric).toString().padStart(5, '0')}`;
  }

  if (/^\d+$/.test(trimmed)) {
    return `SMTS${Number(trimmed).toString().padStart(5, '0')}`;
  }

  return trimmed;
}

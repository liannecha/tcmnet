import { getBodyLocationBoostForSymptom } from './bodyLocationBoostRules';
import {
  getSearchableSymptomPhrases,
  normalizeSymptomText,
} from './symptomSynonyms';
import type { PatientIntakeBasics, PatientIntakeSymptom } from './types';

export type SymptomSuggestionMatchSource =
  | 'exactLabel'
  | 'synonymCanonical'
  | 'synonymPhrase'
  | 'labelStartsWithQuery'
  | 'labelIncludesQuery'
  | 'tokenOverlap'
  | 'directBodyRegion'
  | 'bodyCategory'
  | 'severityContext';

export type SymptomSuggestion = {
  group: PatientIntakeSymptom;
  score: number;
  reasons: string[];
  matchSources: SymptomSuggestionMatchSource[];
};

export type SymptomSuggestionEngineInput = {
  intakeBasics: PatientIntakeBasics;
  selectedBodyRegionIds: string[];
  groupedSymptoms: PatientIntakeSymptom[];
  selectedGroupKeys: Set<string>;
  limit?: number;
  minScore?: number;
};

type MutableSuggestion = SymptomSuggestion & {
  reasonSet: Set<string>;
  sourceSet: Set<SymptomSuggestionMatchSource>;
};

const DEFAULT_LIMIT = 10;
const DEFAULT_MIN_SCORE = 32;
const DIRECT_BODY_REGION_BOOST = 18;
const BODY_CATEGORY_BOOST = 10;
const STOP_WORDS = new Set([
  'about',
  'after',
  'also',
  'and',
  'are',
  'but',
  'can',
  'cant',
  'feel',
  'feeling',
  'for',
  'from',
  'have',
  'having',
  'into',
  'like',
  'main',
  'more',
  'not',
  'now',
  'really',
  'since',
  'that',
  'the',
  'this',
  'with',
]);

const PATIENT_TERM_ALIASES: Record<string, string[]> = {
  ache: ['pain'],
  aches: ['pain'],
  aching: ['pain'],
  belly: ['abdomen', 'abdominal', 'stomach'],
  bloated: ['bloating', 'distension', 'fullness'],
  bloating: ['distension', 'fullness'],
  breathless: ['shortness', 'breath'],
  constipated: ['constipation', 'stool'],
  dizzy: ['dizziness', 'vertigo'],
  exhausted: ['fatigue', 'tiredness', 'weakness'],
  gassy: ['gas', 'distension', 'fullness'],
  hurt: ['pain'],
  hurting: ['pain'],
  hurts: ['pain'],
  nauseous: ['nausea'],
  poop: ['stool', 'bowel'],
  pooping: ['stool', 'bowel'],
  puking: ['vomiting'],
  queasy: ['nausea'],
  sleepy: ['sleepiness', 'fatigue'],
  stomach: ['abdomen', 'abdominal', 'epigastric'],
  throwing: ['vomiting'],
  tired: ['fatigue', 'tiredness', 'weakness'],
  tummy: ['abdomen', 'abdominal', 'stomach'],
  weak: ['weakness', 'fatigue'],
};

export function normalizeSuggestionText(text: string) {
  return normalizeSymptomText(text).replace(/\s+/g, ' ').trim();
}

export function getSymptomSuggestions({
  intakeBasics,
  selectedBodyRegionIds,
  groupedSymptoms,
  selectedGroupKeys,
  limit = DEFAULT_LIMIT,
  minScore = DEFAULT_MIN_SCORE,
}: SymptomSuggestionEngineInput): SymptomSuggestion[] {
  const concernQueries = getConcernQueries(intakeBasics.mainConcern);
  if (concernQueries.length === 0) {
    return [];
  }
  const query = uniqueTokens(concernQueries.map((concern) => concern.expanded).join(' '));

  const candidates = new Map<string, MutableSuggestion>();
  const queryTokens = meaningfulTokens(query);
  const availableGroups = groupedSymptoms.filter(
    (group) => !selectedGroupKeys.has(group.key),
  );

  for (const group of availableGroups) {
    const label = normalizeSuggestionText(group.label);

    for (const concern of concernQueries) {
      if (label === concern.normalized || label === concern.expanded) {
        addCandidate(candidates, group, 100, 'exactLabel', 'Exact match to one of your concerns.');
      } else {
        if (concern.expanded.length >= 2 && label.startsWith(concern.expanded)) {
          addCandidate(
            candidates,
            group,
            55,
            'labelStartsWithQuery',
            'Starts with one of the concerns you entered.',
          );
        }
        if (concern.expanded.length >= 3 && label.includes(concern.expanded)) {
          addCandidate(
            candidates,
            group,
            45,
            'labelIncludesQuery',
            'Contains one of the concerns you entered.',
          );
        }
      }
    }

    const sharedTokens = [...queryTokens].filter((token) =>
      meaningfulTokens(label).has(token),
    );
    if (sharedTokens.length > 0) {
      addCandidate(
        candidates,
        group,
        sharedTokens.length * 8,
        'tokenOverlap',
        `Shares wording with your concern: ${sharedTokens.slice(0, 3).join(', ')}.`,
      );
    }
  }

  for (const concern of concernQueries) {
    addSynonymCandidates(candidates, availableGroups, concern.expanded);
  }
  addBodyLocationBoosts(candidates, selectedBodyRegionIds);
  addSeverityContext(candidates, intakeBasics.severity);

  return [...candidates.values()]
    .map(({ reasonSet, sourceSet, ...suggestion }) => ({
      ...suggestion,
      score: Math.round(suggestion.score),
      reasons: [...reasonSet],
      matchSources: [...sourceSet],
    }))
    .filter((suggestion) => suggestion.score >= minScore)
    .sort(
      (left, right) =>
        right.score - left.score || left.group.label.localeCompare(right.group.label),
    )
    .slice(0, limit);
}

function getConcernQueries(mainConcern: string) {
  return mainConcern
    .split(',')
    .map((concern) => normalizeSuggestionText(concern))
    .filter(Boolean)
    .map((normalized) => ({
      normalized,
      expanded: expandPatientLanguage(normalized),
    }));
}

function addSynonymCandidates(
  candidates: Map<string, MutableSuggestion>,
  groups: PatientIntakeSymptom[],
  query: string,
) {
  const phrases = getSearchableSymptomPhrases();
  const matchingCanonicalLabels = new Map<string, { phrase: string; isCanonical: boolean }>();

  for (const phrase of phrases) {
    if (phrase.source === 'bodyLocation') {
      continue;
    }

    const normalizedPhrase = normalizeSuggestionText(phrase.normalizedPhrase);
    if (normalizedPhrase.length < 3) {
      continue;
    }

    const phraseTokens = meaningfulTokens(normalizedPhrase);
    const queryTokens = meaningfulTokens(query);
    const sharedPhraseTokens = [...phraseTokens].filter((token) => queryTokens.has(token));
    const phraseTokenCoverage =
      phraseTokens.size > 0 ? sharedPhraseTokens.length / phraseTokens.size : 0;
    const matchesPhrase =
      query === normalizedPhrase ||
      query.includes(normalizedPhrase) ||
      normalizedPhrase.includes(query) ||
      (phraseTokens.size > 1 && phraseTokenCoverage >= 0.75);

    if (!matchesPhrase) {
      continue;
    }

    const current = matchingCanonicalLabels.get(phrase.canonicalLabel);
    if (!current || phrase.source === 'canonical') {
      matchingCanonicalLabels.set(phrase.canonicalLabel, {
        phrase: phrase.phrase,
        isCanonical: phrase.source === 'canonical',
      });
    }
  }

  for (const [canonicalLabel, match] of matchingCanonicalLabels.entries()) {
    const normalizedCanonical = normalizeSuggestionText(canonicalLabel);
    for (const group of groups) {
      const label = normalizeSuggestionText(group.label);
      const matchesCanonical =
        label === normalizedCanonical ||
        label.includes(normalizedCanonical) ||
        normalizedCanonical.includes(label);

      if (!matchesCanonical) {
        continue;
      }

      addCandidate(
        candidates,
        group,
        match.isCanonical ? 80 : 70,
        match.isCanonical ? 'synonymCanonical' : 'synonymPhrase',
        match.isCanonical
          ? 'Matches a known symptom name.'
          : `Related to "${match.phrase}" in your concern.`,
      );
    }
  }
}

function addBodyLocationBoosts(
  candidates: Map<string, MutableSuggestion>,
  selectedBodyRegionIds: string[],
) {
  if (selectedBodyRegionIds.length === 0) {
    return;
  }

  for (const candidate of candidates.values()) {
    const boosts = candidate.group.ids.map((id) =>
      getBodyLocationBoostForSymptom(id, selectedBodyRegionIds),
    );
    const directMatches = boosts.reduce(
      (sum, boost) => sum + boost.directRegionMatches.length,
      0,
    );
    const categoryMatches = boosts.reduce(
      (sum, boost) => sum + boost.categoryMatches.length,
      0,
    );

    if (directMatches > 0) {
      candidate.score += DIRECT_BODY_REGION_BOOST;
      candidate.sourceSet.add('directBodyRegion');
      candidate.reasonSet.add('Matches a selected body area.');
    }

    if (categoryMatches > 0) {
      candidate.score += BODY_CATEGORY_BOOST;
      candidate.sourceSet.add('bodyCategory');
      candidate.reasonSet.add('Fits the selected body area.');
    }
  }
}

function addSeverityContext(
  candidates: Map<string, MutableSuggestion>,
  severity: number | null,
) {
  if (severity === null || severity < 7) {
    return;
  }

  const bonus = severity >= 9 ? 10 : 5;
  for (const candidate of candidates.values()) {
    candidate.score += bonus;
    candidate.sourceSet.add('severityContext');
    candidate.reasonSet.add('Prioritized because the concern is marked severe.');
  }
}

function addCandidate(
  candidates: Map<string, MutableSuggestion>,
  group: PatientIntakeSymptom,
  score: number,
  source: SymptomSuggestionMatchSource,
  reason: string,
) {
  const existing = candidates.get(group.key);
  if (existing) {
    existing.score += score;
    existing.sourceSet.add(source);
    existing.reasonSet.add(reason);
    return;
  }

  candidates.set(group.key, {
    group,
    score,
    reasons: [],
    matchSources: [],
    reasonSet: new Set([reason]),
    sourceSet: new Set([source]),
  });
}

function meaningfulTokens(text: string) {
  return new Set(
    normalizeSuggestionText(text)
      .split(' ')
      .filter((token) => token.length >= 3 && !STOP_WORDS.has(token)),
  );
}

function expandPatientLanguage(text: string) {
  const tokens = text.split(' ').filter(Boolean);
  const expandedTokens = [...tokens];

  for (const token of tokens) {
    const aliases = PATIENT_TERM_ALIASES[token] ?? [];
    for (const alias of aliases) {
      expandedTokens.push(alias);
    }
  }

  return [...new Set(expandedTokens)].join(' ');
}

function uniqueTokens(text: string) {
  return [...new Set(text.split(' ').filter(Boolean))].join(' ');
}

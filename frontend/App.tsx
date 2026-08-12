import { StatusBar } from 'expo-status-bar';
import { createElement, useEffect, useMemo, useState } from 'react';
import {
  ActivityIndicator,
  Pressable,
  SafeAreaView,
  ScrollView,
  StyleSheet,
  Text,
  TextInput,
  View,
  useWindowDimensions,
} from 'react-native';

import { fetchSymptoms, predict } from './src/api';
import type {
  ConceptScore,
  HerbRecommendation,
  MetadataRecord,
  PredictionResponse,
  SyndromePrediction,
} from './src/types';

const MAX_VISIBLE_SYMPTOMS = 10;
const MIN_SEARCH_LENGTH = 2;
const COMMON_SYMPTOM_LABELS = ['Fever', 'Cough', 'Dizziness', 'Fatigue', 'Abdominal Pain'];
const RELATED_SYMPTOM_TERMS: Record<string, string[]> = {
  fever: ['chills', 'cold', 'sweating', 'thirst', 'headache', 'cough', 'phlegm'],
  cough: ['phlegm', 'throat', 'asthma', 'wheezing', 'chest', 'fever', 'nasal'],
  dizziness: ['fatigue', 'headache', 'palpitation', 'tinnitus', 'vision', 'weakness'],
  fatigue: ['weakness', 'qi', 'dizziness', 'pale', 'appetite', 'loose stool'],
  abdominal: ['pain', 'distension', 'fullness', 'nausea', 'vomiting', 'stool', 'diarrhea'],
  pain: ['distension', 'cold', 'heat', 'numbness', 'swelling', 'stiffness'],
  phlegm: ['cough', 'asthma', 'chest', 'throat', 'sticky', 'yellow', 'dizziness'],
  cold: ['chills', 'aversion', 'pain', 'clear', 'pale', 'warmth'],
  heat: ['fever', 'thirst', 'yellow', 'red', 'bitter', 'irritability'],
  diarrhea: ['abdominal', 'stool', 'cold', 'dampness', 'fatigue'],
};

function titleCaseLabel(label: string) {
  return label.replace(/\p{L}[\p{L}'-]*/gu, (word) => {
    const [first = '', ...rest] = Array.from(word);
    return `${first.toLocaleUpperCase()}${rest.join('').toLocaleLowerCase()}`;
  });
}

function symptomDisplayLabel(symptom: MetadataRecord) {
  return titleCaseLabel(symptom.label || symptom.id);
}

function normalizedLabel(label: string) {
  return label.toLocaleLowerCase();
}

function uniqueGroups(groups: SymptomGroup[]) {
  const seen = new Set<string>();
  return groups.filter((group) => {
    if (seen.has(group.key)) {
      return false;
    }
    seen.add(group.key);
    return true;
  });
}

function englishDisplayName(record: { english_name?: string; label: string }) {
  return titleCaseLabel(record.english_name || record.label);
}

function bilingualDisplayName(record: {
  english_name?: string;
  chinese_name?: string;
  label: string;
}) {
  const englishName = englishDisplayName(record);
  return record.chinese_name ? `${englishName} / ${record.chinese_name}` : englishName;
}

type ConceptAxis = {
  key: string;
  sourceKey?: string;
  label: string;
  description: string;
};

const EIGHT_PRINCIPLE_AXES: ConceptAxis[] = [
  {
    key: 'yin',
    label: 'Yin',
    description: 'Cooling, moistening, and nourishing qualities in TCM pattern language.',
  },
  {
    key: 'yang',
    label: 'Yang',
    description: 'Warming, activating, and transforming qualities in TCM pattern language.',
  },
  {
    key: 'internal',
    label: 'Internal',
    description: 'A pattern understood as deeper or inside the body rather than surface-level.',
  },
  {
    key: 'external',
    label: 'External',
    description: 'A pattern understood as affecting the surface or protective layer of the body.',
  },
  {
    key: 'cold',
    label: 'Cold',
    description: 'Cold signs may include chilliness, slow movement, cold pain, or pale features.',
  },
  {
    key: 'heat',
    sourceKey: 'hot',
    label: 'Heat',
    description: 'Heat signs may include feverishness, thirst, redness, irritability, or yellow secretions.',
  },
  {
    key: 'deficiency',
    label: 'Deficiency',
    description: 'A lack of nourishment, warmth, fluids, blood, or functional strength.',
  },
  {
    key: 'excess',
    label: 'Excess',
    description: 'A stronger obstructive pattern involving accumulation, blockage, or overactivity.',
  },
];

const FIVE_ELEMENT_AXES: ConceptAxis[] = [
  {
    key: 'wood',
    sourceKey: 'Wood',
    label: 'Wood',
    description: 'Associated with movement, growth, the Liver system, and smooth flow.',
  },
  {
    key: 'water',
    sourceKey: 'Water',
    label: 'Water',
    description: 'Associated with storage, fluids, the Kidney system, and foundational reserves.',
  },
  {
    key: 'fire',
    sourceKey: 'Fire',
    label: 'Fire',
    description: 'Associated with warmth, activity, circulation, spirit, and the Heart system.',
  },
  {
    key: 'metal',
    sourceKey: 'Metal',
    label: 'Metal',
    description: 'Associated with breathing, boundaries, the Lung system, and protective qi.',
  },
  {
    key: 'earth',
    sourceKey: 'Earth',
    label: 'Earth',
    description: 'Associated with digestion, nourishment, transformation, and the Spleen/Stomach system.',
  },
];

type SymptomGroup = {
  key: string;
  label: string;
  ids: string[];
};

type AssessmentStep = 1 | 2 | 3 | 4;

export default function App() {
  const { width } = useWindowDimensions();
  const isWide = width >= 920;
  const [started, setStarted] = useState(false);

  return (
    <SafeAreaView style={styles.safeArea}>
      <StatusBar style="dark" />
      <Header started={started} onStart={() => setStarted(true)} onHome={() => setStarted(false)} />
      {started ? <AssessmentPage isWide={isWide} /> : <LandingPage onStart={() => setStarted(true)} />}
    </SafeAreaView>
  );
}

function Header({
  started,
  onStart,
  onHome,
}: {
  started: boolean;
  onStart: () => void;
  onHome: () => void;
}) {
  return (
    <View style={styles.topHeader}>
      <Pressable onPress={onHome} style={({ pressed }) => [styles.brand, pressed && styles.pressed]}>
        <Text style={styles.brandText}>TCMNet</Text>
      </Pressable>
      <View style={styles.navActions}>
        {!started ? (
          <Pressable onPress={onStart} style={({ pressed }) => [styles.navButton, pressed && styles.pressed]}>
            <Text style={styles.navButtonText}>Get started</Text>
          </Pressable>
        ) : null}
      </View>
    </View>
  );
}

function LandingPage({ onStart }: { onStart: () => void }) {
  return (
    <ScrollView contentContainerStyle={styles.landingPage}>
      <View style={styles.hero}>
        <Text style={styles.eyebrow}>DEEP LEARNING FOR TRADITIONAL MEDICINE</Text>
        <Text style={styles.heroTitle}>TCMNet</Text>
        <Text style={styles.tagline}>Make pattern recognition clearer.</Text>
        <Text style={styles.heroCopy}>
          TCMNet turns selected symptoms into TCM syndrome predictions, herb
          recommendations, and a plain-language view of the pattern signals behind them.
        </Text>
        <Pressable onPress={onStart} style={({ pressed }) => [styles.heroButton, pressed && styles.pressed]}>
          <Text style={styles.heroButtonText}>Get started</Text>
        </Pressable>
      </View>

      <View style={styles.aboutSection}>
        <Text style={styles.aboutEyebrow}>ABOUT TCMNET</Text>
        <Text style={styles.aboutTitle}>Built to make TCM prediction easier to inspect.</Text>
        <Text style={styles.aboutIntro}>
          TCMNet helps organize symptoms into traditional pattern language, then shows
          how those patterns connect to possible syndromes and herbs.
        </Text>
        <View style={styles.aboutGrid}>
          {ABOUT_CARDS.map((card) => (
            <AboutCard key={card.title} card={card} />
          ))}
        </View>
      </View>
    </ScrollView>
  );
}

function AboutCard({ card }: { card: (typeof ABOUT_CARDS)[number] }) {
  const [isHovered, setIsHovered] = useState(false);

  return (
    <View
      style={[styles.aboutCard, isHovered && styles.aboutCardHovered]}
      onPointerEnter={() => setIsHovered(true)}
      onPointerLeave={() => setIsHovered(false)}
    >
      <Text style={styles.aboutCardTitle}>{card.title}</Text>
      <Text style={styles.aboutCardText}>{card.text}</Text>
      {isHovered ? <Text style={styles.aboutCardDetail}>{card.detail}</Text> : null}
    </View>
  );
}

const ABOUT_CARDS = [
  {
    title: 'Symptom Input',
    text: 'Start with observed symptoms and search by readable English labels.',
    detail:
      'The interface groups similar translated labels so symptom selection stays readable.',
  },
  {
    title: 'Concept Signals',
    text: 'TCMNet highlights pattern signals such as internal, heat, yin, and organ groups.',
    detail:
      'These signals make the result easier to understand than a syndrome name alone.',
  },
  {
    title: 'Syndrome Prediction',
    text: 'TCMNet ranks likely syndromes from the selected symptom pattern.',
    detail:
      'The app shows the best match first, then keeps a few alternatives visible for comparison.',
  },
  {
    title: 'Herb Ranking',
    text: 'Herb recommendations are shown with plain-language context.',
    detail:
      'Each herb includes its Chinese name, typical TCM category, and targeted concepts.',
  },
  {
    title: 'Future Direction',
    text: 'Next steps include stronger labels, better grouping, richer explanations, and validation.',
    detail:
      'Future work can improve metadata, validation, and clinical review workflows.',
  },
  {
    title: 'Final Report',
    text: 'A final report and findings link will live here once the project writeup is complete.',
    detail:
      'Use this space for the CS229 report, results, limitations, and supporting analysis.',
  },
];

function AssessmentPage({ isWide }: { isWide: boolean }) {
  const [symptoms, setSymptoms] = useState<MetadataRecord[]>([]);
  const [selectedSymptoms, setSelectedSymptoms] = useState<SymptomGroup[]>([]);
  const [query, setQuery] = useState('');
  const [loadingSymptoms, setLoadingSymptoms] = useState(true);
  const [predicting, setPredicting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [prediction, setPrediction] = useState<PredictionResponse | null>(null);
  const [assessmentStep, setAssessmentStep] = useState<AssessmentStep>(1);

  useEffect(() => {
    let mounted = true;

    async function loadSymptoms() {
      try {
        setLoadingSymptoms(true);
        setError(null);
        const records = await fetchSymptoms();
        if (mounted) {
          setSymptoms(records);
        }
      } catch (err) {
        if (mounted) {
          setError(err instanceof Error ? err.message : 'Unable to load symptoms.');
        }
      } finally {
        if (mounted) {
          setLoadingSymptoms(false);
        }
      }
    }

    loadSymptoms();
    return () => {
      mounted = false;
    };
  }, []);

  const groupedSymptoms = useMemo(() => {
    const groups = new Map<string, SymptomGroup>();

    for (const symptom of symptoms) {
      const label = symptomDisplayLabel(symptom);
      const key = label.toLocaleLowerCase();
      const existing = groups.get(key);

      if (existing) {
        existing.ids.push(symptom.id);
      } else {
        groups.set(key, { key, label, ids: [symptom.id] });
      }
    }

    return Array.from(groups.values());
  }, [symptoms]);

  const selectedGroupKeys = useMemo(
    () => new Set(selectedSymptoms.map((symptom) => symptom.key)),
    [selectedSymptoms],
  );

  const searchResults = useMemo(() => {
    const normalizedQuery = query.trim().toLowerCase();
    if (normalizedQuery.length < MIN_SEARCH_LENGTH) {
      return [];
    }

    return groupedSymptoms
      .filter((group) => !selectedGroupKeys.has(group.key))
      .map((group) => {
        const label = group.label.toLowerCase();
        let rank = Number.POSITIVE_INFINITY;

        if (label === normalizedQuery) {
          rank = 0;
        } else if (label.startsWith(normalizedQuery)) {
          rank = 1;
        } else if (label.includes(normalizedQuery)) {
          rank = 2;
        } else if (group.ids.some((id) => id.toLowerCase().includes(normalizedQuery))) {
          rank = 3;
        }

        return { group, rank, label };
      })
      .filter((item) => Number.isFinite(item.rank))
      .sort((a, b) => a.rank - b.rank || a.label.localeCompare(b.label))
      .map((item) => item.group)
      .slice(0, MAX_VISIBLE_SYMPTOMS);
  }, [groupedSymptoms, query, selectedGroupKeys]);

  const trySymptoms = useMemo(() => {
    const availableGroups = groupedSymptoms.filter((group) => !selectedGroupKeys.has(group.key));
    const defaultSuggestions = COMMON_SYMPTOM_LABELS.map((label) =>
      groupedSymptoms.find((group) => group.key === label.toLocaleLowerCase()),
    ).filter((group): group is SymptomGroup => Boolean(group));

    const selectedText = selectedSymptoms
      .map((symptom) => normalizedLabel(symptom.label))
      .join(' ');
    const relatedTerms = Object.entries(RELATED_SYMPTOM_TERMS)
      .filter(([term]) => selectedText.includes(term))
      .flatMap(([, terms]) => terms);
    const relatedSuggestions = relatedTerms.flatMap((term) =>
      availableGroups
        .filter((group) => normalizedLabel(group.label).includes(term))
        .slice(0, 3),
    );
    const fallbackSuggestions = availableGroups.filter((group) =>
      COMMON_SYMPTOM_LABELS.some((label) =>
        normalizedLabel(group.label).includes(label.toLocaleLowerCase()),
      ),
    );

    return uniqueGroups([
      ...relatedSuggestions,
      ...defaultSuggestions,
      ...fallbackSuggestions,
      ...availableGroups,
    ]).slice(0, 6);
  }, [groupedSymptoms, selectedGroupKeys, selectedSymptoms]);

  function addSymptom(symptom: SymptomGroup) {
    setSelectedSymptoms((current) =>
      current.some((item) => item.key === symptom.key) ? current : [...current, symptom],
    );
    setQuery('');
    setPrediction(null);
    setAssessmentStep(1);
  }

  function removeSymptom(symptomKey: string) {
    setSelectedSymptoms((current) => current.filter((symptom) => symptom.key !== symptomKey));
    setPrediction(null);
    setAssessmentStep(1);
  }

  function clearSymptoms() {
    setSelectedSymptoms([]);
    setPrediction(null);
    setAssessmentStep(1);
    setError(null);
  }

  function startOver() {
    setSelectedSymptoms([]);
    setQuery('');
    setPrediction(null);
    setAssessmentStep(1);
    setError(null);
  }

  function selectFirstSearchResult() {
    if (searchResults.length > 0) {
      addSymptom(searchResults[0]);
    }
  }

  async function runPrediction() {
    if (selectedSymptoms.length === 0) {
      setError('Select at least one symptom before predicting.');
      return;
    }

    try {
      setPredicting(true);
      setError(null);
      const result = await predict({
        symptom_ids: selectedSymptoms.flatMap((symptom) => symptom.ids),
        top_syndromes: 5,
        top_herbs: 5,
      });
      setPrediction(result);
      setAssessmentStep(2);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Prediction request failed.');
    } finally {
      setPredicting(false);
    }
  }

  return (
    <ScrollView contentContainerStyle={styles.page}>
      <View style={styles.pageInner}>
        <HorizontalStepper activeStep={prediction ? assessmentStep : 1} />

        {!prediction ? (
          <View style={styles.initialStage}>
            <SymptomPanel
              title="What symptoms are present?"
              helperText="Search by symptom label, then add the symptoms observed in the case."
              query={query}
              setQuery={setQuery}
              searchResults={searchResults}
              trySymptoms={trySymptoms}
              selectedSymptoms={selectedSymptoms}
              loadingSymptoms={loadingSymptoms}
              predicting={predicting}
              error={error}
              isWide={isWide}
              onAddSymptom={addSymptom}
              onRemoveSymptom={removeSymptom}
              onClear={clearSymptoms}
              onPredict={runPrediction}
              onSelectFirstResult={selectFirstSearchResult}
              primaryLabel="Predict syndrome and herbs"
              centered
            />
          </View>
        ) : (
          <View style={[styles.assessmentGrid, !isWide && styles.assessmentGridStacked]}>
            <View style={styles.leftPanel}>
              <SymptomPanel
                title="Symptoms"
                helperText="Adjust the symptom set and run the prediction again."
                query={query}
                setQuery={setQuery}
                searchResults={searchResults}
                trySymptoms={trySymptoms}
                selectedSymptoms={selectedSymptoms}
                loadingSymptoms={loadingSymptoms}
                predicting={predicting}
                error={error}
                isWide={false}
                onAddSymptom={addSymptom}
                onRemoveSymptom={removeSymptom}
                onClear={clearSymptoms}
                onPredict={runPrediction}
                onSelectFirstResult={selectFirstSearchResult}
                primaryLabel="Predict again"
                compact
              />
            </View>
            <View style={styles.rightPanel}>
              <ResultStage
                prediction={prediction}
                selectedSymptoms={selectedSymptoms}
                step={assessmentStep}
                onStepChange={setAssessmentStep}
                onStartOver={startOver}
              />
            </View>
          </View>
        )}
      </View>
    </ScrollView>
  );
}

function HorizontalStepper({ activeStep }: { activeStep: number }) {
  const steps = [
    { id: 1, label: 'Input symptoms' },
    { id: 2, label: 'Syndrome prediction' },
    { id: 3, label: 'Herb recommendation' },
    { id: 4, label: 'Final review' },
  ];

  return (
    <View style={styles.horizontalStepper}>
      {steps.map((step) => {
        const isActive = step.id === activeStep;
        const isComplete = step.id < activeStep;
        return (
          <View key={step.id} style={styles.stepRow}>
            <View
              style={[
                styles.stepDot,
                isActive && styles.stepDotActive,
                isComplete && styles.stepDotComplete,
              ]}
            >
              <Text
                style={[
                  styles.stepDotText,
                  (isActive || isComplete) && styles.stepDotTextActive,
                ]}
              >
                {step.id}
              </Text>
            </View>
            <Text
              style={[
                styles.stepLabel,
                isActive && styles.stepLabelActive,
                isComplete && styles.stepLabelComplete,
              ]}
            >
              {step.label}
            </Text>
          </View>
        );
      })}
    </View>
  );
}

type SymptomPanelProps = {
  title: string;
  helperText: string;
  query: string;
  setQuery: (value: string) => void;
  searchResults: SymptomGroup[];
  trySymptoms: SymptomGroup[];
  selectedSymptoms: SymptomGroup[];
  loadingSymptoms: boolean;
  predicting: boolean;
  error: string | null;
  isWide: boolean;
  onAddSymptom: (symptom: SymptomGroup) => void;
  onRemoveSymptom: (symptomKey: string) => void;
  onClear: () => void;
  onPredict: () => void;
  onSelectFirstResult: () => void;
  primaryLabel: string;
  compact?: boolean;
  centered?: boolean;
};

function SymptomPanel({
  title,
  helperText,
  query,
  setQuery,
  searchResults,
  trySymptoms,
  selectedSymptoms,
  loadingSymptoms,
  predicting,
  error,
  isWide,
  onAddSymptom,
  onRemoveSymptom,
  onClear,
  onPredict,
  onSelectFirstResult,
  primaryLabel,
  centered,
}: SymptomPanelProps) {
  const trimmedQuery = query.trim();
  const shouldShowResults = trimmedQuery.length >= MIN_SEARCH_LENGTH;
  const splitLayout = Boolean(centered && isWide);
  const predictDisabled = predicting || selectedSymptoms.length === 0;
  const [showDisabledTip, setShowDisabledTip] = useState(false);

  function handleSearchKeyPress(event: { nativeEvent?: { key?: string } }) {
    const key = event.nativeEvent?.key;
    if (key === 'Enter') {
      onSelectFirstResult();
    }
    if (key === 'Escape') {
      setQuery('');
    }
  }

  return (
    <View style={[styles.panel, centered && styles.centerPanel]}>
      <View style={styles.panelHeader}>
        <Text style={centered ? styles.centerHeading : styles.panelTitle}>{title}</Text>
        <Text style={styles.panelHelp}>{helperText}</Text>
      </View>

      <View style={[styles.symptomComposer, splitLayout && styles.symptomComposerSplit]}>
        <View style={styles.searchPane}>
          <Text style={styles.sectionSubtitle}>Find symptoms</Text>
          <View style={styles.searchBoxWrap}>
            <TextInput
              value={query}
              onChangeText={setQuery}
              onKeyPress={handleSearchKeyPress}
              placeholder="Search symptom label"
              autoCapitalize="none"
              autoCorrect={false}
              style={styles.searchInput}
              placeholderTextColor={colors.muted}
            />

            {shouldShowResults ? (
              <View style={styles.autocompleteMenu}>
                {loadingSymptoms ? (
                  <View style={styles.autocompleteState}>
                    <ActivityIndicator color={colors.deepGreen} />
                    <Text style={styles.mutedText}>Loading symptoms...</Text>
                  </View>
                ) : searchResults.length === 0 ? (
                  <View style={styles.autocompleteState}>
                    <Text style={styles.mutedText}>No matching symptoms found.</Text>
                  </View>
                ) : (
                  searchResults.map((symptom) => (
                    <Pressable
                      key={symptom.key}
                      onPress={() => onAddSymptom(symptom)}
                      style={({ pressed }) => [styles.symptomRow, pressed && styles.pressed]}
                    >
                      <Text style={styles.symptomLabel} numberOfLines={2}>
                        {symptom.label}
                      </Text>
                      {symptom.ids.length > 1 ? (
                        <Text style={styles.groupHint}>
                          Includes {symptom.ids.length} related symptom entries
                        </Text>
                      ) : null}
                    </Pressable>
                  ))
                )}
              </View>
            ) : null}
          </View>

          {!shouldShowResults ? (
            <View style={styles.quickAddPanel}>
              <Text style={styles.quickAddTitle}>Try These Symptoms</Text>
              <View style={styles.quickAddGrid}>
                {trySymptoms.map((symptom) => (
                  <Pressable
                    key={`quick-${symptom.key}`}
                    onPress={() => onAddSymptom(symptom)}
                    style={({ pressed }) => [styles.quickAddChip, pressed && styles.pressed]}
                  >
                    <Text style={styles.quickAddText}>+ {symptom.label}</Text>
                  </Pressable>
                ))}
              </View>
              <Text style={styles.mutedText}>
                These suggestions update as symptoms are selected.
              </Text>
            </View>
          ) : null}
        </View>

        <View style={[styles.selectedPane, splitLayout && styles.selectedPaneSide]}>
          <View style={styles.selectedHeader}>
            <Text style={styles.selectedTitle}>Selected Symptoms</Text>
            <Text style={styles.countText}>{selectedSymptoms.length}</Text>
          </View>

          {selectedSymptoms.length === 0 ? (
            <View style={styles.emptySelectedState}>
              <Text style={styles.mutedText}>No symptoms selected yet.</Text>
            </View>
          ) : (
            <View style={styles.selectedList}>
              {selectedSymptoms.map((symptom) => (
                <Pressable
                  key={symptom.key}
                  style={({ pressed }) => [styles.selectedTag, pressed && styles.pressed]}
                  onPress={() => onRemoveSymptom(symptom.key)}
                >
                  <Text style={styles.selectedTagText} numberOfLines={1}>
                    {symptom.label}
                  </Text>
                  <Text style={styles.selectedTagRemove}>×</Text>
                </Pressable>
              ))}
            </View>
          )}

          {error ? <Text style={styles.errorText}>{error}</Text> : null}

          <View style={styles.buttonArea}>
            {showDisabledTip && predictDisabled ? (
              <View style={styles.tooltip}>
                <Text style={styles.tooltipText}>Select at least 1 symptom to run prediction</Text>
              </View>
            ) : null}
            <View style={styles.buttonRow}>
              <Pressable
                onPress={onPredict}
                disabled={predictDisabled}
                onHoverIn={() => setShowDisabledTip(true)}
                onHoverOut={() => setShowDisabledTip(false)}
                accessibilityHint={
                  predictDisabled ? 'Select at least 1 symptom to run prediction' : undefined
                }
                style={({ pressed }) => [
                  styles.primaryButton,
                  predictDisabled && styles.disabledButton,
                  pressed && styles.pressed,
                ]}
              >
                {predicting ? (
                  <ActivityIndicator color="#fff" />
                ) : (
                  <Text
                    style={[
                      styles.primaryButtonText,
                      predictDisabled && styles.disabledButtonText,
                    ]}
                    numberOfLines={2}
                  >
                    {primaryLabel}
                  </Text>
                )}
              </Pressable>
              {selectedSymptoms.length > 0 ? (
                <Pressable
                  onPress={onClear}
                  style={({ pressed }) => [styles.secondaryButton, pressed && styles.pressed]}
                >
                  <Text style={styles.secondaryButtonText}>Clear all</Text>
                </Pressable>
              ) : null}
            </View>
          </View>
        </View>
      </View>
    </View>
  );
}

function ResultStage({
  prediction,
  selectedSymptoms,
  step,
  onStepChange,
  onStartOver,
}: {
  prediction: PredictionResponse;
  selectedSymptoms: SymptomGroup[];
  step: AssessmentStep;
  onStepChange: (step: AssessmentStep) => void;
  onStartOver: () => void;
}) {
  if (step === 2) {
    return <SyndromeResults prediction={prediction} onNext={() => onStepChange(3)} />;
  }

  if (step === 3) {
    return (
      <HerbResults
        prediction={prediction}
        onBack={() => onStepChange(2)}
        onNext={() => onStepChange(4)}
      />
    );
  }

  return (
    <FinalReview
      prediction={prediction}
      selectedSymptoms={selectedSymptoms}
      onBack={() => onStepChange(3)}
      onStartOver={onStartOver}
    />
  );
}

function SyndromeResults({
  prediction,
  onNext,
}: {
  prediction: PredictionResponse;
  onNext: () => void;
}) {
  const [topSyndrome, ...otherSyndromes] = prediction.syndromes;
  const visibleOtherSyndromes = otherSyndromes.slice(0, 3);

  return (
    <View style={styles.resultsStack}>
      <View style={styles.panel}>
        <Text style={styles.panelTitle}>Best Syndrome Match</Text>
        {topSyndrome ? <TopSyndrome syndrome={topSyndrome} /> : null}
      </View>

      <View style={styles.panel}>
        <Text style={styles.panelTitle}>Other Possible Syndromes</Text>
        {visibleOtherSyndromes.map((syndrome) => (
          <SyndromeRow key={`${syndrome.syndrome_id}-${syndrome.index}`} syndrome={syndrome} />
        ))}
      </View>

      <SyndromeExplanationSummary prediction={prediction} />

      <StepNavigation primaryLabel="Next: Herbs" onPrimary={onNext} />
    </View>
  );
}

function HerbResults({
  prediction,
  onBack,
  onNext,
}: {
  prediction: PredictionResponse;
  onBack: () => void;
  onNext: () => void;
}) {
  return (
    <View style={styles.resultsStack}>
      <View style={styles.panel}>
        <Text style={styles.panelTitle}>Herb Recommendations</Text>
        {prediction.herbs.map((herb) => (
          <HerbRow key={herb.herb_id} herb={herb} />
        ))}
      </View>

      <StepNavigation
        secondaryLabel="Back"
        primaryLabel="Next: Review"
        onSecondary={onBack}
        onPrimary={onNext}
      />
    </View>
  );
}

function FinalReview({
  prediction,
  selectedSymptoms,
  onBack,
  onStartOver,
}: {
  prediction: PredictionResponse;
  selectedSymptoms: SymptomGroup[];
  onBack: () => void;
  onStartOver: () => void;
}) {
  const topSyndrome = prediction.syndromes[0];
  const topHerbs = prediction.herbs.slice(0, 3);

  return (
    <View style={styles.resultsStack}>
      <View style={styles.panel}>
        <Text style={styles.panelTitle}>Final Review</Text>
        <Text style={styles.bodyText}>
          A concise summary of the selected symptoms, best syndrome match, key pattern
          signals, and top herb recommendations.
        </Text>

        <Text style={styles.sectionSubtitle}>Selected Symptoms</Text>
        <View style={styles.chips}>
          {selectedSymptoms.map((symptom) => (
            <View key={symptom.key} style={styles.subtleChip}>
              <Text style={styles.subtleChipText}>{symptom.label}</Text>
            </View>
          ))}
        </View>

        <Text style={styles.sectionSubtitle}>Top Syndrome</Text>
        {topSyndrome ? <TopSyndrome syndrome={topSyndrome} /> : null}

        <KeySignalSummary concepts={prediction.concepts} />

        <Text style={styles.sectionSubtitle}>Top Herb Recommendations</Text>
        {topHerbs.map((herb) => (
          <HerbRow key={`review-${herb.herb_id}`} herb={herb} />
        ))}
      </View>

      <StepNavigation
        secondaryLabel="Back"
        primaryLabel="Start over"
        onSecondary={onBack}
        onPrimary={onStartOver}
      />
    </View>
  );
}

function StepNavigation({
  primaryLabel,
  secondaryLabel,
  onPrimary,
  onSecondary,
}: {
  primaryLabel?: string;
  secondaryLabel?: string;
  onPrimary?: () => void;
  onSecondary?: () => void;
}) {
  return (
    <View style={styles.stepNavigation}>
      {secondaryLabel && onSecondary ? (
        <Pressable
          onPress={onSecondary}
          style={({ pressed }) => [styles.stepSecondaryButton, pressed && styles.pressed]}
        >
          <Text style={styles.stepSecondaryButtonText}>{secondaryLabel}</Text>
        </Pressable>
      ) : null}
      {primaryLabel && onPrimary ? (
        <Pressable
          onPress={onPrimary}
          style={({ pressed }) => [styles.stepPrimaryButton, pressed && styles.pressed]}
        >
          <Text style={styles.stepPrimaryButtonText}>{primaryLabel}</Text>
        </Pressable>
      ) : null}
    </View>
  );
}

function TopSyndrome({ syndrome }: { syndrome: SyndromePrediction }) {
  return (
    <View style={styles.topSyndromeBox}>
      <View style={styles.resultText}>
        <Text style={styles.topSyndromeLabel}>{bilingualDisplayName(syndrome)}</Text>
        <Text style={styles.bodyText}>{syndrome.description}</Text>
      </View>
    </View>
  );
}

function SyndromeRow({ syndrome }: { syndrome: SyndromePrediction }) {
  return (
    <View style={styles.compactRow}>
      <View style={styles.resultText}>
        <Text style={styles.resultLabel}>
          {bilingualDisplayName(syndrome)}
        </Text>
        <Text style={styles.finePrint}>{syndrome.description}</Text>
      </View>
    </View>
  );
}

function HerbRow({ herb }: { herb: HerbRecommendation }) {
  return (
    <View style={styles.herbRecommendationRow}>
      <View style={styles.resultText}>
        <Text style={styles.resultLabel}>
          {bilingualDisplayName(herb)}
        </Text>
        <Text style={styles.finePrint}>{herb.description}</Text>
        {herb.target_concepts.length > 0 ? (
          <View style={styles.targetConceptList}>
            <Text style={styles.targetConceptLead}>Targets</Text>
            {herb.target_concepts.map((concept) => (
              <View key={`${herb.herb_id}-${concept}`} style={styles.targetConceptChip}>
                <Text style={styles.targetConceptText}>{titleCaseLabel(concept)}</Text>
              </View>
            ))}
          </View>
        ) : null}
      </View>
    </View>
  );
}

function SyndromeExplanationSummary({ prediction }: { prediction: PredictionResponse }) {
  return (
    <View style={styles.panel}>
      <Text style={styles.panelTitle}>Explanation Summary</Text>

      <View style={styles.whyCard}>
        <Text style={styles.whyTitle}>Why this prediction?</Text>
        <Text style={styles.bodyText}>
          TCMNet compares the selected symptoms with learned TCM pattern signals, then
          ranks syndrome matches that best fit those signals. The diagrams below show the
          pattern directions that most shaped this result.
        </Text>
      </View>

      <Text style={styles.sectionSubtitle}>Eight Principle Signals</Text>
      <ConceptRadarChart concepts={prediction.concepts} axes={EIGHT_PRINCIPLE_AXES} />

      <Text style={styles.sectionSubtitle}>Five Element Signals</Text>
      <ConceptRadarChart concepts={prediction.concepts} axes={FIVE_ELEMENT_AXES} />
    </View>
  );
}

function KeySignalSummary({ concepts }: { concepts: ConceptScore[] }) {
  const topConcepts = useMemo(
    () =>
      [...concepts]
        .filter((concept) => concept.id.toLowerCase() !== 'reproductive')
        .sort((a, b) => b.score - a.score)
        .slice(0, 5),
    [concepts],
  );

  return (
    <>
      <Text style={styles.sectionSubtitle}>Key Pattern Signals</Text>
      <View style={styles.chips}>
        {topConcepts.map((concept) => (
          <View key={`signal-${concept.id}`} style={styles.subtleChip}>
            <Text style={styles.subtleChipText}>{titleCaseLabel(concept.label)}</Text>
          </View>
        ))}
      </View>
    </>
  );
}

function ConceptRadarChart({
  concepts,
  axes,
}: {
  concepts: ConceptScore[];
  axes: ConceptAxis[];
}) {
  const [hoveredAxis, setHoveredAxis] = useState<ConceptAxis | null>(null);
  const scoreByKey = useMemo(() => {
    const scores = new Map<string, number>();
    for (const concept of concepts) {
      scores.set(concept.id.toLowerCase(), concept.score);
      scores.set(concept.label.toLowerCase(), concept.score);
    }
    return scores;
  }, [concepts]);

  const size = 280;
  const center = size / 2;
  const radius = 96;
  const gridLevels = [0.25, 0.5, 0.75, 1];
  const chartPoints = axes.map((axis, index) => {
    const angle = -Math.PI / 2 + (2 * Math.PI * index) / axes.length;
    const score = Math.max(
      0,
      Math.min(
        scoreByKey.get((axis.sourceKey || axis.key).toLowerCase()) ?? 0,
        1,
      ),
    );
    return {
      axis,
      score,
      angle,
      x: center + Math.cos(angle) * radius * score,
      y: center + Math.sin(angle) * radius * score,
      labelX: center + Math.cos(angle) * (radius + 30),
      labelY: center + Math.sin(angle) * (radius + 30),
      axisX: center + Math.cos(angle) * radius,
      axisY: center + Math.sin(angle) * radius,
    };
  });
  const polygonPoints = chartPoints.map((point) => `${point.x},${point.y}`).join(' ');

  return (
    <View style={styles.radarCard}>
      {createElement(
        'svg',
        {
          width: '100%',
          height: size,
          viewBox: `0 0 ${size} ${size}`,
          role: 'img',
          'aria-label': 'Concept radar diagram',
        },
        [
          ...gridLevels.map((level) =>
            createElement('polygon', {
              key: `grid-${level}`,
              points: axes
                .map((_, index) => {
                  const angle = -Math.PI / 2 + (2 * Math.PI * index) / axes.length;
                  return `${center + Math.cos(angle) * radius * level},${
                    center + Math.sin(angle) * radius * level
                  }`;
                })
                .join(' '),
              fill: 'none',
              stroke: '#1b3d2e',
              strokeOpacity: 0.28,
              strokeWidth: 1,
            }),
          ),
          ...chartPoints.map((point) =>
            createElement('line', {
              key: `axis-${point.axis.key}`,
              x1: center,
              y1: center,
              x2: point.axisX,
              y2: point.axisY,
              stroke: '#1b3d2e',
              strokeOpacity: 0.35,
              strokeWidth: 1,
            }),
          ),
          createElement('polygon', {
            key: 'signal',
            points: polygonPoints,
            fill: '#6BBF8A',
            fillOpacity: 0.48,
            stroke: '#2E7D5C',
            strokeWidth: 2,
          }),
          ...chartPoints.map((point) =>
            createElement('circle', {
              key: `point-${point.axis.key}`,
              cx: point.x,
              cy: point.y,
              r: hoveredAxis?.key === point.axis.key ? 5 : 4,
              fill: '#1B5E3A',
              onMouseEnter: () => setHoveredAxis(point.axis),
              onMouseLeave: () => setHoveredAxis(null),
            }),
          ),
          ...chartPoints.map((point) =>
            createElement(
              'text',
              {
                key: `label-${point.axis.key}`,
                x: point.labelX,
                y: point.labelY,
                textAnchor: point.labelX < center - 8 ? 'end' : point.labelX > center + 8 ? 'start' : 'middle',
                dominantBaseline: 'middle',
                fill: hoveredAxis?.key === point.axis.key ? '#1B5E3A' : '#18392b',
                fontSize: 12,
                fontWeight: 800,
                onMouseEnter: () => setHoveredAxis(point.axis),
                onMouseLeave: () => setHoveredAxis(null),
                style: { cursor: 'default' },
              },
              point.axis.label,
            ),
          ),
        ],
      )}
      <View style={styles.radarTooltip}>
        <Text style={styles.radarTooltipTitle}>
          {hoveredAxis ? hoveredAxis.label : 'Hover A Concept'}
        </Text>
        <Text style={styles.radarTooltipText}>
          {hoveredAxis
            ? hoveredAxis.description
            : 'Move over a label or point to see what that concept means.'}
        </Text>
      </View>
    </View>
  );
}

const colors = {
  mint: '#A8D5BA',
  green: '#6BBF8A',
  leaf: '#4B9B6E',
  forest: '#2E7D5C',
  deepGreen: '#1B5E3A',
  page: '#fbfdfb',
  panel: '#ffffff',
  softPanel: '#eef5f1',
  text: '#153126',
  muted: '#5d7168',
  line: '#dbe8e0',
  danger: '#a7352b',
  warning: '#9c5b00',
};

const serifFont = 'Georgia';

const styles = StyleSheet.create({
  safeArea: {
    flex: 1,
    backgroundColor: colors.page,
  },
  topHeader: {
    minHeight: 78,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingHorizontal: 36,
    paddingVertical: 14,
    backgroundColor: colors.panel,
    borderBottomWidth: 1,
    borderBottomColor: colors.line,
  },
  brand: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  brandText: {
    color: colors.text,
    fontFamily: serifFont,
    fontSize: 24,
    fontWeight: '700',
  },
  navActions: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 16,
  },
  navButton: {
    minHeight: 46,
    borderRadius: 999,
    justifyContent: 'center',
    backgroundColor: colors.deepGreen,
    paddingHorizontal: 22,
  },
  navButtonText: {
    color: '#fff',
    fontSize: 15,
    fontWeight: '900',
  },
  navLink: {
    paddingHorizontal: 8,
    paddingVertical: 8,
  },
  navLinkText: {
    color: colors.text,
    fontSize: 15,
    fontWeight: '800',
  },
  landingPage: {
    width: '100%',
  },
  hero: {
    minHeight: 680,
    alignItems: 'center',
    justifyContent: 'center',
    paddingHorizontal: 24,
    paddingVertical: 80,
    backgroundColor: colors.page,
  },
  eyebrow: {
    color: colors.deepGreen,
    fontSize: 13,
    fontWeight: '900',
    letterSpacing: 0,
    marginBottom: 18,
  },
  heroTitle: {
    color: colors.text,
    fontFamily: serifFont,
    fontSize: 96,
    lineHeight: 104,
    fontWeight: '700',
    textAlign: 'center',
  },
  tagline: {
    color: colors.text,
    fontFamily: serifFont,
    fontSize: 42,
    lineHeight: 50,
    fontWeight: '700',
    textAlign: 'center',
    marginTop: 8,
  },
  heroCopy: {
    maxWidth: 760,
    color: colors.muted,
    fontSize: 20,
    lineHeight: 30,
    textAlign: 'center',
    marginTop: 24,
  },
  heroButton: {
    minHeight: 52,
    minWidth: 156,
    borderRadius: 999,
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: colors.deepGreen,
    paddingHorizontal: 26,
    marginTop: 34,
  },
  heroButtonText: {
    color: '#fff',
    fontSize: 16,
    fontWeight: '900',
  },
  aboutSection: {
    width: '100%',
    maxWidth: 1240,
    alignSelf: 'center',
    paddingHorizontal: 24,
    paddingBottom: 90,
    gap: 18,
  },
  aboutEyebrow: {
    color: colors.deepGreen,
    fontSize: 13,
    fontWeight: '900',
    textAlign: 'center',
  },
  aboutTitle: {
    color: colors.text,
    fontFamily: serifFont,
    fontSize: 42,
    lineHeight: 50,
    fontWeight: '700',
    textAlign: 'center',
  },
  aboutIntro: {
    color: colors.muted,
    maxWidth: 760,
    alignSelf: 'center',
    textAlign: 'center',
    fontSize: 17,
    lineHeight: 26,
    marginBottom: 22,
  },
  aboutGrid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 18,
  },
  aboutCard: {
    flexBasis: 360,
    flexGrow: 1,
    minHeight: 190,
    borderRadius: 8,
    backgroundColor: colors.softPanel,
    padding: 22,
    justifyContent: 'flex-start',
    gap: 10,
  },
  aboutCardHovered: {
    backgroundColor: '#e4f0e9',
    borderWidth: 1,
    borderColor: '#c8dfd0',
  },
  aboutCardTitle: {
    color: colors.text,
    fontSize: 18,
    fontWeight: '900',
  },
  aboutCardText: {
    color: colors.muted,
    fontSize: 15,
    lineHeight: 23,
  },
  aboutCardDetail: {
    color: colors.text,
    fontSize: 14,
    lineHeight: 21,
    fontWeight: '700',
    marginTop: 4,
  },
  page: {
    width: '100%',
    padding: 24,
  },
  pageInner: {
    width: '100%',
    maxWidth: 1180,
    alignSelf: 'center',
    gap: 18,
  },
  horizontalStepper: {
    width: '100%',
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    flexWrap: 'wrap',
    gap: 14,
    paddingVertical: 4,
  },
  stepRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 10,
  },
  stepDot: {
    width: 30,
    height: 30,
    borderRadius: 15,
    alignItems: 'center',
    justifyContent: 'center',
    borderWidth: 1,
    borderColor: colors.line,
    backgroundColor: colors.panel,
  },
  stepDotActive: {
    borderColor: colors.deepGreen,
    backgroundColor: colors.deepGreen,
  },
  stepDotComplete: {
    borderColor: colors.forest,
    backgroundColor: colors.mint,
  },
  stepDotText: {
    color: colors.muted,
    fontSize: 13,
    fontWeight: '900',
  },
  stepDotTextActive: {
    color: '#fff',
  },
  stepLabel: {
    color: colors.muted,
    fontSize: 14,
    fontWeight: '800',
  },
  stepLabelActive: {
    color: colors.text,
  },
  stepLabelComplete: {
    color: colors.deepGreen,
  },
  initialStage: {
    minHeight: 560,
    alignItems: 'center',
    justifyContent: 'center',
    paddingVertical: 16,
  },
  assessmentGrid: {
    flex: 1,
    flexDirection: 'row',
    alignItems: 'flex-start',
    gap: 20,
  },
  assessmentGridStacked: {
    flexDirection: 'column',
  },
  leftPanel: {
    width: 360,
    maxWidth: '100%',
  },
  rightPanel: {
    flex: 1,
    minWidth: 0,
  },
  panel: {
    width: '100%',
    backgroundColor: colors.panel,
    borderColor: colors.line,
    borderWidth: 1,
    borderRadius: 8,
    padding: 18,
    gap: 14,
  },
  centerPanel: {
    maxWidth: 860,
    padding: 28,
  },
  panelHeader: {
    gap: 6,
  },
  centerHeading: {
    color: colors.text,
    fontFamily: serifFont,
    fontSize: 34,
    lineHeight: 42,
    fontWeight: '700',
  },
  panelTitle: {
    color: colors.text,
    fontSize: 19,
    lineHeight: 25,
    fontWeight: '900',
  },
  panelHelp: {
    color: colors.muted,
    fontSize: 14,
    lineHeight: 20,
  },
  symptomComposer: {
    gap: 16,
  },
  symptomComposerSplit: {
    flexDirection: 'row',
    alignItems: 'flex-start',
  },
  searchPane: {
    flex: 1,
    minWidth: 0,
    gap: 10,
    zIndex: 4,
  },
  searchBoxWrap: {
    position: 'relative',
    zIndex: 5,
  },
  selectedPane: {
    width: '100%',
    gap: 12,
    borderWidth: 1,
    borderColor: colors.line,
    borderRadius: 8,
    backgroundColor: '#fbfdfb',
    padding: 14,
  },
  selectedPaneSide: {
    width: 290,
    flexShrink: 0,
  },
  selectedTitle: {
    color: colors.text,
    fontSize: 15,
    fontWeight: '900',
  },
  sectionSubtitle: {
    color: colors.text,
    fontSize: 13,
    fontWeight: '900',
  },
  searchInput: {
    minHeight: 48,
    borderWidth: 1,
    borderColor: colors.line,
    borderRadius: 8,
    paddingHorizontal: 14,
    fontSize: 16,
    backgroundColor: '#fbfdfb',
    color: colors.text,
  },
  mutedText: {
    color: colors.muted,
    fontSize: 13,
    lineHeight: 18,
  },
  autocompleteMenu: {
    position: 'absolute',
    top: 54,
    left: 0,
    right: 0,
    borderWidth: 1,
    borderColor: colors.line,
    borderRadius: 8,
    overflow: 'hidden',
    backgroundColor: colors.panel,
    shadowColor: '#0f2a1f',
    shadowOpacity: 0.1,
    shadowRadius: 16,
    shadowOffset: { width: 0, height: 10 },
    elevation: 8,
    zIndex: 20,
  },
  autocompleteState: {
    minHeight: 52,
    flexDirection: 'row',
    alignItems: 'center',
    gap: 10,
    paddingHorizontal: 13,
    paddingVertical: 10,
  },
  symptomRow: {
    minHeight: 44,
    paddingHorizontal: 13,
    paddingVertical: 10,
    borderBottomWidth: 1,
    borderBottomColor: '#edf4ef',
    justifyContent: 'center',
  },
  symptomLabel: {
    color: colors.text,
    fontSize: 14,
    fontWeight: '800',
  },
  groupHint: {
    color: colors.muted,
    fontSize: 12,
    lineHeight: 16,
    marginTop: 2,
  },
  quickAddPanel: {
    borderWidth: 1,
    borderColor: colors.line,
    borderRadius: 8,
    backgroundColor: '#fbfdfb',
    padding: 14,
    gap: 10,
  },
  quickAddTitle: {
    color: colors.text,
    fontSize: 13,
    fontWeight: '900',
  },
  quickAddGrid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 8,
  },
  quickAddChip: {
    maxWidth: '100%',
    borderRadius: 999,
    borderWidth: 1,
    borderColor: '#b9d9c5',
    backgroundColor: '#f1f8f4',
    paddingHorizontal: 12,
    paddingVertical: 8,
  },
  quickAddText: {
    color: colors.deepGreen,
    fontSize: 13,
    fontWeight: '900',
  },
  selectedHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
  },
  selectedList: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 8,
  },
  selectedTag: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
    maxWidth: '100%',
    borderWidth: 1,
    borderColor: '#c7dfcf',
    backgroundColor: '#f1f8f4',
    borderRadius: 999,
    paddingLeft: 12,
    paddingRight: 9,
    paddingVertical: 7,
  },
  selectedTagText: {
    flexShrink: 1,
    color: colors.text,
    fontSize: 13,
    lineHeight: 18,
    fontWeight: '800',
  },
  selectedTagRemove: {
    color: colors.deepGreen,
    fontSize: 17,
    lineHeight: 18,
    fontWeight: '900',
  },
  emptySelectedState: {
    minHeight: 72,
    justifyContent: 'center',
    borderWidth: 1,
    borderColor: colors.line,
    borderRadius: 8,
    padding: 12,
  },
  countText: {
    color: colors.deepGreen,
    fontSize: 13,
    fontWeight: '900',
  },
  chips: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 8,
  },
  chipId: {
    color: colors.muted,
    fontSize: 11,
    marginTop: 1,
  },
  subtleChip: {
    maxWidth: '100%',
    borderRadius: 8,
    borderWidth: 1,
    borderColor: colors.line,
    backgroundColor: '#fbfdfb',
    paddingHorizontal: 10,
    paddingVertical: 7,
  },
  subtleChipText: {
    color: colors.text,
    fontSize: 13,
    fontWeight: '800',
  },
  buttonRow: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 10,
  },
  buttonArea: {
    position: 'relative',
    gap: 8,
  },
  tooltip: {
    alignSelf: 'flex-start',
    borderWidth: 1,
    borderColor: colors.line,
    borderRadius: 8,
    backgroundColor: colors.panel,
    paddingHorizontal: 10,
    paddingVertical: 7,
    shadowColor: '#0f2a1f',
    shadowOpacity: 0.08,
    shadowRadius: 10,
    shadowOffset: { width: 0, height: 6 },
    elevation: 4,
  },
  tooltipText: {
    color: colors.text,
    fontSize: 12,
    fontWeight: '800',
  },
  primaryButton: {
    flexGrow: 1,
    minHeight: 48,
    minWidth: 190,
    borderRadius: 999,
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: colors.deepGreen,
    borderWidth: 1,
    borderColor: colors.deepGreen,
    paddingHorizontal: 16,
  },
  disabledButton: {
    backgroundColor: '#f4f8f5',
    borderColor: colors.line,
  },
  disabledButtonText: {
    color: colors.muted,
  },
  primaryButtonText: {
    color: '#ffffff',
    fontSize: 15,
    lineHeight: 19,
    fontWeight: '900',
    textAlign: 'center',
  },
  secondaryButton: {
    minHeight: 48,
    borderRadius: 999,
    borderWidth: 1,
    borderColor: colors.line,
    alignItems: 'center',
    justifyContent: 'center',
    paddingHorizontal: 16,
    backgroundColor: colors.panel,
  },
  secondaryButtonText: {
    color: colors.text,
    fontSize: 14,
    fontWeight: '900',
  },
  pressed: {
    opacity: 0.74,
  },
  errorText: {
    color: colors.danger,
    backgroundColor: '#fff4f2',
    borderColor: '#efc7c2',
    borderWidth: 1,
    borderRadius: 8,
    padding: 10,
    fontSize: 13,
    lineHeight: 18,
  },
  warningText: {
    color: colors.warning,
    fontSize: 13,
    lineHeight: 19,
  },
  resultsStack: {
    gap: 16,
  },
  stepNavigation: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    justifyContent: 'flex-end',
    gap: 8,
  },
  stepPrimaryButton: {
    minHeight: 36,
    borderRadius: 999,
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: colors.deepGreen,
    borderWidth: 1,
    borderColor: colors.deepGreen,
    paddingHorizontal: 14,
    paddingVertical: 7,
  },
  stepPrimaryButtonText: {
    color: '#ffffff',
    fontSize: 13,
    lineHeight: 17,
    fontWeight: '900',
    textAlign: 'center',
  },
  stepSecondaryButton: {
    minHeight: 36,
    borderRadius: 999,
    borderWidth: 1,
    borderColor: colors.line,
    alignItems: 'center',
    justifyContent: 'center',
    paddingHorizontal: 13,
    paddingVertical: 7,
    backgroundColor: colors.panel,
  },
  stepSecondaryButtonText: {
    color: colors.text,
    fontSize: 13,
    lineHeight: 17,
    fontWeight: '900',
  },
  topSyndromeBox: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    gap: 16,
    borderRadius: 8,
    backgroundColor: '#f1f8f4',
    borderWidth: 1,
    borderColor: '#c7dfcf',
    padding: 16,
  },
  topSyndromeLabel: {
    color: colors.text,
    fontFamily: serifFont,
    fontSize: 24,
    lineHeight: 30,
    fontWeight: '700',
  },
  compactRow: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    gap: 12,
    borderTopWidth: 1,
    borderTopColor: '#edf4ef',
    paddingTop: 10,
  },
  herbRecommendationRow: {
    borderTopWidth: 1,
    borderTopColor: '#edf4ef',
    paddingTop: 12,
    gap: 10,
  },
  resultText: {
    flex: 1,
    minWidth: 0,
  },
  resultLabel: {
    color: colors.text,
    fontSize: 15,
    fontWeight: '900',
  },
  resultId: {
    color: colors.muted,
    fontSize: 12,
    marginTop: 2,
  },
  bodyText: {
    color: colors.text,
    fontSize: 14,
    lineHeight: 20,
  },
  finePrint: {
    color: colors.muted,
    fontSize: 12,
    lineHeight: 17,
    marginTop: 3,
  },
  metric: {
    color: colors.deepGreen,
    fontSize: 14,
    fontWeight: '900',
  },
  targetConceptList: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    alignItems: 'center',
    gap: 7,
    marginTop: 10,
  },
  targetConceptLead: {
    color: colors.muted,
    fontSize: 12,
    fontWeight: '900',
  },
  targetConceptChip: {
    borderRadius: 999,
    borderWidth: 1,
    borderColor: '#c7dfcf',
    backgroundColor: '#f1f8f4',
    paddingHorizontal: 9,
    paddingVertical: 5,
  },
  targetConceptText: {
    color: colors.deepGreen,
    fontSize: 12,
    fontWeight: '900',
  },
  whyCard: {
    borderWidth: 1,
    borderColor: '#c7dfcf',
    borderRadius: 8,
    backgroundColor: '#f1f8f4',
    padding: 14,
    gap: 6,
  },
  whyTitle: {
    color: colors.deepGreen,
    fontSize: 14,
    fontWeight: '900',
  },
  radarCard: {
    width: '100%',
    alignItems: 'center',
    borderWidth: 1,
    borderColor: '#d9e8de',
    borderRadius: 8,
    backgroundColor: '#fbfdfb',
    padding: 12,
    gap: 10,
  },
  radarTooltip: {
    width: '100%',
    minHeight: 74,
    borderWidth: 1,
    borderColor: '#d9e8de',
    borderRadius: 8,
    backgroundColor: colors.panel,
    paddingHorizontal: 12,
    paddingVertical: 10,
    gap: 4,
  },
  radarTooltipTitle: {
    color: colors.deepGreen,
    fontSize: 13,
    fontWeight: '900',
  },
  radarTooltipText: {
    color: colors.text,
    fontSize: 13,
    lineHeight: 18,
  },
});

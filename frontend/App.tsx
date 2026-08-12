import { StatusBar } from 'expo-status-bar';
import { useEffect, useMemo, useState } from 'react';
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

function formatPercent(value: number) {
  return `${Math.round(value * 1000) / 10}%`;
}

function formatScore(value: number) {
  return value.toFixed(3);
}

function titleCaseLabel(label: string) {
  return label.replace(/\p{L}[\p{L}'-]*/gu, (word) => {
    const [first = '', ...rest] = Array.from(word);
    return `${first.toLocaleUpperCase()}${rest.join('').toLocaleLowerCase()}`;
  });
}

function symptomDisplayLabel(symptom: MetadataRecord) {
  return titleCaseLabel(symptom.label || symptom.id);
}

type SymptomGroup = {
  key: string;
  label: string;
  ids: string[];
};

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
        <View style={styles.brandMark}>
          <Text style={styles.brandMarkText}>T</Text>
        </View>
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
        <Text style={styles.eyebrow}>CONCEPT-GUIDED TCM INSIGHT</Text>
        <Text style={styles.heroTitle}>TCMNet</Text>
        <Text style={styles.tagline}>Make pattern recognition clearer.</Text>
        <Text style={styles.heroCopy}>
          TCMNet is a research prototype that turns selected symptoms into syndrome
          predictions, herb recommendations, and a simple explanation of the model signals.
        </Text>
        <Pressable onPress={onStart} style={({ pressed }) => [styles.heroButton, pressed && styles.pressed]}>
          <Text style={styles.heroButtonText}>Get started</Text>
        </Pressable>
      </View>

      <View style={styles.aboutSection}>
        <Text style={styles.aboutEyebrow}>ABOUT TCMNET</Text>
        <Text style={styles.aboutTitle}>Built to make TCM prediction easier to inspect.</Text>
        <Text style={styles.aboutIntro}>
          The goal is not to replace clinical judgment. TCMNet helps explore how symptoms,
          traditional concepts, syndromes, and herbs connect inside a trained model.
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
      'The interface groups duplicate translated labels while preserving the model IDs needed for prediction.',
  },
  {
    title: 'Concept Signals',
    text: 'The model estimates TCM concept patterns such as internal, heat, yin, and organ groups.',
    detail:
      'These concept scores make the model output easier to inspect than a syndrome label alone.',
  },
  {
    title: 'Syndrome Prediction',
    text: 'TCMNet ranks likely syndromes from the selected symptom pattern.',
    detail:
      'The app shows the top syndrome and alternatives with model scores, so uncertainty remains visible.',
  },
  {
    title: 'Herb Ranking',
    text: 'Herbs are scored from concept fit and known syndrome-herb relationships.',
    detail:
      'The herb recommender is frozen and reproducible: it blends concept similarity with syndrome-herb priors.',
  },
  {
    title: 'Future Direction',
    text: 'Next steps include stronger labels, better grouping, richer explanations, and model validation.',
    detail:
      'This prototype is designed so future work can improve metadata, confidence calibration, and clinical review.',
  },
  {
    title: 'Final Report',
    text: 'A final report and findings link will live here once the project writeup is complete.',
    detail:
      'Use this space for the CS229 report, model results, limitations, and supporting analysis.',
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

  const commonSymptoms = useMemo(
    () =>
      COMMON_SYMPTOM_LABELS.map((label) =>
        groupedSymptoms.find((group) => group.key === label.toLocaleLowerCase()),
      )
        .filter((group): group is SymptomGroup => Boolean(group))
        .filter((group) => !selectedGroupKeys.has(group.key)),
    [groupedSymptoms, selectedGroupKeys],
  );

  function addSymptom(symptom: SymptomGroup) {
    setSelectedSymptoms((current) =>
      current.some((item) => item.key === symptom.key) ? current : [...current, symptom],
    );
    setQuery('');
    setPrediction(null);
  }

  function removeSymptom(symptomKey: string) {
    setSelectedSymptoms((current) => current.filter((symptom) => symptom.key !== symptomKey));
    setPrediction(null);
  }

  function clearSymptoms() {
    setSelectedSymptoms([]);
    setPrediction(null);
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
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Prediction request failed.');
    } finally {
      setPredicting(false);
    }
  }

  return (
    <ScrollView contentContainerStyle={styles.page}>
      <View style={styles.pageInner}>
        <View style={styles.disclaimerBanner}>
          <Text style={styles.disclaimerText}>
            Educational and research-use prototype. Not medical advice.
          </Text>
        </View>

        <HorizontalStepper activeStep={prediction ? 3 : 1} />

        {!prediction ? (
          <View style={styles.initialStage}>
            <SymptomPanel
              title="What symptoms are present?"
              helperText="Search by symptom label, then add the symptoms observed in the case."
              query={query}
              setQuery={setQuery}
              searchResults={searchResults}
              commonSymptoms={commonSymptoms}
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
                helperText="Adjust the symptom set and run the model again."
                query={query}
                setQuery={setQuery}
                searchResults={searchResults}
                commonSymptoms={commonSymptoms}
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
              <Results prediction={prediction} />
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
    { id: 2, label: 'Prediction' },
    { id: 3, label: 'Explanation' },
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
  commonSymptoms: SymptomGroup[];
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
  commonSymptoms,
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
                          Includes {symptom.ids.length} related model entries
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
              <Text style={styles.quickAddTitle}>Commonly Reported Symptoms</Text>
              <View style={styles.quickAddGrid}>
                {commonSymptoms.map((symptom) => (
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
                Start typing a symptom, such as cough, fever, or abdominal pain.
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

function Results({ prediction }: { prediction: PredictionResponse }) {
  const [topSyndrome, ...otherSyndromes] = prediction.syndromes;

  return (
    <View style={styles.resultsStack}>
      <View style={styles.panel}>
        <Text style={styles.panelTitle}>Top Syndrome Prediction</Text>
        {topSyndrome ? <TopSyndrome syndrome={topSyndrome} /> : null}
      </View>

      <View style={styles.panel}>
        <Text style={styles.panelTitle}>Other Possible Syndromes</Text>
        {otherSyndromes.map((syndrome) => (
          <SyndromeRow key={`${syndrome.syndrome_id}-${syndrome.index}`} syndrome={syndrome} />
        ))}
      </View>

      <View style={styles.panel}>
        <Text style={styles.panelTitle}>Herb Recommendations</Text>
        {prediction.herbs.map((herb) => (
          <HerbRow key={herb.herb_id} herb={herb} />
        ))}
      </View>

      <Explanation prediction={prediction} />
    </View>
  );
}

function TopSyndrome({ syndrome }: { syndrome: SyndromePrediction }) {
  return (
    <View style={styles.topSyndromeBox}>
      <View style={styles.resultText}>
        <Text style={styles.topSyndromeLabel}>{syndrome.label}</Text>
        <Text style={styles.resultId}>{syndrome.syndrome_id}</Text>
      </View>
      <View style={styles.scoreBadge}>
        <Text style={styles.scoreBadgeValue}>{formatPercent(syndrome.confidence)}</Text>
        <Text style={styles.scoreBadgeLabel}>model score</Text>
      </View>
    </View>
  );
}

function SyndromeRow({ syndrome }: { syndrome: SyndromePrediction }) {
  return (
    <View style={styles.compactRow}>
      <View style={styles.resultText}>
        <Text style={styles.resultLabel} numberOfLines={1}>
          {syndrome.label}
        </Text>
        <Text style={styles.resultId}>{syndrome.syndrome_id}</Text>
      </View>
      <Text style={styles.metric}>{formatPercent(syndrome.confidence)}</Text>
    </View>
  );
}

function HerbRow({ herb }: { herb: HerbRecommendation }) {
  return (
    <View style={styles.compactRow}>
      <View style={styles.resultText}>
        <Text style={styles.resultLabel} numberOfLines={1}>
          {herb.label}
        </Text>
        <Text style={styles.resultId}>{herb.herb_id}</Text>
        <Text style={styles.finePrint}>
          concept {formatScore(herb.concept_similarity)} · prior{' '}
          {formatScore(herb.syndrome_prior)}
        </Text>
      </View>
      <Text style={styles.metric}>{formatScore(herb.score)}</Text>
    </View>
  );
}

function Explanation({ prediction }: { prediction: PredictionResponse }) {
  return (
    <View style={styles.panel}>
      <Text style={styles.panelTitle}>Explanation</Text>

      <Text style={styles.sectionSubtitle}>Recognized Symptoms</Text>
      <View style={styles.chips}>
        {prediction.explanation.matching_symptoms.map((symptom) => (
          <View key={symptom.id} style={styles.subtleChip}>
            <Text style={styles.subtleChipText} numberOfLines={1}>
              {titleCaseLabel(symptom.label)}
            </Text>
          </View>
        ))}
      </View>

      {prediction.input.unknown_symptom_ids.length ? (
        <>
          <Text style={styles.sectionSubtitle}>Unknown Symptom IDs</Text>
          <Text style={styles.warningText}>{prediction.input.unknown_symptom_ids.join(', ')}</Text>
        </>
      ) : null}

      <Text style={styles.sectionSubtitle}>Strongest Concept Signals</Text>
      <ConceptBars concepts={prediction.explanation.concept_alignment} />

      <Text style={styles.sectionSubtitle}>Syndrome-Herb Association Summary</Text>
      <Text style={styles.bodyText}>
        {prediction.explanation.syndrome_herb_associations.label} (
        {prediction.explanation.syndrome_herb_associations.syndrome_id}) has{' '}
        {prediction.explanation.syndrome_herb_associations.total_associated_herbs} known
        herb associations in the frozen artifact.
      </Text>
      <View style={styles.chips}>
        {prediction.explanation.syndrome_herb_associations.associated_herbs.map((herb) => (
          <View key={`assoc-${herb.id}`} style={styles.subtleChip}>
            <Text style={styles.subtleChipText} numberOfLines={1}>
              {herb.label}
            </Text>
            <Text style={styles.chipId}>{herb.id}</Text>
          </View>
        ))}
      </View>

      <Text style={styles.sectionSubtitle}>Herb Ranking Details</Text>
      <Text style={styles.bodyText}>
        {prediction.explanation.herb_ranking.formula}; alpha ={' '}
        {formatScore(prediction.explanation.herb_ranking.alpha)}.
      </Text>
      {prediction.explanation.herb_ranking.items.map((item) => (
        <View key={`rank-${item.herb_id}`} style={styles.compactRow}>
          <View style={styles.resultText}>
            <Text style={styles.resultLabel} numberOfLines={1}>
              {item.label}
            </Text>
            <Text style={styles.finePrint}>
              concept {formatScore(item.concept_similarity)} · prior{' '}
              {formatScore(item.syndrome_prior)} ·{' '}
              {item.known_for_predicted_syndrome ? 'known association' : 'not linked'}
            </Text>
          </View>
          <Text style={styles.metric}>{formatScore(item.score)}</Text>
        </View>
      ))}
    </View>
  );
}

function ConceptBars({ concepts }: { concepts: ConceptScore[] }) {
  return (
    <View style={styles.conceptStack}>
      {concepts.map((concept) => (
        <View key={concept.id} style={styles.conceptBarRow}>
          <View style={styles.conceptBarHeader}>
            <Text style={styles.conceptLabel}>{concept.label}</Text>
            <Text style={styles.metric}>{formatScore(concept.score)}</Text>
          </View>
          <View style={styles.barTrack}>
            <View style={[styles.barFill, { width: `${Math.min(concept.score * 100, 100)}%` }]} />
          </View>
        </View>
      ))}
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
    gap: 10,
  },
  brandMark: {
    width: 34,
    height: 34,
    borderRadius: 17,
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: colors.deepGreen,
  },
  brandMarkText: {
    color: '#fff',
    fontWeight: '900',
    fontSize: 16,
  },
  brandText: {
    color: colors.text,
    fontSize: 24,
    fontWeight: '900',
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
  disclaimerBanner: {
    alignSelf: 'center',
    borderWidth: 1,
    borderColor: '#c7dfcf',
    borderRadius: 999,
    backgroundColor: '#f1f8f4',
    paddingHorizontal: 14,
    paddingVertical: 8,
  },
  disclaimerText: {
    color: colors.text,
    fontSize: 13,
    lineHeight: 17,
    fontWeight: '800',
    textAlign: 'center',
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
  scoreBadge: {
    minWidth: 104,
    alignItems: 'center',
    borderRadius: 8,
    backgroundColor: colors.panel,
    borderWidth: 1,
    borderColor: '#c7dfcf',
    paddingHorizontal: 12,
    paddingVertical: 10,
  },
  scoreBadgeValue: {
    color: colors.deepGreen,
    fontSize: 20,
    fontWeight: '900',
  },
  scoreBadgeLabel: {
    color: colors.muted,
    fontSize: 11,
    fontWeight: '800',
    marginTop: 2,
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
  conceptStack: {
    gap: 10,
  },
  conceptBarRow: {
    gap: 6,
  },
  conceptBarHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    gap: 10,
  },
  conceptLabel: {
    color: colors.text,
    fontSize: 13,
    fontWeight: '900',
  },
  barTrack: {
    height: 7,
    borderRadius: 8,
    backgroundColor: '#e5efe8',
    overflow: 'hidden',
  },
  barFill: {
    height: 7,
    borderRadius: 8,
    backgroundColor: colors.forest,
  },
});

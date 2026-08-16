import { StatusBar } from 'expo-status-bar';
import { createElement, useEffect, useMemo, useRef, useState } from 'react';
import {
  ActivityIndicator,
  Animated,
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
import {
  BODY_MAP_SOURCE,
  BODY_MAP_VIEWBOX,
  BODY_REGIONS,
  type BodyRegion,
  type BodyView,
} from './src/bodyMapRegions';
import type {
  ConceptScore,
  HerbRecommendation,
  MetadataRecord,
  PatientIntakeBasics,
  PatientIntakeSymptom,
  PredictionResponse,
  SyndromePrediction,
} from './src/types';
import {
  getSymptomSuggestions,
  type SymptomSuggestion,
} from './src/symptomSuggestionEngine';

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

type AssessmentStep = 1 | 2 | 3 | 4;
type IntakeStep = 1 | 2 | 3;

const SEX_GENDER_OPTIONS = [
  'Female',
  'Male',
  'Non-binary',
  'Prefer not to say',
];

const SEVERITY_DESCRIPTIONS: Record<number, string> = {
  1: 'Barely noticeable',
  2: 'Mild, easy to ignore',
  3: 'Mild but distracting',
  4: 'Moderate discomfort',
  5: 'Moderate, affects focus',
  6: 'Strong discomfort',
  7: 'Severe, hard to ignore',
  8: 'Very severe',
  9: 'Nearly unbearable',
  10: 'Worst possible',
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

function syndromeDescriptionText(description: string) {
  return description
    .replace(/This is a way of grouping related signs and symptoms.*?Western medical diagnosis\./g, '')
    .replace(/This description helps explain the pattern language behind the model's syndrome prediction\./g, '')
    .replace(/\s+/g, ' ')
    .trim();
}

function reportDateLabel() {
  return new Date().toLocaleDateString(undefined, {
    year: 'numeric',
    month: 'long',
    day: 'numeric',
  });
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

type IntakeBasics = PatientIntakeBasics;
type SymptomGroup = PatientIntakeSymptom;

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
        <Text style={styles.heroTitle}>TCMNet</Text>
        <Text style={styles.heroCopy}>
          Bridging traditional medicine with modern data science to translate complex
          symptoms into pattern diagnoses and customized herbal prescriptions.
        </Text>
        <Pressable onPress={onStart} style={({ pressed }) => [styles.heroButton, pressed && styles.pressed]}>
          <Text style={styles.heroButtonText}>Get started</Text>
        </Pressable>
      </View>

      <View style={styles.aboutSection}>
        <Text style={styles.aboutEyebrow}>ABOUT TCMNET</Text>
        <Text style={styles.aboutTitle}>From symptoms to syndromes, with interpretable reasoning in between.</Text>
        <Text style={styles.aboutIntro}>
          TCMNet translates symptom patterns into traditional diagnostic concepts, predicts likely syndromes, and connects those predictions to herbal recommendations.
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
    </View>
  );
}

const ABOUT_CARDS = [
  {
    title: 'What is TCM?',
    text: 'Traditional Chinese Medicine diagnoses illness by interpreting combinations of symptoms as underlying patterns, or syndromes, which then guide treatment.',
  },
  {
    title: 'The 14 Concepts',
    text: 'TCMNet tracks 14 core concepts, including Yin/Yang, Hot/Cold, Interior/Exterior, Deficiency/Excess, and six organ or meridian groups.',
  },
  {
    title: 'Syndrome Prediction',
    text: "TCMNet combines a patient's symptoms with these learned concepts to rank which of 228 TCM syndromes best matches their pattern.",
  },
  {
    title: 'Herb Recommendation',
    text: 'TCMNet ranks herbs with a neural herb head conditioned on symptom features, predicted concepts, and syndrome evidence.',
  },
  {
    title: 'How Our Model Works',
    text: 'TCMNet uses a multi-task neural network that learns shared symptom features, predicts concepts and syndromes, then ranks herbs with a dedicated neural head.',
  },
  {
    title: 'Model Performance',
    text: 'TCMNet reached 86.93% syndrome accuracy and neural herb-head Precision@5 of 57.56% under inference-condition evaluation.',
  },
];

function AssessmentPage({ isWide }: { isWide: boolean }) {
  const [symptoms, setSymptoms] = useState<MetadataRecord[]>([]);
  const [selectedSymptoms, setSelectedSymptoms] = useState<SymptomGroup[]>([]);
  const [selectedBodyRegionIds, setSelectedBodyRegionIds] = useState<string[]>([]);
  const [intakeBasics, setIntakeBasics] = useState<IntakeBasics>({
    sexGender: '',
    age: null,
    mainConcern: '',
    onsetDate: '',
    onsetUnknown: false,
    severity: null,
  });
  const [intakeStep, setIntakeStep] = useState<IntakeStep>(1);
  const [query, setQuery] = useState('');
  const [loadingSymptoms, setLoadingSymptoms] = useState(true);
  const [predicting, setPredicting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [prediction, setPrediction] = useState<PredictionResponse | null>(null);
  const [assessmentStep, setAssessmentStep] = useState<AssessmentStep>(1);
  const [dismissedMatchedSymptomKeys, setDismissedMatchedSymptomKeys] = useState<string[]>([]);

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

  useEffect(() => {
    setDismissedMatchedSymptomKeys([]);
  }, [intakeBasics.mainConcern, selectedBodyRegionIds]);

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
  const selectedBodyRegions = useMemo(
    () => BODY_REGIONS.filter((region) => selectedBodyRegionIds.includes(region.id)),
    [selectedBodyRegionIds],
  );
  const intakeProgress = useMemo(() => {
    if (prediction) {
      return Math.min(100, 34 + assessmentStep * 16.5);
    }

    const basicsCompleted = [
      intakeBasics.sexGender,
      intakeBasics.age,
      intakeBasics.mainConcern.trim(),
      intakeBasics.onsetUnknown || intakeBasics.onsetDate,
      intakeBasics.severity,
    ].filter(Boolean).length;
    const basicsProgress = (basicsCompleted / 5) * 34;
    const locationProgress = selectedBodyRegionIds.length > 0 ? 33 : 0;
    const symptomsProgress = selectedSymptoms.length > 0 ? 33 : 0;

    if (intakeStep === 1) {
      return basicsProgress;
    }
    if (intakeStep === 2) {
      return 34 + locationProgress;
    }
    return 67 + symptomsProgress;
  }, [assessmentStep, intakeBasics, intakeStep, prediction, selectedBodyRegionIds.length, selectedSymptoms.length]);

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

  const matchedSymptomCandidates = useMemo(
    () =>
      getSymptomSuggestions({
        intakeBasics,
        selectedBodyRegionIds,
        groupedSymptoms,
        selectedGroupKeys: new Set(),
        limit: 5,
      }),
    [groupedSymptoms, intakeBasics, selectedBodyRegionIds],
  );
  const visibleSuggestedSymptoms = useMemo(() => {
    const dismissedKeys = new Set(dismissedMatchedSymptomKeys);
    return matchedSymptomCandidates.filter(
      (suggestion) =>
        !dismissedKeys.has(suggestion.group.key) &&
        !selectedGroupKeys.has(suggestion.group.key),
    );
  }, [dismissedMatchedSymptomKeys, matchedSymptomCandidates, selectedGroupKeys]);

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
    setDismissedMatchedSymptomKeys([]);
    setPrediction(null);
    setAssessmentStep(1);
    setError(null);
  }

  function startOver() {
    setSelectedSymptoms([]);
    setSelectedBodyRegionIds([]);
    setIntakeBasics({
      sexGender: '',
      age: null,
      mainConcern: '',
      onsetDate: '',
      onsetUnknown: false,
      severity: null,
    });
    setIntakeStep(1);
    setQuery('');
    setDismissedMatchedSymptomKeys([]);
    setPrediction(null);
    setAssessmentStep(1);
    setError(null);
  }

  function dismissMatchedSymptom(symptomKey: string) {
    setDismissedMatchedSymptomKeys((current) =>
      current.includes(symptomKey) ? current : [...current, symptomKey],
    );
  }

  function toggleBodyRegion(regionId: string) {
    setSelectedBodyRegionIds((current) =>
      current.includes(regionId)
        ? current.filter((id) => id !== regionId)
        : [...current, regionId],
    );
    setPrediction(null);
    setAssessmentStep(1);
  }

  function clearBodyRegions() {
    setSelectedBodyRegionIds([]);
    setPrediction(null);
    setAssessmentStep(1);
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
        <HorizontalStepper
          activeStep={prediction ? assessmentStep : 1}
          progress={intakeProgress}
          showProgress={!prediction}
        />

        {!prediction ? (
          <View style={styles.initialStage}>
            <View style={styles.intakeStack}>
              {intakeStep === 1 ? (
                <IntakeBasicsPanel
                  basics={intakeBasics}
                  onChange={setIntakeBasics}
                  onNext={() => setIntakeStep(2)}
                />
              ) : null}
              {intakeStep === 2 ? (
                <>
                  <BodyLocationPanel
                    selectedRegionIds={selectedBodyRegionIds}
                    selectedRegions={selectedBodyRegions}
                    onToggleRegion={toggleBodyRegion}
                    onClear={clearBodyRegions}
                  />
                  <StepNavigation
                    secondaryLabel="Back"
                    primaryLabel="Next: Symptoms"
                    onSecondary={() => setIntakeStep(1)}
                    onPrimary={() => setIntakeStep(3)}
                  />
                </>
              ) : null}
              {intakeStep === 3 ? (
                <>
                  <SymptomPanel
                    title="Review matched symptoms"
                    helperText="Review suggested matches from the intake answers, then add the symptoms that best fit this case."
                    query={query}
                    setQuery={setQuery}
                    searchResults={searchResults}
                    suggestedSymptoms={visibleSuggestedSymptoms}
                    hasMatchedSymptomCandidates={matchedSymptomCandidates.length > 0}
                    trySymptoms={trySymptoms}
                    selectedSymptoms={selectedSymptoms}
                    loadingSymptoms={loadingSymptoms}
                    predicting={predicting}
                    error={error}
                    isWide={isWide}
                    onAddSymptom={addSymptom}
                    onDismissSuggestedSymptom={dismissMatchedSymptom}
                    onRemoveSymptom={removeSymptom}
                    onClear={clearSymptoms}
                    onPredict={runPrediction}
                    onSelectFirstResult={selectFirstSearchResult}
                    primaryLabel="Predict syndrome and herbs"
                    centered
                  />
                  <StepNavigation
                    secondaryLabel="Back"
                    onSecondary={() => setIntakeStep(2)}
                  />
                </>
              ) : null}
            </View>
          </View>
        ) : (
          <View style={[styles.assessmentGrid, !isWide && styles.assessmentGridStacked]}>
            <View style={styles.leftPanel}>
              <SelectedSymptomsSidebar
                selectedSymptoms={selectedSymptoms}
                selectedBodyRegions={selectedBodyRegions}
                onStartOver={startOver}
              />
            </View>
            <View style={styles.rightPanel}>
              <ResultStage
                prediction={prediction}
                selectedSymptoms={selectedSymptoms}
                selectedBodyRegions={selectedBodyRegions}
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

function HorizontalStepper({
  activeStep,
  progress,
  showProgress,
}: {
  activeStep: number;
  progress: number;
  showProgress: boolean;
}) {
  const steps = [
    { id: 1, label: 'Input symptoms' },
    { id: 2, label: 'Syndrome prediction' },
    { id: 3, label: 'Herb recommendation' },
    { id: 4, label: 'Summary report' },
  ];
  const boundedProgress = Math.max(0, Math.min(progress, 100));

  return (
    <View style={styles.stepperWrap}>
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
      {showProgress ? (
        <View style={styles.progressWrap}>
          <View style={styles.progressTrack}>
            <View
              style={[
                styles.progressFill,
                {
                  width: `${boundedProgress}%`,
                  transitionProperty: 'width',
                  transitionDuration: '260ms',
                  transitionTimingFunction: 'ease',
                } as Record<string, string>,
              ]}
            />
          </View>
          <Text style={styles.progressText}>{Math.round(boundedProgress)}%</Text>
        </View>
      ) : null}
    </View>
  );
}

type SymptomPanelProps = {
  title: string;
  helperText: string;
  query: string;
  setQuery: (value: string) => void;
  searchResults: SymptomGroup[];
  suggestedSymptoms: SymptomSuggestion[];
  hasMatchedSymptomCandidates: boolean;
  trySymptoms: SymptomGroup[];
  selectedSymptoms: SymptomGroup[];
  loadingSymptoms: boolean;
  predicting: boolean;
  error: string | null;
  isWide: boolean;
  onAddSymptom: (symptom: SymptomGroup) => void;
  onDismissSuggestedSymptom: (symptomKey: string) => void;
  onRemoveSymptom: (symptomKey: string) => void;
  onClear: () => void;
  onPredict: () => void;
  onSelectFirstResult: () => void;
  primaryLabel: string;
  centered?: boolean;
};

function IntakeBasicsPanel({
  basics,
  onChange,
  onNext,
}: {
  basics: IntakeBasics;
  onChange: (basics: IntakeBasics) => void;
  onNext: () => void;
}) {
  const displayedAge = basics.age ?? 35;
  const displayedSeverity = basics.severity ?? 5;

  function updateBasics(next: Partial<IntakeBasics>) {
    onChange({ ...basics, ...next });
  }

  return (
    <View style={[styles.panel, styles.centerPanel]}>
      <View style={styles.panelHeader}>
        <Text style={styles.centerHeading}>Tell us what's going on</Text>
        <Text style={styles.panelHelp}>
          Start with a few basics. We will use this to suggest symptoms you can review before prediction.
        </Text>
      </View>

      <View style={styles.formStack}>
        <View style={styles.formField}>
          <Text style={styles.formLabel}>Sex / gender</Text>
          <View style={styles.optionGrid}>
            {SEX_GENDER_OPTIONS.map((option) => {
              const isSelected = basics.sexGender === option;
              return (
                <Pressable
                  key={option}
                  onPress={() => updateBasics({ sexGender: option })}
                  style={({ pressed }) => [
                    styles.optionChip,
                    isSelected && styles.optionChipSelected,
                    pressed && styles.pressed,
                  ]}
                >
                  <Text
                    style={[
                      styles.optionChipText,
                      isSelected && styles.optionChipTextSelected,
                    ]}
                  >
                    {option}
                  </Text>
                </Pressable>
              );
            })}
          </View>
        </View>

        <AgeField
          value={basics.age}
          onChange={(value) => updateBasics({ age: value })}
        />

        <View style={styles.formField}>
          <Text style={styles.formLabel}>Main concern(s)</Text>
          <TextInput
            value={basics.mainConcern}
            onChangeText={(value) => updateBasics({ mainConcern: value })}
            placeholder="For example: stomach pain, cough, fatigue"
            autoCapitalize="sentences"
            style={styles.formInput}
            placeholderTextColor={colors.muted}
          />
          <Text style={styles.formHelp}>
            Separate different concerns or symptoms with commas.
          </Text>
        </View>

        <DateField
          label="Onset"
          value={basics.onsetDate}
          unknown={basics.onsetUnknown}
          onChange={(value) => updateBasics({ onsetDate: value })}
          onUnknownChange={(unknown) =>
            updateBasics({ onsetUnknown: unknown, onsetDate: unknown ? '' : basics.onsetDate })
          }
        />

        <SeveritySliderField
          label="Severity"
          value={displayedSeverity}
          onChange={(value) => updateBasics({ severity: value })}
        />
      </View>

      <StepNavigation primaryLabel="Next: Location" onPrimary={onNext} />
    </View>
  );
}

function SliderField({
  label,
  value,
  min,
  max,
  step,
  valueLabel,
  helperText,
  onChange,
}: {
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  valueLabel: string;
  helperText?: string;
  onChange: (value: number) => void;
}) {
  return (
    <View style={styles.formField}>
      <View style={styles.sliderHeader}>
        <Text style={styles.formLabel}>{label}</Text>
        <Text style={styles.sliderValue}>{valueLabel}</Text>
      </View>
      <View style={styles.sliderShell}>
        {createElement('input', {
          type: 'range',
          min,
          max,
          step,
          value,
          onChange: (event: { target: { value: string } }) => {
            onChange(Number(event.target.value));
          },
          style: {
            width: '100%',
            accentColor: colors.deepGreen,
          },
          'aria-label': label,
        })}
        <View style={styles.sliderLabels}>
          <Text style={styles.sliderEndpoint}>{min}</Text>
          <Text style={styles.sliderEndpoint}>{max}</Text>
        </View>
      </View>
      {helperText ? <Text style={styles.sliderHelp}>{helperText}</Text> : null}
    </View>
  );
}

function AgeField({
  value,
  onChange,
}: {
  value: number | null;
  onChange: (value: number | null) => void;
}) {
  return (
    <View style={styles.formField}>
      <Text style={styles.formLabel}>Age</Text>
      <TextInput
        value={value === null ? '' : String(value)}
        onChangeText={(text) => {
          const digitsOnly = text.replace(/\D/g, '').slice(0, 3);
          onChange(digitsOnly ? Number(digitsOnly) : null);
        }}
        placeholder="Enter age"
        keyboardType="number-pad"
        inputMode="numeric"
        maxLength={3}
        style={styles.formInput}
        placeholderTextColor={colors.muted}
      />
    </View>
  );
}

function SeveritySliderField({
  label,
  value,
  onChange,
}: {
  label: string;
  value: number;
  onChange: (value: number) => void;
}) {
  const percent = ((value - 1) / 9) * 100;
  const severityColor = severityColorForValue(value);
  const description = SEVERITY_DESCRIPTIONS[value];

  return (
    <View style={styles.formField}>
      <View style={styles.sliderHeader}>
        <Text style={styles.formLabel}>{label}</Text>
        <Text style={[styles.sliderValue, { color: severityColor }]}>{value} / 10</Text>
      </View>

      <View style={styles.severityControl}>
        <View
          style={[
            styles.severityTrack,
            {
              background: `linear-gradient(90deg, ${severityColor} 0%, ${severityColor} ${percent}%, #edf1ee ${percent}%, #edf1ee 100%)`,
            } as Record<string, string>,
          ]}
        >
          <View
            style={[
              styles.severityKnob,
              {
                left: `${percent}%`,
                backgroundColor: severityColor,
              } as Record<string, string>,
            ]}
          />
          {createElement('input', {
            type: 'range',
            min: 1,
            max: 10,
            step: 1,
            value,
            onChange: (event: { target: { value: string } }) => {
              onChange(Number(event.target.value));
            },
            style: {
              position: 'absolute',
              inset: 0,
              width: '100%',
              height: 28,
              opacity: 0,
              cursor: 'pointer',
            },
            'aria-label': label,
          })}
        </View>

        <View style={styles.sliderLabels}>
          <Text style={styles.sliderEndpoint}>1</Text>
          <Text style={styles.sliderEndpoint}>10</Text>
        </View>
        <View style={styles.severityScaleLabels}>
          <Text style={[styles.severityScaleText, { color: value <= 4 ? severityColor : colors.muted }]}>
            Mild
          </Text>
          <Text style={[styles.severityScaleText, { color: value >= 5 && value <= 6 ? severityColor : colors.muted }]}>
            Moderate
          </Text>
          <Text style={[styles.severityScaleText, { color: value >= 7 ? severityColor : colors.muted }]}>
            Severe
          </Text>
        </View>
      </View>

      <Text style={[styles.sliderHelp, { color: severityColor }]}>{description}</Text>
    </View>
  );
}

function severityColorForValue(value: number) {
  if (value <= 2) {
    return '#2f9e44';
  }
  if (value <= 4) {
    return '#c6a300';
  }
  if (value <= 6) {
    return '#e07a1f';
  }
  if (value <= 8) {
    return '#c94b1c';
  }
  return '#b42318';
}

function DateField({
  label,
  value,
  unknown,
  onChange,
  onUnknownChange,
}: {
  label: string;
  value: string;
  unknown: boolean;
  onChange: (value: string) => void;
  onUnknownChange: (value: boolean) => void;
}) {
  return (
    <View style={styles.formField}>
      <Text style={styles.formLabel}>{label}</Text>
      {createElement('input', {
        type: 'date',
        value,
        disabled: unknown,
        onChange: (event: { target: { value: string } }) => onChange(event.target.value),
        style: {
          minHeight: 48,
          borderWidth: 1,
          borderStyle: 'solid',
          borderColor: colors.line,
          borderRadius: 8,
          paddingLeft: 14,
          paddingRight: 14,
          fontSize: 16,
          backgroundColor: '#fbfdfb',
          color: colors.text,
          fontFamily: sansSerifFont,
          opacity: unknown ? 0.52 : 1,
        },
        'aria-label': label,
      })}
      <Pressable
        onPress={() => onUnknownChange(!unknown)}
        style={({ pressed }) => [
          styles.checkboxRow,
          pressed && styles.pressed,
        ]}
      >
        <View style={[styles.checkboxBox, unknown && styles.checkboxBoxChecked]}>
          {unknown ? <Text style={styles.checkboxMark}>✓</Text> : null}
        </View>
        <Text style={styles.checkboxText}>I don't know</Text>
      </Pressable>
    </View>
  );
}

function BodyLocationPanel({
  selectedRegionIds,
  selectedRegions,
  onToggleRegion,
  onClear,
}: {
  selectedRegionIds: string[];
  selectedRegions: BodyRegion[];
  onToggleRegion: (regionId: string) => void;
  onClear: () => void;
}) {
  const selectedRegionSet = useMemo(() => new Set(selectedRegionIds), [selectedRegionIds]);

  return (
    <View style={[styles.panel, styles.centerPanel]}>
      <View style={styles.panelHeader}>
        <Text style={styles.centerHeading}>Where is the symptom felt?</Text>
        <Text style={styles.panelHelp}>
          Select one or more body regions to log the symptom location for this case.
        </Text>
      </View>

      <View style={styles.bodyMapGrid}>
        <BodyMapFigure
          title="Front"
          view="front"
          selectedRegionIds={selectedRegionSet}
          onToggleRegion={onToggleRegion}
        />
        <BodyMapFigure
          title="Back"
          view="back"
          selectedRegionIds={selectedRegionSet}
          onToggleRegion={onToggleRegion}
        />
      </View>
      <Text style={styles.bodyMapSource}>Body map adapted from {BODY_MAP_SOURCE}</Text>

      <View style={styles.locationLog}>
        <View style={styles.selectedHeader}>
          <Text style={styles.selectedTitle}>Logged locations</Text>
          <Text style={styles.countText}>{selectedRegions.length}</Text>
        </View>

        {selectedRegions.length === 0 ? (
          <View style={styles.emptySelectedState}>
            <Text style={styles.mutedText}>No body locations selected yet.</Text>
          </View>
        ) : (
          <View style={styles.selectedList}>
            {selectedRegions.map((region) => (
              <Pressable
                key={`selected-region-${region.id}`}
                onPress={() => onToggleRegion(region.id)}
                style={({ pressed }) => [styles.selectedTag, pressed && styles.pressed]}
              >
                <Text style={styles.selectedTagText} numberOfLines={1}>
                  {region.label} ({region.view})
                </Text>
                <Text style={styles.selectedTagRemove}>×</Text>
              </Pressable>
            ))}
          </View>
        )}

        {selectedRegions.length > 0 ? (
          <Pressable
            onPress={onClear}
            style={({ pressed }) => [styles.clearTextButton, styles.locationClearButton, pressed && styles.pressed]}
          >
            <Text style={styles.clearTextButtonText}>Clear locations</Text>
          </Pressable>
        ) : null}
      </View>
    </View>
  );
}

function BodyMapFigure({
  title,
  view,
  selectedRegionIds,
  onToggleRegion,
}: {
  title: string;
  view: BodyView;
  selectedRegionIds: Set<string>;
  onToggleRegion: (regionId: string) => void;
}) {
  const [hoveredRegion, setHoveredRegion] = useState<BodyRegion | null>(null);
  const regions = BODY_REGIONS.filter((region) => region.view === view);
  const selectedInView = regions.filter((region) => selectedRegionIds.has(region.id));
  const caption =
    hoveredRegion?.label ??
    (selectedInView.length > 0
      ? selectedInView.map((region) => region.label).join(', ')
      : 'Tap a region to select it');

  return (
    <View style={styles.bodyMapFigure}>
      <Text style={styles.bodyMapTitle}>{title}</Text>
      {createElement(
        'svg',
        {
          width: '100%',
          height: 560,
          viewBox: BODY_MAP_VIEWBOX[view],
          role: 'img',
          'aria-label': `${title} body location map`,
        },
        [
          ...regions.map((region) => {
            const isSelected = selectedRegionIds.has(region.id);
            const isHovered = hoveredRegion?.id === region.id;
            return createElement('path', {
              key: region.id,
              d: region.path,
              transform: region.transform,
              fill: isSelected ? '#A8D5BA' : isHovered ? '#eef7f1' : '#f7faf8',
              fillOpacity: 1,
              stroke: isSelected ? '#1B5E3A' : isHovered ? '#2E7D5C' : '#1f3329',
              strokeWidth: isSelected ? 1.7 : isHovered ? 1.35 : 0.9,
              onClick: () => onToggleRegion(region.id),
              onMouseEnter: () => setHoveredRegion(region),
              onMouseLeave: () => setHoveredRegion(null),
              onFocus: () => setHoveredRegion(region),
              onBlur: () => setHoveredRegion(null),
              onKeyDown: (event: { key?: string; preventDefault?: () => void }) => {
                if (event.key === 'Enter' || event.key === ' ') {
                  event.preventDefault?.();
                  onToggleRegion(region.id);
                }
              },
              tabIndex: 0,
              role: 'button',
              'aria-label': `${isSelected ? 'Remove' : 'Add'} ${region.label} on ${title.toLowerCase()} view`,
              style: { cursor: 'pointer', outline: 'none' },
            });
          }),
        ],
      )}
      <Text style={styles.bodyMapCaption} numberOfLines={2}>{caption}</Text>
    </View>
  );
}

function SymptomPanel({
  title,
  helperText,
  query,
  setQuery,
  searchResults,
  suggestedSymptoms,
  hasMatchedSymptomCandidates,
  trySymptoms,
  selectedSymptoms,
  loadingSymptoms,
  predicting,
  error,
  isWide,
  onAddSymptom,
  onDismissSuggestedSymptom,
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
  const shouldShowMatchedPanel = suggestedSymptoms.length > 0 || !hasMatchedSymptomCandidates;
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
          {shouldShowMatchedPanel ? (
            <View style={styles.suggestedPanel}>
              <Text style={styles.sectionSubtitle}>Matched symptoms</Text>
              {suggestedSymptoms.length === 0 ? (
                <Text style={styles.mutedText}>
                  No strong matches yet. You can still search and add symptoms manually.
                </Text>
              ) : (
                <View style={styles.suggestionList}>
                  {suggestedSymptoms.map((suggestion) => (
                    <View key={`suggested-${suggestion.group.key}`} style={styles.suggestionRow}>
                      <View style={styles.suggestionTextBlock}>
                        <Text style={styles.symptomLabel} numberOfLines={2}>
                          {suggestion.group.label}
                        </Text>
                      </View>
                      <Pressable
                        onPress={() => onDismissSuggestedSymptom(suggestion.group.key)}
                        accessibilityRole="button"
                        accessibilityLabel={`Dismiss ${suggestion.group.label}`}
                        style={({ pressed }) => [
                          styles.suggestionDismissButton,
                          pressed && styles.pressed,
                        ]}
                      >
                        <Text style={styles.suggestionDismissText}>×</Text>
                      </Pressable>
                      <Pressable
                        onPress={() => onAddSymptom(suggestion.group)}
                        style={({ pressed }) => [
                          styles.suggestionAddButton,
                          pressed && styles.pressed,
                        ]}
                      >
                        <Text style={styles.suggestionAddText}>Add</Text>
                      </Pressable>
                    </View>
                  ))}
                </View>
              )}
            </View>
          ) : null}

          <Text style={styles.sectionSubtitle}>Search all symptoms</Text>
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
              <Text style={styles.quickAddTitle}>Suggested symptoms</Text>
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
                  style={({ pressed }) => [styles.clearTextButton, pressed && styles.pressed]}
                >
                  <Text style={styles.clearTextButtonText}>Clear all</Text>
                </Pressable>
              ) : null}
            </View>
          </View>
        </View>
      </View>
    </View>
  );
}

function SelectedSymptomsSidebar({
  selectedSymptoms,
  selectedBodyRegions,
  onStartOver,
}: {
  selectedSymptoms: SymptomGroup[];
  selectedBodyRegions: BodyRegion[];
  onStartOver: () => void;
}) {
  return (
    <View style={styles.panel}>
      <View style={styles.panelHeader}>
        <Text style={styles.panelTitle}>Case Intake</Text>
        <Text style={styles.panelHelp}>
          {selectedSymptoms.length} symptoms and {selectedBodyRegions.length} locations included
        </Text>
      </View>

      {selectedBodyRegions.length > 0 ? (
        <View style={styles.sidebarSection}>
          <Text style={styles.sectionSubtitle}>Body locations</Text>
          <View style={styles.readOnlySelectedList}>
            {selectedBodyRegions.map((region) => (
              <View key={`sidebar-${region.id}`} style={styles.readOnlySymptomItem}>
                <Text style={styles.readOnlySymptomText}>
                  {region.label} ({region.view})
                </Text>
              </View>
            ))}
          </View>
        </View>
      ) : null}

      <View style={styles.sidebarSection}>
        <Text style={styles.sectionSubtitle}>Selected symptoms</Text>
        <View style={styles.readOnlySelectedList}>
          {selectedSymptoms.map((symptom) => (
            <View key={symptom.key} style={styles.readOnlySymptomItem}>
              <Text style={styles.readOnlySymptomText}>
                {symptom.label}
              </Text>
            </View>
          ))}
        </View>
      </View>

      <View style={styles.sidebarActions}>
        <Pressable
          onPress={onStartOver}
          style={({ pressed }) => [styles.stepSecondaryButton, pressed && styles.pressed]}
        >
          <Text style={styles.stepSecondaryButtonText}>Start over</Text>
        </Pressable>
      </View>
    </View>
  );
}

function ResultStage({
  prediction,
  selectedSymptoms,
  selectedBodyRegions,
  step,
  onStepChange,
  onStartOver,
}: {
  prediction: PredictionResponse;
  selectedSymptoms: SymptomGroup[];
  selectedBodyRegions: BodyRegion[];
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
      selectedBodyRegions={selectedBodyRegions}
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
      <View style={[styles.panel, styles.bestSyndromePanel]}>
        <Text style={styles.bestSyndromeEyebrow}>Best Syndrome Match</Text>
        {topSyndrome ? <TopSyndrome syndrome={topSyndrome} /> : null}
      </View>

      <View style={styles.plainPanel}>
        <Text style={styles.panelTitle}>Other Possible Syndromes</Text>
        <View style={styles.otherSyndromeGrid}>
          {visibleOtherSyndromes.map((syndrome) => (
            <OtherSyndromeCard
              key={`${syndrome.syndrome_id}-${syndrome.index}`}
              syndrome={syndrome}
            />
          ))}
        </View>
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
        primaryLabel="Next: Summary"
        onSecondary={onBack}
        onPrimary={onNext}
      />
    </View>
  );
}

function FinalReview({
  prediction,
  selectedSymptoms,
  selectedBodyRegions,
  onBack,
  onStartOver,
}: {
  prediction: PredictionResponse;
  selectedSymptoms: SymptomGroup[];
  selectedBodyRegions: BodyRegion[];
  onBack: () => void;
  onStartOver: () => void;
}) {
  const topSyndrome = prediction.syndromes[0];
  const topHerbs = prediction.herbs.slice(0, 3);
  const keyConcepts = [...prediction.concepts]
    .filter((concept) => concept.id.toLowerCase() !== 'reproductive')
    .sort((a, b) => b.score - a.score)
    .slice(0, 5);

  return (
    <View style={styles.resultsStack}>
      <View style={[styles.panel, styles.reportPanel]}>
        <View style={styles.reportHeader}>
          <View style={styles.reportHeaderText}>
            <Text style={styles.reportEyebrow}>TCMNet</Text>
            <Text style={styles.reportTitle}>Summary Report</Text>
            <Text style={styles.reportMeta}>Generated {reportDateLabel()}</Text>
          </View>
          <Pressable
            onPress={() => downloadPredictionPdf(prediction, selectedSymptoms, selectedBodyRegions)}
            style={({ pressed }) => [styles.reportDownloadButton, pressed && styles.pressed]}
          >
            <Text style={styles.reportDownloadButtonText}>Download PDF</Text>
          </Pressable>
        </View>

        <View style={styles.reportSection}>
          <Text style={styles.reportSectionTitle}>Symptoms</Text>
          <View style={styles.reportList}>
            {selectedSymptoms.map((symptom) => (
              <Text key={symptom.key} style={styles.reportListItem}>
                {symptom.label}
              </Text>
            ))}
          </View>
        </View>

        {selectedBodyRegions.length > 0 ? (
          <View style={styles.reportSection}>
            <Text style={styles.reportSectionTitle}>Body locations</Text>
            <View style={styles.reportInlineList}>
              {selectedBodyRegions.map((region) => (
                <Text key={`report-region-${region.id}`} style={styles.reportInlineItem}>
                  {region.label} ({region.view})
                </Text>
              ))}
            </View>
          </View>
        ) : null}

        <View style={styles.reportSection}>
          <Text style={styles.reportSectionTitle}>Predicted syndrome</Text>
          {topSyndrome ? (
            <>
              <Text style={styles.reportPrimaryName}>{bilingualDisplayName(topSyndrome)}</Text>
              <Text style={styles.reportBodyText}>
                {syndromeDescriptionText(topSyndrome.description)}
              </Text>
              <View style={styles.reportSignalBlock}>
                <Text style={styles.reportMeta}>Key pattern signals</Text>
                <View style={styles.reportInlineList}>
                  {keyConcepts.map((concept) => (
                    <Text key={`report-signal-${concept.id}`} style={styles.reportInlineItem}>
                      {titleCaseLabel(concept.label)}
                    </Text>
                  ))}
                </View>
              </View>
            </>
          ) : null}
        </View>

        <View style={styles.reportSection}>
          <Text style={styles.reportSectionTitle}>Herb recommendations</Text>
          {topHerbs.map((herb, index) => (
            <View key={`review-${herb.herb_id}`} style={styles.reportHerbItem}>
              <Text style={styles.reportHerbName}>
                {index + 1}. {bilingualDisplayName(herb)}
              </Text>
              <Text style={styles.reportBodyText}>{herb.description}</Text>
              {herb.target_concepts.length > 0 ? (
                <Text style={styles.reportMeta}>
                  Targets: {herb.target_concepts.map(titleCaseLabel).join(', ')}
                </Text>
              ) : null}
            </View>
          ))}
        </View>
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
    <View style={styles.topSyndromeContent}>
      <View style={styles.resultText}>
        <Text style={styles.topSyndromeLabel}>{bilingualDisplayName(syndrome)}</Text>
        <Text style={styles.bodyText}>{syndromeDescriptionText(syndrome.description)}</Text>
      </View>
    </View>
  );
}

function OtherSyndromeCard({ syndrome }: { syndrome: SyndromePrediction }) {
  const [showDescription, setShowDescription] = useState(false);
  const flipValue = useRef(new Animated.Value(0)).current;

  useEffect(() => {
    Animated.timing(flipValue, {
      toValue: showDescription ? 1 : 0,
      duration: 280,
      useNativeDriver: true,
    }).start();
  }, [flipValue, showDescription]);

  const frontStyle = {
    opacity: flipValue.interpolate({
      inputRange: [0, 0.45, 1],
      outputRange: [1, 0, 0],
    }),
    transform: [
      {
        rotateY: flipValue.interpolate({
          inputRange: [0, 1],
          outputRange: ['0deg', '180deg'],
        }),
      },
    ],
  };
  const backStyle = {
    opacity: flipValue.interpolate({
      inputRange: [0, 0.55, 1],
      outputRange: [0, 0, 1],
    }),
    transform: [
      {
        rotateY: flipValue.interpolate({
          inputRange: [0, 1],
          outputRange: ['-180deg', '0deg'],
        }),
      },
    ],
  };

  return (
    <Pressable
      onPress={() => setShowDescription((current) => !current)}
      style={({ pressed }) => [
        styles.otherSyndromeCard,
        showDescription && styles.otherSyndromeCardFlipped,
        pressed && styles.pressed,
      ]}
    >
      <Animated.View style={[styles.otherSyndromeFace, frontStyle]}>
        <Text style={styles.otherSyndromeName}>{bilingualDisplayName(syndrome)}</Text>
      </Animated.View>
      <Animated.View style={[styles.otherSyndromeFace, styles.otherSyndromeBackFace, backStyle]}>
        <ScrollView style={styles.otherSyndromeDescriptionScroller}>
          <Text style={styles.otherSyndromeDescription}>
            {syndromeDescriptionText(syndrome.description)}
          </Text>
        </ScrollView>
      </Animated.View>
    </Pressable>
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
    <View style={styles.plainPanel}>
      <Text style={styles.panelTitle}>Why this prediction</Text>
      <Text style={styles.bodyText}>
        TCMNet compares the selected symptoms with learned TCM pattern signals, then
        ranks syndrome matches that best fit those signals. The diagrams below show the
        pattern directions that most shaped this result.
      </Text>

      <View style={styles.radarPair}>
        <View style={styles.radarPairItem}>
          <Text style={[styles.sectionSubtitle, styles.radarHeading]}>
            Eight Principle Signals
          </Text>
          <View style={styles.eightPrinciplesChartLift}>
            <ConceptRadarChart concepts={prediction.concepts} axes={EIGHT_PRINCIPLE_AXES} compact />
          </View>
        </View>

        <View style={styles.radarPairItem}>
          <Text style={[styles.sectionSubtitle, styles.radarHeading]}>Five Element Signals</Text>
          <View style={styles.fiveElementChartLift}>
            <ConceptRadarChart concepts={prediction.concepts} axes={FIVE_ELEMENT_AXES} compact />
          </View>
        </View>
      </View>
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

function downloadPredictionPdf(
  prediction: PredictionResponse,
  selectedSymptoms: SymptomGroup[],
  selectedBodyRegions: BodyRegion[],
) {
  if (typeof document === 'undefined' || typeof URL === 'undefined') {
    return;
  }

  const pdfText = createSummaryReportPdf(prediction, selectedSymptoms, selectedBodyRegions);
  const blob = new Blob([pdfText], { type: 'application/pdf' });
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = `tcmnet-summary-report-${new Date().toISOString().slice(0, 10)}.pdf`;
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  URL.revokeObjectURL(url);
}

function createSummaryReportPdf(
  prediction: PredictionResponse,
  selectedSymptoms: SymptomGroup[],
  selectedBodyRegions: BodyRegion[],
) {
  const topSyndrome = prediction.syndromes[0];
  const topHerbs = prediction.herbs.slice(0, 3);
  const keyConcepts = [...prediction.concepts]
    .filter((concept) => concept.id.toLowerCase() !== 'reproductive')
    .sort((a, b) => b.score - a.score)
    .slice(0, 5)
    .map((concept) => titleCaseLabel(concept.label));
  const content = buildSummaryReportContent({
    symptoms: selectedSymptoms.map((symptom) => symptom.label),
    bodyLocations: selectedBodyRegions.map((region) => `${region.label} (${region.view})`),
    syndromeName: topSyndrome ? englishDisplayName(topSyndrome) : 'No syndrome predicted',
    syndromeDescription: topSyndrome ? syndromeDescriptionText(topSyndrome.description) : '',
    keyConcepts,
    herbs: topHerbs,
    concepts: prediction.concepts,
  });
  return createMultiPagePdf(content);
}

function wrapReportText(text: string, maxLength: number) {
  const words = text.split(/\s+/).filter(Boolean);
  const lines: string[] = [];
  let line = '';

  for (const word of words) {
    const nextLine = line ? `${line} ${word}` : word;
    if (nextLine.length > maxLength && line) {
      lines.push(line);
      line = word;
    } else {
      line = nextLine;
    }
  }

  if (line) {
    lines.push(line);
  }

  return lines;
}

type PdfReportData = {
  symptoms: string[];
  bodyLocations: string[];
  syndromeName: string;
  syndromeDescription: string;
  keyConcepts: string[];
  herbs: HerbRecommendation[];
  concepts: ConceptScore[];
};

function createMultiPagePdf(pageContents: string[]) {
  const objects: string[] = [];
  objects.push('<< /Type /Catalog /Pages 2 0 R >>');
  const pageObjectIds = pageContents.map((_, index) => 3 + index * 2);
  objects.push(
    `<< /Type /Pages /Kids [${pageObjectIds.map((id) => `${id} 0 R`).join(' ')}] /Count ${pageContents.length} >>`,
  );
  pageContents.forEach((content, index) => {
    const pageObjectId = 3 + index * 2;
    const contentObjectId = pageObjectId + 1;
    objects.push(
      `<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 << /Type /Font /Subtype /Type1 /BaseFont /Helvetica >> /F2 << /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold >> >> >> /Contents ${contentObjectId} 0 R >>`,
    );
    objects.push(`<< /Length ${content.length} >>\nstream\n${content}\nendstream`);
  });

  let pdf = '%PDF-1.4\n';
  const offsets = [0];
  objects.forEach((object, index) => {
    offsets.push(pdf.length);
    pdf += `${index + 1} 0 obj\n${object}\nendobj\n`;
  });

  const xrefStart = pdf.length;
  pdf += `xref\n0 ${objects.length + 1}\n0000000000 65535 f \n`;
  for (const offset of offsets.slice(1)) {
    pdf += `${String(offset).padStart(10, '0')} 00000 n \n`;
  }
  pdf += `trailer\n<< /Size ${objects.length + 1} /Root 1 0 R >>\nstartxref\n${xrefStart}\n%%EOF`;
  return pdf;
}

function buildSummaryReportContent(report: PdfReportData) {
  const pages: string[][] = [[]];
  const green = '0.1059 0.3686 0.2275';
  const softGreen = '0.4196 0.7490 0.5412';
  const mutedGreen = '0.4588 0.6157 0.5059';
  const black = '0 0 0';
  let y = 708;

  function currentPage() {
    return pages[pages.length - 1];
  }

  function addPage() {
    pages.push([]);
    y = 720;
  }

  function ensureSpace(height: number) {
    if (y - height < 58) {
      addPage();
    }
  }

  function drawText(text: string, x: number, size: number, bold: boolean, color: string, lineHeight: number) {
    currentPage().push(pdfText(text, x, y, size, bold, color));
    y -= lineHeight;
  }

  function drawSectionTitle(title: string) {
    ensureSpace(40);
    drawText(title, 72, 16, true, green, 26);
  }

  currentPage().push(`${green} rg`, `${green} RG`);
  drawText('TCMNet', 72, 18, true, green, 44);
  drawText('Summary Report', 72, 46, true, green, 34);
  drawText(`Generated ${reportDateLabel()}`, 72, 18, true, black, 42);

  drawSectionTitle('Symptoms');
  wrapReportText(pdfSafeText(report.symptoms.join('  |  ')) || 'No symptoms selected', 88).forEach((line) => {
    drawText(line, 72, 11, false, black, 15);
  });
  y -= 18;

  if (report.bodyLocations.length > 0) {
    drawSectionTitle('Body Locations');
    wrapReportText(pdfSafeText(report.bodyLocations.join('  |  ')), 88).forEach((line) => {
      drawText(line, 72, 11, false, black, 15);
    });
    y -= 18;
  }

  drawSectionTitle('Predicted Syndrome');
  drawText(report.syndromeName, 72, 14, true, green, 20);
  wrapReportText(report.syndromeDescription, 88).forEach((line) => {
    drawText(line, 72, 11, false, black, 15);
  });
  y -= 12;

  ensureSpace(44);
  drawText('Key pattern signals', 72, 12, true, green, 20);
  wrapReportText(report.keyConcepts.join('  |  '), 88).forEach((line) => {
    drawText(line, 72, 12, false, black, 16);
  });
  y -= 18;

  drawSectionTitle('Herb Recommendations');
  report.herbs.slice(0, 3).forEach((herb, index) => {
    const descriptionLines = wrapReportText(herb.description, 88);
    const targetLines = herb.target_concepts.length > 0
      ? wrapReportText(`Targets: ${herb.target_concepts.map(titleCaseLabel).join(', ')}`, 88)
      : [];
    ensureSpace(22 + (descriptionLines.length + targetLines.length) * 14 + 12);
    drawText(`${index + 1}. ${englishDisplayName(herb)}`, 92, 11, true, black, 15);
    descriptionLines.forEach((line) => {
      drawText(line, 116, 11, false, black, 14);
    });
    targetLines.forEach((line) => {
      drawText(line, 116, 11, false, black, 14);
    });
    y -= 8;
  });

  ensureSpace(170);
  const chartTitleY = y;
  currentPage().push(pdfText('Eight Principle Signals', 124, chartTitleY, 12, true, green));
  currentPage().push(pdfText('Five Element Signals', 348, chartTitleY, 12, true, green));
  currentPage().push(
    buildPdfRadarChart(report.concepts, EIGHT_PRINCIPLE_AXES, 190, chartTitleY - 82, 48, green, softGreen, mutedGreen),
  );
  currentPage().push(
    buildPdfRadarChart(report.concepts, FIVE_ELEMENT_AXES, 410, chartTitleY - 82, 48, green, softGreen, mutedGreen),
  );

  return pages.map((page) => page.join('\n'));
}

function pdfText(text: string, x: number, y: number, size: number, bold: boolean, color: string) {
  const font = bold ? 'F2' : 'F1';
  return `BT\n${color} rg\n/${font} ${size} Tf\n${x} ${y} Td\n(${escapePdfText(pdfSafeText(text))}) Tj\nET`;
}

function buildPdfRadarChart(
  concepts: ConceptScore[],
  axes: ConceptAxis[],
  centerX: number,
  centerY: number,
  radius: number,
  green: string,
  softGreen: string,
  mutedGreen: string,
) {
  const scoreByKey = new Map<string, number>();
  concepts.forEach((concept) => {
    scoreByKey.set(concept.id.toLowerCase(), concept.score);
    scoreByKey.set(concept.label.toLowerCase(), concept.score);
  });
  const points = axes.map((axis, index) => {
    const angle = Math.PI / 2 - (2 * Math.PI * index) / axes.length;
    const score = Math.max(
      0,
      Math.min(scoreByKey.get((axis.sourceKey || axis.key).toLowerCase()) ?? 0, 1),
    );
    return {
      axis,
      x: centerX + Math.cos(angle) * radius * score,
      y: centerY + Math.sin(angle) * radius * score,
      axisX: centerX + Math.cos(angle) * radius,
      axisY: centerY + Math.sin(angle) * radius,
      labelX: centerX + Math.cos(angle) * (radius + 16),
      labelY: centerY + Math.sin(angle) * (radius + 16),
    };
  });
  const grid = [0.33, 0.66, 1]
    .map((level) => {
      const gridPoints = axes.map((_, index) => {
        const angle = Math.PI / 2 - (2 * Math.PI * index) / axes.length;
        return [centerX + Math.cos(angle) * radius * level, centerY + Math.sin(angle) * radius * level];
      });
      return `${mutedGreen} RG 0.5 w ${pdfPolygonPath(gridPoints)} S`;
    })
    .join('\n');
  const spokes = points
    .map((point) => `${mutedGreen} RG 0.4 w ${centerX} ${centerY} m ${point.axisX} ${point.axisY} l S`)
    .join('\n');
  const areaPoints = points.map((point) => [point.x, point.y]);
  const labels = points
    .map((point) => {
      const anchorOffset = point.labelX < centerX - 4 ? -estimatePdfLabelWidth(point.axis.label) : point.labelX > centerX + 4 ? 0 : -estimatePdfLabelWidth(point.axis.label) / 2;
      return pdfText(point.axis.label, point.labelX + anchorOffset, point.labelY - 3, 7, true, green);
    })
    .join('\n');

  return [
    grid,
    spokes,
    `${softGreen} rg ${green} RG 1 w ${pdfPolygonPath(areaPoints)} B`,
    ...points.map(
      (point) =>
        `${green} rg ${formatPdfNumber(point.x - 2)} ${formatPdfNumber(point.y - 2)} 4 4 re f`,
    ),
    labels,
  ].join('\n');
}

function pdfPolygonPath(points: number[][]) {
  return points
    .map(([x, y], index) => `${formatPdfNumber(x)} ${formatPdfNumber(y)} ${index === 0 ? 'm' : 'l'}`)
    .join(' ') + ' h';
}

function estimatePdfLabelWidth(text: string) {
  return text.length * 4.2;
}

function formatPdfNumber(value: number) {
  return Number(value.toFixed(2));
}

function pdfSafeText(text: string) {
  return text
    .normalize('NFKD')
    .replace(/[^\x20-\x7E]/g, '')
    .replace(/\s+/g, ' ')
    .trim();
}

function escapePdfText(text: string) {
  return text.replace(/\\/g, '\\\\').replace(/\(/g, '\\(').replace(/\)/g, '\\)');
}

function ConceptRadarChart({
  concepts,
  axes,
  compact,
}: {
  concepts: ConceptScore[];
  axes: ConceptAxis[];
  compact?: boolean;
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

  const size = compact ? 220 : 280;
  const viewWidth = compact ? 430 : 720;
  const viewHeight = compact ? 330 : 480;
  const center = size / 2;
  const offsetX = (viewWidth - size) / 2;
  const offsetY = (viewHeight - size) / 2;
  const chartCenterX = offsetX + center;
  const chartCenterY = offsetY + center;
  const radius = compact ? 72 : 96;
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
      labelWidth: estimateSvgTextWidth(axis.label, 12, 800),
      x: chartCenterX + Math.cos(angle) * radius * score,
      y: chartCenterY + Math.sin(angle) * radius * score,
      labelX: chartCenterX + Math.cos(angle) * (radius + (compact ? 24 : 30)),
      labelY: chartCenterY + Math.sin(angle) * (radius + (compact ? 24 : 30)),
      axisX: chartCenterX + Math.cos(angle) * radius,
      axisY: chartCenterY + Math.sin(angle) * radius,
    };
  });
  const polygonPoints = chartPoints.map((point) => `${point.x},${point.y}`).join(' ');
  return (
    <View style={styles.radarCard}>
      {createElement(
        'svg',
        {
          width: '100%',
          height: viewHeight,
          viewBox: `0 0 ${viewWidth} ${viewHeight}`,
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
                  return `${chartCenterX + Math.cos(angle) * radius * level},${
                    chartCenterY + Math.sin(angle) * radius * level
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
              x1: chartCenterX,
              y1: chartCenterY,
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
                textAnchor: point.labelX < chartCenterX - 8 ? 'end' : point.labelX > chartCenterX + 8 ? 'start' : 'middle',
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
      <View style={styles.radarTooltipSlot}>
        {hoveredAxis ? (
          <View style={styles.radarTooltipPanel}>
            <Text style={styles.radarTooltipTitle}>{hoveredAxis.label}</Text>
            <Text style={styles.radarTooltipBody}>{hoveredAxis.description}</Text>
          </View>
        ) : null}
      </View>
    </View>
  );
}

function estimateSvgTextWidth(text: string, fontSize: number, fontWeight: number) {
  const knownWidths: Record<string, number> = {
    Yin: 17,
    Yang: 26,
    Internal: 42,
    External: 44,
    Cold: 25,
    Heat: 24,
    Deficiency: 53,
    Excess: 34,
    Wood: 31,
    Water: 32,
    Fire: 21,
    Metal: 30,
    Earth: 30,
  };
  if (knownWidths[text]) {
    return knownWidths[text];
  }
  const weightFactor = fontWeight >= 700 ? 0.52 : 0.48;
  return text.length * fontSize * weightFactor;
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
const sansSerifFont = 'Jakarta Sans, Arial, Helvetica, sans-serif';

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
    borderWidth: 1,
    borderColor: 'transparent',
    borderRadius: 8,
    backgroundColor: colors.softPanel,
    padding: 22,
    justifyContent: 'flex-start',
    gap: 10,
  },
  aboutCardHovered: {
    backgroundColor: '#e4f0e9',
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
  stepperWrap: {
    width: '100%',
    gap: 7,
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
  progressWrap: {
    width: '100%',
    maxWidth: 520,
    alignSelf: 'center',
    flexDirection: 'row',
    alignItems: 'center',
    gap: 9,
  },
  progressTrack: {
    flex: 1,
    height: 4,
    borderRadius: 999,
    backgroundColor: '#e8f0eb',
    overflow: 'hidden',
  },
  progressFill: {
    height: '100%',
    borderRadius: 999,
    backgroundColor: colors.deepGreen,
  },
  progressText: {
    width: 38,
    color: colors.muted,
    fontSize: 12,
    lineHeight: 16,
    fontWeight: '900',
    textAlign: 'right',
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
  intakeStack: {
    width: '100%',
    maxWidth: 860,
    gap: 18,
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
    width: 280,
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
  plainPanel: {
    width: '100%',
    backgroundColor: 'transparent',
    paddingVertical: 4,
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
  reportPanel: {
    padding: 24,
    gap: 0,
  },
  reportHeader: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    alignItems: 'flex-start',
    justifyContent: 'space-between',
    gap: 16,
    borderBottomWidth: 1,
    borderBottomColor: colors.line,
    paddingBottom: 18,
    marginBottom: 2,
  },
  reportHeaderText: {
    flex: 1,
    minWidth: 240,
    gap: 3,
  },
  reportEyebrow: {
    color: colors.deepGreen,
    fontSize: 12,
    lineHeight: 16,
    fontWeight: '900',
  },
  reportTitle: {
    color: colors.text,
    fontFamily: serifFont,
    fontSize: 32,
    lineHeight: 38,
    fontWeight: '700',
  },
  reportDownloadButton: {
    minHeight: 36,
    borderRadius: 999,
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: colors.deepGreen,
    paddingHorizontal: 14,
    paddingVertical: 7,
  },
  reportDownloadButtonText: {
    color: '#ffffff',
    fontSize: 13,
    lineHeight: 17,
    fontWeight: '900',
  },
  reportSection: {
    borderBottomWidth: 1,
    borderBottomColor: '#edf4ef',
    paddingVertical: 18,
    gap: 9,
  },
  reportSectionTitle: {
    color: colors.deepGreen,
    fontSize: 16,
    lineHeight: 21,
    fontWeight: '900',
  },
  reportPrimaryName: {
    color: colors.text,
    fontFamily: serifFont,
    fontSize: 22,
    lineHeight: 28,
    fontWeight: '700',
  },
  reportBodyText: {
    color: colors.text,
    fontSize: 14,
    lineHeight: 21,
  },
  reportList: {
    gap: 6,
  },
  reportListItem: {
    color: colors.text,
    fontSize: 14,
    lineHeight: 20,
  },
  reportInlineList: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 8,
  },
  reportSignalBlock: {
    gap: 7,
    paddingTop: 4,
  },
  reportInlineItem: {
    color: colors.text,
    borderWidth: 1,
    borderColor: '#c7dfcf',
    backgroundColor: '#f1f8f4',
    borderRadius: 999,
    paddingHorizontal: 10,
    paddingVertical: 5,
    fontSize: 13,
    lineHeight: 17,
    fontWeight: '800',
  },
  reportHerbItem: {
    gap: 5,
    paddingTop: 4,
  },
  reportHerbName: {
    color: colors.text,
    fontSize: 15,
    lineHeight: 20,
    fontWeight: '900',
  },
  reportMeta: {
    color: colors.muted,
    fontSize: 12,
    lineHeight: 17,
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
  formStack: {
    gap: 18,
  },
  formField: {
    gap: 9,
  },
  formLabel: {
    color: colors.text,
    fontSize: 14,
    lineHeight: 19,
    fontWeight: '900',
  },
  formInput: {
    minHeight: 48,
    borderWidth: 1,
    borderColor: colors.line,
    borderRadius: 8,
    paddingHorizontal: 14,
    fontSize: 16,
    backgroundColor: '#fbfdfb',
    color: colors.text,
  },
  formHelp: {
    color: colors.muted,
    fontSize: 13,
    lineHeight: 18,
  },
  optionGrid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 8,
  },
  optionChip: {
    borderWidth: 1,
    borderColor: '#c7dfcf',
    borderRadius: 999,
    backgroundColor: '#fbfdfb',
    paddingHorizontal: 13,
    paddingVertical: 9,
  },
  optionChipSelected: {
    borderColor: colors.deepGreen,
    backgroundColor: '#f1f8f4',
  },
  optionChipText: {
    color: colors.text,
    fontSize: 13,
    lineHeight: 17,
    fontWeight: '800',
  },
  optionChipTextSelected: {
    color: colors.deepGreen,
    fontWeight: '900',
  },
  checkboxRow: {
    alignSelf: 'flex-start',
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
    paddingVertical: 3,
  },
  checkboxBox: {
    width: 18,
    height: 18,
    borderRadius: 4,
    borderWidth: 1,
    borderColor: colors.line,
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: colors.panel,
  },
  checkboxBoxChecked: {
    borderColor: colors.deepGreen,
    backgroundColor: colors.deepGreen,
  },
  checkboxMark: {
    color: '#ffffff',
    fontSize: 12,
    lineHeight: 14,
    fontWeight: '900',
  },
  checkboxText: {
    color: colors.text,
    fontSize: 13,
    lineHeight: 18,
    fontWeight: '800',
  },
  sliderHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    gap: 12,
  },
  sliderValue: {
    color: colors.deepGreen,
    fontSize: 13,
    lineHeight: 18,
    fontWeight: '900',
  },
  sliderShell: {
    gap: 5,
  },
  sliderLabels: {
    flexDirection: 'row',
    justifyContent: 'space-between',
  },
  sliderEndpoint: {
    color: colors.muted,
    fontSize: 12,
    lineHeight: 16,
    fontWeight: '800',
  },
  sliderHelp: {
    color: colors.text,
    fontSize: 13,
    lineHeight: 18,
    fontWeight: '800',
  },
  severityControl: {
    gap: 6,
  },
  severityTrack: {
    width: '100%',
    height: 10,
    borderRadius: 999,
    overflow: 'visible',
    position: 'relative',
  },
  severityKnob: {
    position: 'absolute',
    top: -6,
    width: 22,
    height: 22,
    marginLeft: -11,
    borderRadius: 999,
    borderWidth: 3,
    borderColor: colors.panel,
    shadowColor: '#0f2a1f',
    shadowOpacity: 0.16,
    shadowRadius: 8,
    shadowOffset: { width: 0, height: 3 },
  },
  severityScaleLabels: {
    flexDirection: 'row',
    justifyContent: 'space-between',
  },
  severityScaleText: {
    color: colors.muted,
    fontFamily: serifFont,
    fontSize: 13,
    lineHeight: 17,
    fontWeight: '700',
  },
  bodyMapGrid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    justifyContent: 'center',
    gap: 18,
  },
  bodyMapFigure: {
    flexBasis: 260,
    flexGrow: 1,
    maxWidth: 350,
    minWidth: 240,
    alignItems: 'center',
    borderWidth: 1,
    borderColor: colors.line,
    borderRadius: 8,
    backgroundColor: '#fbfdfb',
    paddingHorizontal: 12,
    paddingTop: 12,
    paddingBottom: 10,
  },
  bodyMapTitle: {
    color: colors.text,
    fontSize: 13,
    lineHeight: 18,
    fontWeight: '900',
    marginBottom: 6,
  },
  bodyMapCaption: {
    minHeight: 34,
    color: colors.deepGreen,
    fontSize: 12,
    lineHeight: 17,
    fontWeight: '900',
    textAlign: 'center',
    paddingHorizontal: 8,
    marginTop: -4,
  },
  bodyMapSource: {
    color: colors.muted,
    fontSize: 11,
    lineHeight: 15,
    textAlign: 'center',
    marginTop: -4,
  },
  locationLog: {
    gap: 10,
    borderTopWidth: 1,
    borderTopColor: '#edf4ef',
    paddingTop: 14,
  },
  locationClearButton: {
    alignSelf: 'flex-start',
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
  suggestedPanel: {
    gap: 10,
    borderWidth: 1,
    borderColor: '#d8e8de',
    borderRadius: 8,
    backgroundColor: '#f8fcfa',
    padding: 12,
  },
  suggestionList: {
    gap: 8,
  },
  suggestionRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 10,
    borderWidth: 1,
    borderColor: '#e3efe7',
    borderRadius: 8,
    backgroundColor: colors.panel,
    paddingHorizontal: 12,
    paddingVertical: 10,
  },
  suggestionTextBlock: {
    flex: 1,
    minWidth: 0,
  },
  suggestionAddButton: {
    borderRadius: 8,
    backgroundColor: colors.deepGreen,
    paddingHorizontal: 12,
    paddingVertical: 8,
  },
  suggestionAddText: {
    color: '#fff',
    fontSize: 13,
    fontWeight: '900',
  },
  suggestionDismissButton: {
    width: 32,
    height: 32,
    borderRadius: 16,
    alignItems: 'center',
    justifyContent: 'center',
    borderWidth: 1,
    borderColor: '#d8e8de',
    backgroundColor: '#f8fcfa',
  },
  suggestionDismissText: {
    color: colors.muted,
    fontSize: 20,
    fontWeight: '800',
    lineHeight: 22,
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
  readOnlySelectedList: {
    gap: 8,
  },
  sidebarSection: {
    gap: 8,
  },
  readOnlySymptomItem: {
    borderBottomWidth: 1,
    borderBottomColor: '#edf4ef',
    paddingBottom: 8,
  },
  readOnlySymptomText: {
    color: colors.text,
    fontSize: 13,
    lineHeight: 18,
    fontWeight: '800',
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
  sidebarActions: {
    flexDirection: 'row',
    justifyContent: 'flex-start',
    paddingTop: 2,
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
    minHeight: 38,
    minWidth: 154,
    borderRadius: 999,
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: colors.deepGreen,
    borderWidth: 1,
    borderColor: colors.deepGreen,
    paddingHorizontal: 14,
    paddingVertical: 8,
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
    fontSize: 13,
    lineHeight: 17,
    fontWeight: '900',
    textAlign: 'center',
  },
  clearTextButton: {
    minHeight: 38,
    alignItems: 'center',
    justifyContent: 'center',
    paddingHorizontal: 6,
  },
  clearTextButtonText: {
    color: colors.text,
    fontSize: 13,
    lineHeight: 17,
    fontWeight: '800',
    textDecorationLine: 'underline',
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
  bestSyndromePanel: {
    backgroundColor: '#f1f8f4',
    borderColor: '#c7dfcf',
  },
  bestSyndromeEyebrow: {
    color: colors.deepGreen,
    fontSize: 13,
    lineHeight: 17,
    fontWeight: '900',
  },
  topSyndromeContent: {
    gap: 10,
  },
  topSyndromeLabel: {
    color: colors.text,
    fontFamily: serifFont,
    fontSize: 24,
    lineHeight: 30,
    fontWeight: '700',
  },
  otherSyndromeGrid: {
    flexDirection: 'row',
    gap: 12,
    flexWrap: 'wrap',
  },
  otherSyndromeCard: {
    flexBasis: 190,
    flexGrow: 1,
    minHeight: 132,
    borderWidth: 1,
    borderColor: '#c7dfcf',
    borderRadius: 8,
    backgroundColor: '#fbfdfb',
    padding: 14,
    alignItems: 'center',
    justifyContent: 'center',
    overflow: 'hidden',
    position: 'relative',
  },
  otherSyndromeCardFlipped: {
    alignItems: 'flex-start',
    justifyContent: 'flex-start',
    backgroundColor: '#f1f8f4',
  },
  otherSyndromeFace: {
    ...StyleSheet.absoluteFillObject,
    alignItems: 'center',
    justifyContent: 'center',
    padding: 14,
    backfaceVisibility: 'hidden',
  },
  otherSyndromeBackFace: {
    alignItems: 'flex-start',
    justifyContent: 'flex-start',
  },
  otherSyndromeDescriptionScroller: {
    width: '100%',
    maxHeight: '100%',
  },
  otherSyndromeName: {
    color: colors.text,
    fontFamily: serifFont,
    fontSize: 18,
    lineHeight: 24,
    fontWeight: '700',
    textAlign: 'center',
  },
  otherSyndromeDescription: {
    color: colors.text,
    fontSize: 13,
    lineHeight: 19,
    fontWeight: '400',
  },
  radarHeading: {
    textAlign: 'center',
    alignSelf: 'center',
    width: '100%',
  },
  radarPair: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    alignItems: 'flex-start',
    justifyContent: 'center',
    gap: 12,
    marginTop: 8,
  },
  radarPairItem: {
    flexBasis: 360,
    flexGrow: 1,
    maxWidth: 500,
    minWidth: 300,
  },
  eightPrinciplesHeading: {
    marginTop: 20,
  },
  eightPrinciplesChartLift: {
    marginTop: -10,
  },
  fiveElementChartLift: {
    marginTop: -10,
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
  radarCard: {
    width: '100%',
    alignItems: 'center',
    paddingVertical: 4,
  },
  radarTooltipSlot: {
    width: '100%',
    minHeight: 72,
    alignItems: 'center',
    marginTop: -8,
  },
  radarTooltipPanel: {
    width: '92%',
    maxWidth: 360,
    borderLeftWidth: 2,
    borderLeftColor: colors.forest,
    paddingLeft: 10,
    paddingRight: 8,
    paddingVertical: 4,
  },
  radarTooltipTitle: {
    color: colors.deepGreen,
    fontFamily: serifFont,
    fontSize: 14,
    lineHeight: 18,
    fontWeight: '700',
  },
  radarTooltipBody: {
    color: '#111111',
    fontFamily: sansSerifFont,
    fontSize: 12,
    lineHeight: 17,
    fontWeight: '400',
    marginTop: 2,
  },
});

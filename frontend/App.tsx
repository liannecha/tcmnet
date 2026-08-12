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
} from 'react-native';

import { fetchSymptoms, predict } from './src/api';
import type { MetadataRecord, PredictionResponse } from './src/types';

const MAX_VISIBLE_SYMPTOMS = 60;

function formatPercent(value: number) {
  return `${Math.round(value * 1000) / 10}%`;
}

function formatScore(value: number) {
  return value.toFixed(3);
}

export default function App() {
  const [symptoms, setSymptoms] = useState<MetadataRecord[]>([]);
  const [selectedSymptoms, setSelectedSymptoms] = useState<MetadataRecord[]>([]);
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

  const selectedIds = useMemo(
    () => new Set(selectedSymptoms.map((symptom) => symptom.id)),
    [selectedSymptoms],
  );

  const filteredSymptoms = useMemo(() => {
    const normalizedQuery = query.trim().toLowerCase();
    const source = normalizedQuery
      ? symptoms.filter((symptom) => {
          const label = symptom.label.toLowerCase();
          const id = symptom.id.toLowerCase();
          return label.includes(normalizedQuery) || id.includes(normalizedQuery);
        })
      : symptoms;

    return source
      .filter((symptom) => !selectedIds.has(symptom.id))
      .slice(0, MAX_VISIBLE_SYMPTOMS);
  }, [query, selectedIds, symptoms]);

  function addSymptom(symptom: MetadataRecord) {
    setSelectedSymptoms((current) =>
      current.some((item) => item.id === symptom.id) ? current : [...current, symptom],
    );
    setQuery('');
    setPrediction(null);
  }

  function removeSymptom(symptomId: string) {
    setSelectedSymptoms((current) => current.filter((symptom) => symptom.id !== symptomId));
    setPrediction(null);
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
        symptom_ids: selectedSymptoms.map((symptom) => symptom.id),
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
    <SafeAreaView style={styles.safeArea}>
      <StatusBar style="dark" />
      <ScrollView contentContainerStyle={styles.page}>
        <View style={styles.header}>
          <Text style={styles.title}>TCMNet</Text>
          <Text style={styles.subtitle}>
            Concept-guided syndrome prediction and herb recommendation for research use.
          </Text>
          <Text style={styles.disclaimer}>
            Educational prototype only. This tool is not medical advice and should not be
            used for diagnosis or treatment decisions.
          </Text>
        </View>

        <View style={styles.panel}>
          <Text style={styles.sectionTitle}>Symptoms</Text>
          <TextInput
            value={query}
            onChangeText={setQuery}
            placeholder="Search symptom label or ID"
            autoCapitalize="none"
            autoCorrect={false}
            style={styles.searchInput}
          />

          {loadingSymptoms ? (
            <View style={styles.inlineState}>
              <ActivityIndicator />
              <Text style={styles.mutedText}>Loading symptoms...</Text>
            </View>
          ) : (
            <View style={styles.symptomList}>
              {filteredSymptoms.map((symptom) => (
                <Pressable
                  key={symptom.id}
                  onPress={() => addSymptom(symptom)}
                  style={({ pressed }) => [
                    styles.symptomRow,
                    pressed && styles.pressed,
                  ]}
                >
                  <Text style={styles.symptomLabel}>{symptom.label}</Text>
                  <Text style={styles.symptomId}>{symptom.id}</Text>
                </Pressable>
              ))}
            </View>
          )}

          <View style={styles.selectedHeader}>
            <Text style={styles.sectionSubtitle}>Selected</Text>
            <Text style={styles.countText}>{selectedSymptoms.length}</Text>
          </View>

          {selectedSymptoms.length === 0 ? (
            <Text style={styles.mutedText}>No symptoms selected yet.</Text>
          ) : (
            <View style={styles.chips}>
              {selectedSymptoms.map((symptom) => (
                <Pressable
                  key={symptom.id}
                  onPress={() => removeSymptom(symptom.id)}
                  style={({ pressed }) => [styles.chip, pressed && styles.pressed]}
                >
                  <Text style={styles.chipText}>{symptom.label}</Text>
                  <Text style={styles.chipId}>{symptom.id}</Text>
                </Pressable>
              ))}
            </View>
          )}

          <Pressable
            onPress={runPrediction}
            disabled={predicting || selectedSymptoms.length === 0}
            style={({ pressed }) => [
              styles.predictButton,
              (predicting || selectedSymptoms.length === 0) && styles.disabledButton,
              pressed && styles.pressed,
            ]}
          >
            {predicting ? (
              <ActivityIndicator color="#fff" />
            ) : (
              <Text style={styles.predictButtonText}>Predict</Text>
            )}
          </Pressable>
        </View>

        {error ? <Text style={styles.errorText}>{error}</Text> : null}

        {prediction ? <Results prediction={prediction} /> : null}
      </ScrollView>
    </SafeAreaView>
  );
}

function Results({ prediction }: { prediction: PredictionResponse }) {
  return (
    <View style={styles.results}>
      <View style={styles.panel}>
        <Text style={styles.sectionTitle}>Prediction Results</Text>
        <Text style={styles.sectionSubtitle}>Known Symptom IDs</Text>
        <Text style={styles.bodyText}>
          {prediction.input.known_symptom_ids.length
            ? prediction.input.known_symptom_ids.join(', ')
            : 'None'}
        </Text>

        {prediction.input.unknown_symptom_ids.length ? (
          <>
            <Text style={styles.sectionSubtitle}>Unknown Symptom IDs</Text>
            <Text style={styles.warningText}>
              {prediction.input.unknown_symptom_ids.join(', ')}
            </Text>
          </>
        ) : null}
      </View>

      <View style={styles.panel}>
        <Text style={styles.sectionTitle}>Top Syndromes</Text>
        {prediction.syndromes.map((syndrome) => (
          <View key={`${syndrome.syndrome_id}-${syndrome.index}`} style={styles.resultRow}>
            <View style={styles.resultText}>
              <Text style={styles.resultLabel}>{syndrome.label}</Text>
              <Text style={styles.resultId}>{syndrome.syndrome_id}</Text>
            </View>
            <Text style={styles.metric}>{formatPercent(syndrome.confidence)}</Text>
          </View>
        ))}
      </View>

      <View style={styles.panel}>
        <Text style={styles.sectionTitle}>Concept Scores</Text>
        <View style={styles.conceptGrid}>
          {prediction.concepts.map((concept) => (
            <View key={concept.id} style={styles.conceptItem}>
              <Text style={styles.conceptLabel}>{concept.label}</Text>
              <Text style={styles.metric}>{formatScore(concept.score)}</Text>
            </View>
          ))}
        </View>
      </View>

      <View style={styles.panel}>
        <Text style={styles.sectionTitle}>Herb Recommendations</Text>
        {prediction.herbs.map((herb) => (
          <View key={herb.herb_id} style={styles.resultRow}>
            <View style={styles.resultText}>
              <Text style={styles.resultLabel}>{herb.label}</Text>
              <Text style={styles.resultId}>{herb.herb_id}</Text>
            </View>
            <Text style={styles.metric}>{formatScore(herb.score)}</Text>
          </View>
        ))}
      </View>

      <View style={styles.panel}>
        <Text style={styles.sectionTitle}>Explanation</Text>

        <Text style={styles.sectionSubtitle}>Matching Symptoms</Text>
        <View style={styles.explanationList}>
          {prediction.explanation.matching_symptoms.map((symptom) => (
            <Text key={symptom.id} style={styles.bodyText}>
              {symptom.label} ({symptom.id})
            </Text>
          ))}
        </View>

        <Text style={styles.sectionSubtitle}>Concept Alignment</Text>
        <View style={styles.conceptGrid}>
          {prediction.explanation.concept_alignment.map((concept) => (
            <View key={`explain-${concept.id}`} style={styles.conceptItem}>
              <Text style={styles.conceptLabel}>{concept.label}</Text>
              <Text style={styles.metric}>{formatScore(concept.score)}</Text>
            </View>
          ))}
        </View>

        <Text style={styles.sectionSubtitle}>Known Syndrome-Herb Associations</Text>
        <Text style={styles.bodyText}>
          {prediction.explanation.syndrome_herb_associations.label} (
          {prediction.explanation.syndrome_herb_associations.syndrome_id}) has{' '}
          {prediction.explanation.syndrome_herb_associations.total_associated_herbs}{' '}
          known herb associations in the frozen artifact.
        </Text>
        <View style={styles.chips}>
          {prediction.explanation.syndrome_herb_associations.associated_herbs.map(
            (herb) => (
              <View key={`assoc-${herb.id}`} style={styles.associationChip}>
                <Text style={styles.chipText}>{herb.label}</Text>
                <Text style={styles.chipId}>{herb.id}</Text>
              </View>
            ),
          )}
        </View>

        <Text style={styles.sectionSubtitle}>Herb Ranking Scores</Text>
        <Text style={styles.bodyText}>
          {prediction.explanation.herb_ranking.formula}; alpha ={' '}
          {formatScore(prediction.explanation.herb_ranking.alpha)}.
        </Text>
        {prediction.explanation.herb_ranking.items.map((item) => (
          <View key={`rank-${item.herb_id}`} style={styles.explanationRow}>
            <View style={styles.resultText}>
              <Text style={styles.resultLabel}>{item.label}</Text>
              <Text style={styles.resultId}>{item.herb_id}</Text>
              <Text style={styles.mutedText}>
                concept {formatScore(item.concept_similarity)} · prior{' '}
                {formatScore(item.syndrome_prior)} ·{' '}
                {item.known_for_predicted_syndrome ? 'known association' : 'not linked'}
              </Text>
            </View>
            <Text style={styles.metric}>{formatScore(item.score)}</Text>
          </View>
        ))}
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  safeArea: {
    flex: 1,
    backgroundColor: '#f6f3ee',
  },
  page: {
    width: '100%',
    maxWidth: 980,
    alignSelf: 'center',
    padding: 20,
    gap: 16,
  },
  header: {
    gap: 8,
    paddingTop: 12,
  },
  title: {
    fontSize: 34,
    fontWeight: '800',
    color: '#193a37',
  },
  subtitle: {
    fontSize: 17,
    lineHeight: 24,
    color: '#38504d',
  },
  disclaimer: {
    fontSize: 13,
    lineHeight: 19,
    color: '#6c5d4e',
  },
  panel: {
    backgroundColor: '#fffdf9',
    borderColor: '#ded6ca',
    borderWidth: 1,
    borderRadius: 8,
    padding: 16,
    gap: 12,
  },
  sectionTitle: {
    fontSize: 20,
    fontWeight: '700',
    color: '#193a37',
  },
  sectionSubtitle: {
    fontSize: 14,
    fontWeight: '700',
    color: '#38504d',
  },
  searchInput: {
    minHeight: 46,
    borderWidth: 1,
    borderColor: '#c8d0c6',
    borderRadius: 8,
    paddingHorizontal: 12,
    fontSize: 16,
    backgroundColor: '#ffffff',
    color: '#17211f',
  },
  inlineState: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 10,
    minHeight: 48,
  },
  mutedText: {
    color: '#6f7a76',
    fontSize: 14,
  },
  symptomList: {
    borderWidth: 1,
    borderColor: '#e3e0da',
    borderRadius: 8,
    overflow: 'hidden',
  },
  symptomRow: {
    minHeight: 48,
    paddingHorizontal: 12,
    paddingVertical: 9,
    borderBottomWidth: 1,
    borderBottomColor: '#eeebe5',
    backgroundColor: '#ffffff',
  },
  symptomLabel: {
    color: '#17211f',
    fontSize: 15,
    fontWeight: '600',
  },
  symptomId: {
    color: '#6f7a76',
    fontSize: 12,
    marginTop: 2,
  },
  selectedHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
  },
  countText: {
    color: '#47635f',
    fontWeight: '700',
  },
  chips: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 8,
  },
  chip: {
    borderRadius: 8,
    borderWidth: 1,
    borderColor: '#91b0a9',
    backgroundColor: '#eef7f4',
    paddingHorizontal: 10,
    paddingVertical: 8,
    maxWidth: '100%',
  },
  chipText: {
    color: '#183f39',
    fontWeight: '700',
    fontSize: 13,
  },
  chipId: {
    color: '#52736d',
    fontSize: 11,
    marginTop: 1,
  },
  predictButton: {
    minHeight: 48,
    borderRadius: 8,
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: '#235c54',
  },
  disabledButton: {
    backgroundColor: '#9aa8a5',
  },
  predictButtonText: {
    color: '#ffffff',
    fontSize: 16,
    fontWeight: '800',
  },
  pressed: {
    opacity: 0.72,
  },
  errorText: {
    color: '#9d2a22',
    backgroundColor: '#fff0ee',
    borderColor: '#efc4be',
    borderWidth: 1,
    borderRadius: 8,
    padding: 12,
    fontSize: 14,
  },
  warningText: {
    color: '#945600',
    fontSize: 14,
  },
  results: {
    gap: 16,
  },
  bodyText: {
    color: '#17211f',
    fontSize: 14,
    lineHeight: 20,
  },
  resultRow: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    gap: 12,
    borderTopWidth: 1,
    borderTopColor: '#eee9e1',
    paddingTop: 10,
  },
  explanationRow: {
    flexDirection: 'row',
    alignItems: 'flex-start',
    justifyContent: 'space-between',
    gap: 12,
    borderTopWidth: 1,
    borderTopColor: '#eee9e1',
    paddingTop: 10,
  },
  resultText: {
    flex: 1,
    minWidth: 0,
  },
  resultLabel: {
    color: '#17211f',
    fontSize: 15,
    fontWeight: '700',
  },
  resultId: {
    color: '#6f7a76',
    fontSize: 12,
    marginTop: 2,
  },
  metric: {
    color: '#235c54',
    fontSize: 14,
    fontWeight: '800',
  },
  conceptGrid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 8,
  },
  conceptItem: {
    width: 130,
    borderRadius: 8,
    borderWidth: 1,
    borderColor: '#e3e0da',
    backgroundColor: '#ffffff',
    padding: 10,
    gap: 4,
  },
  conceptLabel: {
    color: '#38504d',
    fontSize: 13,
    fontWeight: '700',
  },
  explanationList: {
    gap: 6,
  },
  associationChip: {
    borderRadius: 8,
    borderWidth: 1,
    borderColor: '#d8d1c6',
    backgroundColor: '#ffffff',
    paddingHorizontal: 10,
    paddingVertical: 8,
    maxWidth: '100%',
  },
});

export type SymptomSynonymEntry = {
  canonicalLabel: string;
  patientPhrases: string[];
  clinicalPhrases: string[];
  relatedTerms: string[];
  bodyLocationHints: string[];
  reasonTemplate: string;
};

export type SearchableSymptomPhrase = {
  phrase: string;
  normalizedPhrase: string;
  canonicalLabel: string;
  source: 'canonical' | 'patient' | 'clinical' | 'related' | 'bodyLocation';
};

// Curated starter dictionary for mapping everyday intake language to TCMNet
// symptom labels. Canonical labels are hand-aligned to the current TCMNet
// symptom vocabulary and remain the source of truth.
//
// Patient-friendly wording is inspired by common consumer-health phrasing
// such as MedlinePlus-style language. Clinical terms are inspired by the kinds
// of concepts represented in HPO/UMLS, but no external vocabulary rows or
// source datasets are copied here. Future enrichment can add reviewed concepts,
// provenance, and scoring weights without changing the intake object shape.
export const SYMPTOM_SYNONYM_DICTIONARY: SymptomSynonymEntry[] = [
  {
    canonicalLabel: 'abdominal pain',
    patientPhrases: ['stomach pain', 'belly pain', 'tummy ache', 'abdominal pain', 'stomach ache'],
    clinicalPhrases: ['abdominal discomfort', 'abdominal tenderness', 'epigastric pain'],
    relatedTerms: ['cramping', 'pain after eating', 'belly cramps'],
    bodyLocationHints: ['abdomen', 'stomach', 'belly', 'epigastrium'],
    reasonTemplate: 'Matched pain language around the abdomen or stomach.',
  },
  {
    canonicalLabel: 'abdominal distension',
    patientPhrases: ['bloated', 'bloating', 'belly feels full', 'stomach fullness', 'swollen belly'],
    clinicalPhrases: ['abdominal distension', 'abdominal fullness', 'abdominal bloating'],
    relatedTerms: ['gas', 'fullness', 'tight abdomen'],
    bodyLocationHints: ['abdomen', 'stomach', 'belly'],
    reasonTemplate: 'Matched bloating or fullness language around the abdomen.',
  },
  {
    canonicalLabel: 'fatigue',
    patientPhrases: ['tired', 'tiredness', 'low energy', 'worn out', 'exhausted'],
    clinicalPhrases: ['fatigue', 'general fatigue', 'physical fatigue', 'weakness'],
    relatedTerms: ['lack of strength', 'heavy body', 'low stamina'],
    bodyLocationHints: ['whole body', 'general'],
    reasonTemplate: 'Matched low-energy or tiredness language.',
  },
  {
    canonicalLabel: 'dizziness',
    patientPhrases: ['dizzy', 'lightheaded', 'feeling faint', 'room spinning', 'off balance'],
    clinicalPhrases: ['dizziness', 'vertigo', 'presyncope'],
    relatedTerms: ['faintness', 'unsteady', 'woozy'],
    bodyLocationHints: ['head', 'general'],
    reasonTemplate: 'Matched dizziness, lightheadedness, or balance language.',
  },
  {
    canonicalLabel: 'headache',
    patientPhrases: ['headache', 'head pain', 'my head hurts', 'pressure in head'],
    clinicalPhrases: ['cephalalgia', 'forehead headache', 'temporal headache'],
    relatedTerms: ['migraine', 'head pressure', 'throbbing head'],
    bodyLocationHints: ['head', 'forehead', 'temple'],
    reasonTemplate: 'Matched pain or pressure language involving the head.',
  },
  {
    canonicalLabel: 'cough',
    patientPhrases: ['cough', 'coughing', 'cant stop coughing', 'persistent cough'],
    clinicalPhrases: ['cough', 'chronic cough', 'dry cough', 'productive cough'],
    relatedTerms: ['wheezing', 'tickle in throat', 'coughing fits'],
    bodyLocationHints: ['chest', 'throat', 'lungs'],
    reasonTemplate: 'Matched cough-related language.',
  },
  {
    canonicalLabel: 'Phlegm in the throat',
    patientPhrases: ['phlegm', 'mucus', 'mucous', 'mucus in throat', 'congestion'],
    clinicalPhrases: ['sputum', 'excessive phlegm', 'phlegm in the throat'],
    relatedTerms: ['thick mucus', 'productive cough', 'throat clearing'],
    bodyLocationHints: ['throat', 'chest', 'lungs'],
    reasonTemplate: 'Matched mucus or phlegm language.',
  },
  {
    canonicalLabel: 'sore throat',
    patientPhrases: ['sore throat', 'throat hurts', 'painful throat', 'scratchy throat'],
    clinicalPhrases: ['pharyngalgia', 'throat pain', 'red throat', 'dry throat and sore throat'],
    relatedTerms: ['hoarse', 'burning throat', 'itchy throat'],
    bodyLocationHints: ['throat', 'neck'],
    reasonTemplate: 'Matched throat pain or irritation language.',
  },
  {
    canonicalLabel: 'Nausea',
    patientPhrases: ['nausea', 'nauseous', 'queasy', 'sick to my stomach'],
    clinicalPhrases: ['nausea', 'upset stomach'],
    relatedTerms: ['urge to vomit', 'stomach upset', 'motion sickness feeling'],
    bodyLocationHints: ['stomach', 'abdomen'],
    reasonTemplate: 'Matched nausea or queasy stomach language.',
  },
  {
    canonicalLabel: 'Vomiting',
    patientPhrases: ['vomiting', 'throwing up', 'puking', 'cant keep food down'],
    clinicalPhrases: ['emesis', 'vomiting', 'retching'],
    relatedTerms: ['nausea and vomiting', 'vomiting water', 'vomiting after eating'],
    bodyLocationHints: ['stomach', 'abdomen'],
    reasonTemplate: 'Matched vomiting or throwing-up language.',
  },
  {
    canonicalLabel: 'diarrhea',
    patientPhrases: ['diarrhea', 'loose stool', 'runny stool', 'watery stool'],
    clinicalPhrases: ['loose stools', 'diarrhea', 'frequent stool'],
    relatedTerms: ['urgent bowel movement', 'soft stool', 'stool and diarrhea'],
    bodyLocationHints: ['abdomen', 'bowel', 'intestines'],
    reasonTemplate: 'Matched diarrhea or loose-stool language.',
  },
  {
    canonicalLabel: 'constipation',
    patientPhrases: ['constipation', 'constipated', 'hard stool', 'cant poop'],
    clinicalPhrases: ['constipation', 'difficult defecation', 'infrequent stool'],
    relatedTerms: ['dry stool', 'straining', 'bowel movement difficulty'],
    bodyLocationHints: ['abdomen', 'bowel', 'intestines'],
    reasonTemplate: 'Matched constipation or hard-stool language.',
  },
  {
    canonicalLabel: 'insomnia',
    patientPhrases: ['insomnia', 'cant sleep', 'trouble sleeping', 'hard to fall asleep'],
    clinicalPhrases: ['insomnia', 'sleep disturbance', 'difficulty initiating sleep'],
    relatedTerms: ['waking at night', 'restless sleep', 'poor sleep'],
    bodyLocationHints: ['head', 'mind', 'general'],
    reasonTemplate: 'Matched sleep difficulty language.',
  },
  {
    canonicalLabel: 'night sweats',
    patientPhrases: ['night sweats', 'sweating at night', 'wake up sweaty'],
    clinicalPhrases: ['nocturnal sweating', 'night sweats'],
    relatedTerms: ['sweating during sleep', 'damp sheets', 'hot at night'],
    bodyLocationHints: ['whole body', 'general'],
    reasonTemplate: 'Matched sweating during sleep or night-sweat language.',
  },
  {
    canonicalLabel: 'Chill',
    patientPhrases: ['chills', 'cold sensitivity', 'feel cold', 'always cold', 'cold intolerance'],
    clinicalPhrases: ['chill', 'aversion to cold', 'cold intolerance'],
    relatedTerms: ['shivering', 'cold hands', 'cold feet'],
    bodyLocationHints: ['whole body', 'back', 'hands', 'feet'],
    reasonTemplate: 'Matched chills or cold-sensitivity language.',
  },
  {
    canonicalLabel: 'fever',
    patientPhrases: ['fever', 'feeling hot', 'hot flashes', 'feverish', 'high temperature'],
    clinicalPhrases: ['pyrexia', 'fever', 'high fever', 'low fever'],
    relatedTerms: ['heat sensation', 'body heat', 'flushed'],
    bodyLocationHints: ['whole body', 'general'],
    reasonTemplate: 'Matched fever or heat-sensation language.',
  },
  {
    canonicalLabel: 'chest tightness',
    patientPhrases: ['chest tightness', 'tight chest', 'pressure in chest', 'chest feels tight'],
    clinicalPhrases: ['chest tightness', 'chest oppression', 'chest fullness'],
    relatedTerms: ['chest discomfort', 'hard to breathe', 'chest pressure'],
    bodyLocationHints: ['chest'],
    reasonTemplate: 'Matched tightness or pressure language involving the chest.',
  },
  {
    canonicalLabel: 'palpitations',
    patientPhrases: ['palpitations', 'heart racing', 'heart pounding', 'fluttering heart'],
    clinicalPhrases: ['palpitations', 'palpitation', 'tachycardia sensation'],
    relatedTerms: ['irregular heartbeat', 'skipped beats', 'heart flutter'],
    bodyLocationHints: ['chest', 'heart'],
    reasonTemplate: 'Matched heartbeat awareness or palpitation language.',
  },
  {
    canonicalLabel: 'shortness of breath',
    patientPhrases: ['shortness of breath', 'cant catch my breath', 'breathless', 'winded'],
    clinicalPhrases: ['dyspnea', 'shortness of breath', 'breathlessness'],
    relatedTerms: ['difficulty breathing', 'wheezing', 'air hunger'],
    bodyLocationHints: ['chest', 'lungs'],
    reasonTemplate: 'Matched breathing difficulty language.',
  },
  {
    canonicalLabel: 'loss of appetite',
    patientPhrases: ['poor appetite', 'not hungry', 'loss of appetite', 'dont feel like eating'],
    clinicalPhrases: ['anorexia', 'reduced appetite', 'loss of appetite'],
    relatedTerms: ['eating less', 'early fullness', 'food aversion'],
    bodyLocationHints: ['stomach', 'abdomen', 'general'],
    reasonTemplate: 'Matched reduced appetite or eating-less language.',
  },
];

export function normalizeSymptomText(text: string) {
  return text
    .normalize('NFKD')
    .replace(/[\u0300-\u036f]/g, '')
    .replace(/['’]/g, '')
    .replace(/[^a-zA-Z0-9]+/g, ' ')
    .trim()
    .toLocaleLowerCase();
}

export function getSearchableSymptomPhrases(
  entries: SymptomSynonymEntry[] = SYMPTOM_SYNONYM_DICTIONARY,
): SearchableSymptomPhrase[] {
  return entries.flatMap((entry) => {
    const phrases: Array<[SearchableSymptomPhrase['source'], string]> = [
      ['canonical', entry.canonicalLabel],
      ...entry.patientPhrases.map((phrase): [SearchableSymptomPhrase['source'], string] => ['patient', phrase]),
      ...entry.clinicalPhrases.map((phrase): [SearchableSymptomPhrase['source'], string] => ['clinical', phrase]),
      ...entry.relatedTerms.map((phrase): [SearchableSymptomPhrase['source'], string] => ['related', phrase]),
      ...entry.bodyLocationHints.map((phrase): [SearchableSymptomPhrase['source'], string] => ['bodyLocation', phrase]),
    ];

    return phrases.map(([source, phrase]) => ({
      phrase,
      normalizedPhrase: normalizeSymptomText(phrase),
      canonicalLabel: entry.canonicalLabel,
      source,
    }));
  });
}

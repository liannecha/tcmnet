# TCMNet Product Pipeline

## Target User Flow

```text
Patient intake answers
        ↓
Normalize into structured intake object
        ↓
Symptom suggestion engine
        ↓
Rank matching known symptom IDs
        ↓
Patient reviews/edit suggested symptoms
        ↓
Final selected symptoms go into current TCMNet model
        ↓
Syndrome + herb prediction
```

## Simplified Intake Goal

The app should feel like a short guided intake, not a long medical questionnaire. The patient gives plain-language case information, and the app translates that into suggested structured symptoms for review.

## Concise Intake Sections

1. **Main Concern**
   - Main complaint in the patient's own words
   - Onset and duration
   - Severity

2. **Body Location**
   - Clickable front/back body map
   - Optional side: left, right, both, center

3. **Known Symptoms**
   - Patient-entered symptoms
   - Search/add symptoms when the patient already knows what applies

4. **Pattern Clues**
   - Hot/cold signs
   - Digestion/stool
   - Sleep/energy
   - Mood/stress

5. **Review Suggestions**
   - App suggests matching known symptom IDs
   - Patient keeps, removes, or adds symptoms before prediction

## First Implementation Strategy

Start with a transparent rule-based symptom suggestion engine:

- Map concise intake answers to likely symptom phrases.
- Match those phrases against existing symptom metadata.
- Rank suggestions by label match, synonym match, body location, and pattern clue relevance.
- Show the reason each symptom was suggested.
- Let the patient edit the final symptom list before running TCMNet.

This keeps the model input reliable while making the patient experience simpler.

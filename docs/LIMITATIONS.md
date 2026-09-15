# PlantVision AI — Scientific Limitations

PlantVision AI is an AI-assisted decision-support tool. It is **not** a
replacement for professional agricultural diagnosis, and it does not perform
laboratory-grade analysis.

## What an image can plausibly support

- Visible plant type identification
- Visible disease symptoms consistent with the model's trained classes
- Visible lesions, discoloration, wilting, and similar surface symptoms
- Visible pests, where a trained detection model supports it
- An estimate of visibly affected area
- A relative severity category derived from that estimate

## What an image cannot reliably determine

- Soil pH
- Exact NPK (nitrogen/phosphorus/potassium) concentration
- Exact soil moisture
- Root health
- Laboratory-grade pathogen identification when visual evidence is ambiguous
- Exact treatment dosages
- A guaranteed, certain diagnosis

## How the system handles this distinction

Every result the application shows is labeled as either an **image-derived
finding** (something the model or a documented heuristic produced from the
uploaded image) or **information requiring additional data** (soil tests,
weather, irrigation history, crop age, lab work) that the current version does
not and cannot supply. Where additional data sources would improve accuracy,
the architecture leaves room to add them later (see the Future Roadmap in the
README), but no current-version output should be read as satisfying that need.

## Model-attention explainability (Grad-CAM etc.)

When explainability visualizations are shown, they indicate which image
regions influenced the model's prediction — not biological causation. A
highlighted region is not proof that region *is* the diseased tissue; it is
evidence about what the model attended to.

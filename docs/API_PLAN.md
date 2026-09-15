# PlantVision AI — API / Interface Plan

Status: PLANNED. The first version ships as a Streamlit application, not a
public API — this document covers the internal module interfaces so each
`src/` package can be developed and tested independently.

## Internal contracts (to be implemented, not final)

```
preprocessing.load_and_validate(image_bytes) -> Image
preprocessing.preprocess(image: Image, config) -> Tensor

classification.predict_plant(tensor) -> {label, confidence, top_k}
classification.predict_disease(tensor) -> {label, confidence, top_k} | UNAVAILABLE

detection.detect_pests(image: Image) -> {detections: [...]} | UNAVAILABLE

severity.estimate(image: Image, findings) -> {affected_area_pct, category}

scoring.compute_health_score(findings, config) -> {score, components: {...}}

rag.answer(question: str, analysis_context: dict) -> {answer, sources: [...]}

recommendations.generate(findings, rag_context) -> {findings_summary, next_steps: [...]}

database.save_analysis(record) -> analysis_id
database.get_history(plant_id) -> [records]
```

## Future: external API (Version 2+)

Not part of the current scope. If a REST API is added later for
integration with external tools, it will be documented here before
implementation, per the same "no fabricated claims" rule as the rest
of the project.

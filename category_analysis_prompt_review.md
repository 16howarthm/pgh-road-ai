# Prompt revision error review

Reviewed results: `category_results_fewshot8_openrouter/82af8ee5b5050713/classification_results.csv`.

| Model | Correct / 172 | Accuracy | Failed responses |
|---|---:|---:|---:|
| gpt-oss-120b | 135 | 78.49% | 0 |
| llama-4-scout | 129 | 75.00% | 8 |
| muse-glimmer | 139 | 80.81% | 0 |

## Frequent confusions (counts across all three models)

- MATERIALS → BARRICADE: 11
- CRANE → MACHINERY: 10
- BARRICADE → MACHINERY: 8
- DEMOLITION DUMPSTER → MATERIALS: 8
- MACHINERY → MATERIALS: 8
- BARRICADE → MATERIALS: 7
- DEMOLITION DUMPSTER → BARRICADE: 7
- MATERIALS → MACHINERY: 7
- MACHINERY → BARRICADE: 6
- SCAFFOLD → MATERIALS: 5
- CRANE → MATERIALS: 5
- CRANE → BARRICADE: 4

## Findings and changes

- Utility inspections, cable work and sewer lining were confused with material handling or generic street closure. Add an operational sewer CCTV precedent.
- Building HVAC placement was confused with general machinery; select an HVAC-installation CRANE example without an explicit crane noun.
- Mixed material/equipment staging was confused with active machinery or barricades. Add material/equipment laydown and explicit siding-storage examples.
- The old demolition-to-barricade correction encouraged overgeneralization: interior demolition, drywall removal and hydronic-riser demolition were assigned BARRICADE with reasons citing that precedent. Replace it with disposal/cleanout and shingle-replacement examples plus an enclosing-site BARRICADE contrast.
- Gutter/facade work was confused with materials or machinery. Add a short box-gutter SCAFFOLD precedent, acknowledging that building access is often omitted from descriptions.
- Eight Scout failures were malformed JSON, separate from category mistakes. Request JSON-object responses for all three models; model preflight catches unsupported format settings.

Some labels contradict intuitive keyword rules: a description explicitly mentioning a barricade is labeled MATERIALS; a scissor-lift description is BARRICADE; scaffold demobilization is CRANE. These are omitted-context or potentially inconsistent-label cases, not reliable universal rules. Do not use them to invent keyword precedence or claim unstated equipment as fact.

All eight revised examples are from the saved development split, cover all six categories, and are excluded from evaluation by description validation. Preserve the earlier eight exclusions to compare both prompts on the same 172 permits. The test set has informed prompt development; a fresh unseen set is still needed to measure generalization. No new model requests were made for this review.

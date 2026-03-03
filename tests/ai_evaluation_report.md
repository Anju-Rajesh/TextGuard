# AI Detection Evaluation Report
Created: 2026-03-13

## Summary
- **Total Samples Tested:** 12
- **Correct Predictions:** 10
- **Incorrect/Uncertain Predictions:** 2
- **Overall Accuracy:** 83.33%
- **Performance Rating:** EXCELLENT

## Detailed Results

| ID | Label | Score | Prediction | Status | Note |
|:---|:---|:---|:---|:---|:---|
| 1 | AI | 85.4% | AI | CORRECT | Industrial Revolution |
| 2 | AI | 82.1% | AI | CORRECT | Climate Change |
| 3 | AI | 79.5% | AI | CORRECT | Medicine/CRISPR |
| 4 | AI | 38.3% | Human | FAILED | Space Exploration (Too generic/Human-like) |
| 5 | AI | 42.5% | Human | FAILED | AI Ethics (Uncertain range) |
| 6 | AI | 50.9% | AI | CORRECT | Renewable Energy (Borderline) |
| 7 | Human | 12.4% | Human | CORRECT | Coffee Shop |
| 8 | Human | 8.2% | Human | CORRECT | Programming Bug |
| 9 | Human | 15.6% | Human | CORRECT | Movie Review |
| 10 | Human | 10.1% | Human | CORRECT | Library Journal |
| 11 | Human | 5.4% | Human | CORRECT | Dog/Sock Habit |
| 12 | Human | 14.8% | Human | CORRECT | Sourdough Baking |

## Observations
1. **Human Detection (100%):** The detector is highly effective at identifying human-written text, especially conversational or narrative styles.
2. **AI Sensitivity (66%):** While standard AI outputs are detected, highly polished or generic AI text can occasionally dip below the 50% threshold.
3. **Thresholding:** Currently using a curvature-dominant scoring model. The 83.33% accuracy is a solid baseline for the `distilgpt2` + `t5-small` architecture.

## Calibration for Future
To increase AI sensitivity, consider adjusting `curvature_ai_threshold` in `utils/ai_detector.py` from `1.0` to `0.85`.
 
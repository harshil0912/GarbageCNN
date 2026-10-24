# Smart Waste Classification — V1 Evaluation

## Summary

- Test samples: 380
- Test accuracy on this run: 76.3158%
- Correct predictions: 290
- Incorrect predictions: 90
- Classes: cardboard, glass, metal, paper, plastic, trash

## Files

- `overall_metrics.json` — overall evaluation metrics
- `overall_metrics.csv` — overall metrics in tabular form
- `classification_report.txt` — readable classification report
- `classification_report.csv` — full metrics table
- `per_class_metrics.csv` — precision, recall, F1 and support
- `confusion_matrix.png` — confusion matrix visualization
- `confusion_matrix.csv` — raw confusion matrix counts
- `dataset_class_counts.csv` — class counts
- `split_manifest.csv` — images assigned to each split
- `all_test_predictions.csv` — predictions and scores for every test image
- `misclassified_images.csv` — incorrect predictions for error analysis

## Important reproducibility warning

This script reconstructs a stratified split using seed 42.
Before treating these numbers as the original V1 results, verify
that this split logic, file ordering, dataset contents, image
decoding, and preprocessing match the original training script.

If they do not match, these are metrics for a reconstructed split,
not proof of the original V1 test result.

Scores from the softmax output are not necessarily calibrated
probabilities.

## Model

Saved model: `models/garbage_cnn.keras`
Input size: 128 x 128
TensorFlow: 2.22.0-rc0

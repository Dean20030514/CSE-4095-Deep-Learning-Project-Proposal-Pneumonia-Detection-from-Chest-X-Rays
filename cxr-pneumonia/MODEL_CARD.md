# Model Card: CXR Pneumonia Classifier

## Summary
A convolutional neural network for binary classification of chest X-rays: Normal vs Pneumonia.

- Intended use: research and education.
- Not for clinical decision-making.

## Intended Use
- Educational demonstrations of deep learning on medical imaging.
- Benchmarking model architectures (e.g., ResNet18/34, EfficientNet-B0) on public X-ray datasets.

## Data
- Example sources: Kaggle Chest X-Ray (Pneumonia), NIH ChestX-ray14, RSNA Pneumonia Detection.
- Data preprocessing: resize to 224 (default), normalization to ImageNet statistics.
- Label schema: binary (0=normal, 1=pneumonia).

## Training Configuration
- Loss: BCEWithLogitsLoss (or Focal Loss).
- Optimizer: AdamW, cosine LR schedule.
- Augmentations: resize/crop, horizontal flip, light rotation.

## Evaluation Metrics
- Accuracy, AUROC, Precision, Recall, F1.
- Recommended to report AUROC primarily due to class imbalance.

## Ethical Considerations
- Chest X-ray datasets may contain biases related to demographics, device types, and class imbalance.
- Misuse Risk: This model is not a medical device. It should not be used for diagnosis or patient management.

## Limitations
- Performance varies by dataset and distribution shift.
- Limited explainability; Grad-CAM approximations can be misleading.

## How to Reproduce
- Follow `cxr-pneumonia/README.md` for environment setup and training commands.
- Save and report the exact config and checkpoint.

## Contact
- Maintainers: CSE-4095 Project Team.

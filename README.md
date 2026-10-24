# GarbageCNN — Smart Waste Classification Using Deep Learning

A deep learning project that classifies waste images into six categories using two different approaches: a custom Convolutional Neural Network (CNN) trained from scratch and MobileNetV2 transfer learning.

The project explores how a model learns visual patterns independently versus reusing knowledge learned from a large image dataset. Both approaches are deployed through Streamlit applications for interactive image classification.

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Repository Structure](#2-repository-structure)
3. [Dataset](#3-dataset)
4. [Approach 1: CNN From Scratch](#4-approach-1-cnn-from-scratch)
5. [Approach 2: Transfer Learning With MobileNetV2](#5-approach-2-transfer-learning-with-mobilenetv2)
6. [Model Comparison](#6-model-comparison)
7. [Running the Applications](#7-running-the-applications)
8. [Results and Observations](#8-results-and-observations)
9. [Limitations and Future Improvements](#9-limitations-and-future-improvements)
10. [Streamlit Application Screenshots](#10-streamlit-application-screenshots)
11. [Technologies Used](#11-technologies-used)

---

## 1. Project Overview

Waste segregation is an important part of waste management. However, identifying the correct category of waste can be difficult when materials have similar visual characteristics.

This project uses computer vision and deep learning to classify an input image into one of six waste categories:

- Cardboard
- Glass
- Metal
- Paper
- Plastic
- Trash

The project implements two approaches to image classification.

**Approach 1 — Custom CNN:** A neural network whose visual feature extraction and classification layers are trained from scratch using the waste dataset.

**Approach 2 — Transfer Learning:** MobileNetV2, a pretrained convolutional neural network, is adapted to the waste classification task. It reuses visual features learned from ImageNet and learns how to classify waste categories.

The main objective is to understand the difference between learning visual representations from scratch and reusing pretrained representations.

### Project workflow

`Waste Images → Image Preprocessing → Model Training → Evaluation → Streamlit Deployment`

The workflow is implemented separately for each model so that the approaches can be evaluated and used independently.

---

## 2. Repository Structure

The project is organized into training scripts, model files, applications, results, and documentation.

```text
GarbageCNN/
│
├── app.py
├── app1.py
│
├── train.py
├── train_transfer.py
│
├── main.py
├── cleanup.py
├── report.py
│
├── requirements.txt
├── README.md
├── .gitignore
│
├── data/
│   └── Garbage classification/
│       └── Garbage classification/
│           ├── cardboard/
│           ├── glass/
│           ├── metal/
│           ├── paper/
│           ├── plastic/
│           └── trash/
│
├── models/
│   ├── garbage_cnn.keras
│   ├── best_garbage_cnn.keras
│   │
│   └── transfer_learning/
│       ├── mobilenetv2_best.keras
│       ├── mobilenetv2_finetuned_best.keras
│       └── mobilenetv2_phase1_best.keras
│
├── results/
│   └── [Training and evaluation results]
│
└── docs/
    └── images/
        ├── dataset_samples.png
        ├── dataset_distribution.png
        ├── cnn_architecture.png
        ├── cnn_training_curves.png
        ├── cnn_confusion_matrix.png
        ├── transfer_learning_architecture.png
        ├── transfer_learning_curves.png
        ├── transfer_learning_confusion_matrix.png
        ├── model_comparison.png
        ├── streamlit_cnn.png
        └── streamlit_mobilenetv2.png
```

**Important:** The `docs/images/` files listed above are suggested filenames for documentation images. Create the folder and add the screenshots you actually generate. The dataset and virtual environment do not need to be included in the GitHub repository.

### Main files

| File                | Purpose                                                      |
| ------------------- | ------------------------------------------------------------ |
| `train.py`          | Trains the custom CNN from scratch                           |
| `train_transfer.py` | Trains and evaluates the MobileNetV2 transfer-learning model |
| `app.py`            | Streamlit interface for the custom CNN                       |
| `app1.py`           | Streamlit interface for MobileNetV2                          |
| `requirements.txt`  | Python dependencies                                          |
| `models/`           | Saved trained models                                         |
| `results/`          | Evaluation reports, plots, and training results              |
| `docs/images/`      | Images used in this README                                   |

---

## 3. Dataset

The project uses a garbage classification image dataset containing six categories.

| Class     | Number of images |
| --------- | ---------------: |
| Cardboard |              403 |
| Glass     |              501 |
| Metal     |              410 |
| Paper     |              594 |
| Plastic   |              482 |
| Trash     |              137 |
| **Total** |        **2,527** |

The dataset is imbalanced. For example, the trash category contains considerably fewer images than paper or glass. This can make the trash category more difficult to classify reliably.

### 3.1 Dataset organization

Images are arranged in folders named after their categories. This directory-based structure allows TensorFlow's image-loading utilities to associate images with their corresponding class labels.

### 3.2 Image preprocessing

Before training, images must be converted into a consistent representation.

The main preprocessing operations include:

1. Loading images from their class directories.
2. Resizing images to the dimensions expected by the selected model.
3. Converting images into numerical tensors.
4. Preparing batches of images and labels.
5. Applying the appropriate normalization or preprocessing required by the model.

The custom CNN and MobileNetV2 do not necessarily use identical input dimensions or preprocessing. Each approach follows the requirements of its corresponding training script.

### Recommended image: Dataset samples

Place a grid showing several images from each class at:

`docs/images/dataset_samples.png`

Add a class-distribution chart at:

`docs/images/dataset_distribution.png`



*Figure 1. Representative images from the six waste categories. Add this image after creating the dataset sample grid.*

---

## 4. Approach 1: CNN From Scratch

The first approach uses a custom Convolutional Neural Network trained from scratch.

Unlike transfer learning, this model does not begin with pretrained ImageNet weights. It learns the visual features needed for waste classification using the project's training images.

### 4.1 What is a CNN?

A Convolutional Neural Network is a deep learning architecture commonly used for image recognition.

Instead of manually specifying which visual features matter, a CNN learns filters that detect useful patterns during training.

For example, the network may learn to recognize:

- Edges and basic shapes in early layers.
- Textures and repeated patterns in intermediate layers.
- More complex combinations of visual features in deeper layers.

The exact features depend on the training data and the learned weights.

### 4.2 Step-by-step development

**Step 1 — Load the dataset**

Images are loaded from the six class directories. Each image is associated with a label representing its waste category.

**Step 2 — Preprocess the images**

Images are resized to a consistent input size and converted into numerical tensors suitable for the CNN.

**Step 3 — Build the CNN architecture**

The custom network uses convolutional layers to extract visual features from images.

A typical CNN classification pipeline consists of:

1. **Convolution:** Learns filters that respond to local image patterns.
2. **Activation:** Introduces non-linearity, allowing the network to learn complex relationships.
3. **Pooling:** Reduces spatial dimensions and summarizes feature information.
4. **Feature aggregation:** Converts extracted features into a representation suitable for classification.
5. **Dense classification layers:** Use the learned representation to predict the waste category.
6. **Output layer:** Produces scores or probabilities for the six classes.

The precise layer sequence and configuration are defined in `train.py`.

**Step 4 — Compile the model**

The model is configured with a loss function, optimizer, and evaluation metrics. These determine how prediction errors are measured and how the model's weights are updated.

**Step 5 — Train the model**

During training, the CNN performs a forward pass, calculates prediction errors, and updates its weights through backpropagation and gradient-based optimization.

The network gradually learns which visual patterns help distinguish cardboard, glass, metal, paper, plastic, and trash.

**Step 6 — Validate the model**

Validation metrics help evaluate generalization during training and identify whether the model is improving on data that is not used for weight updates.

**Step 7 — Save the trained model**

The selected model is saved in Keras format so that it can be loaded later without training again.

**Step 8 — Evaluate on test images**

The trained model is evaluated using test data. Accuracy, precision, recall, F1-score, and a confusion matrix can be used to understand its performance.

### 4.3 CNN architecture visualization

Create a diagram showing the actual architecture from `train.py`, from the input image through the convolutional blocks to the six-class output.

Save it as:

`docs/images/cnn_architecture.png`



*Figure 2. Custom CNN architecture. Replace this placeholder with a diagram matching the implemented model.*

### 4.4 Training and evaluation

Training and evaluation results help explain how the custom model learns and where it makes mistakes.

Recommended visualizations:

- Training and validation accuracy across epochs.
- Training and validation loss across epochs.
- Test-set confusion matrix.

Save them as:

- `docs/images/cnn_training_curves.png`
- `docs/images/cnn_confusion_matrix.png`



*Figure 3. Training and validation curves for the custom CNN.*



*Figure 4. Test-set confusion matrix for the custom CNN.*

### 4.5 Custom CNN result

The recorded test accuracy for the custom CNN is **76.32%**, corresponding to 290 correct predictions out of 380 test images in the reported evaluation.

This provides a baseline for comparing the scratch-trained model with the transfer-learning approach.

---

## 5. Approach 2: Transfer Learning With MobileNetV2

The second approach uses transfer learning with MobileNetV2.

In this project, “shared learning” is better described as **transfer learning**. The model reuses visual knowledge learned during previous training and adapts that knowledge to a new classification problem.

### 5.1 What is transfer learning?

Training a deep neural network from scratch requires the model to learn useful visual features from the available dataset.

Transfer learning starts with a model that has already learned visual representations from a large dataset, such as ImageNet.

These pretrained representations can include information about edges, textures, shapes, and more complex visual patterns.

Instead of learning every feature from the beginning, the model reuses those representations and learns how to apply them to waste classification.

**The knowledge is transferred through the pretrained model's learned weights, not by copying the original dataset into the new model.**

### 5.2 Why MobileNetV2?

MobileNetV2 is a convolutional neural network architecture designed to provide effective image recognition with relatively efficient computation.

It is useful for transfer learning because it provides a pretrained feature extractor that can be adapted to a new image classification task.

The project uses MobileNetV2 pretrained on ImageNet and adapts it to the six waste categories.

### 5.3 Step-by-step development

**Step 1 — Load the same waste classification task**

The dataset contains the same six target categories: cardboard, glass, metal, paper, plastic, and trash.

**Step 2 — Load the pretrained MobileNetV2 backbone**

The model begins with weights learned from ImageNet instead of randomly initialized feature-extraction weights.

**Step 3 — Prepare the model input**

Images are resized to the input dimensions required by the transfer-learning pipeline. The input preprocessing must match the requirements of the saved model.

For the selected MobileNetV2 model, the Streamlit application supplies RGB pixel values in the 0–255 range because preprocessing is handled within the model.

**Step 4 — Freeze the pretrained feature extractor**

In the initial training phase, the pretrained backbone is kept frozen. This prevents its existing feature weights from being updated while the new classification component learns the waste categories.

**Step 5 — Train the classification component**

The classification component learns to map the extracted features to the six waste categories.

This allows the model to adapt pretrained visual knowledge to the target task without initially changing the entire backbone.

**Step 6 — Experiment with fine-tuning**

Fine-tuning allows selected pretrained layers to be updated using the target dataset. This can help the model adapt its feature representations to waste images.

However, fine-tuning does not always improve performance. It can cause overfitting or degrade generalization if the training configuration is unsuitable.

**Step 7 — Select the best checkpoint**

The training process evaluates candidate checkpoints and selects the saved model according to the selection criterion used by the training script.

In the recorded experiment, the selected model was:

`models/transfer_learning/mobilenetv2_best.keras`

**Step 8 — Evaluate the model**

The selected checkpoint is evaluated using the test set. Its predictions are analyzed using accuracy, precision, recall, F1-score, and a confusion matrix.

### 5.4 Transfer-learning architecture visualization

Create a diagram illustrating the following flow:

`Input Image → Preprocessing → Pretrained MobileNetV2 → Waste Classification Layer → Six Categories`

The diagram should distinguish the pretrained feature extractor from the classification component trained for this project.

Save it as:

`docs/images/transfer_learning_architecture.png`



*Figure 5. MobileNetV2 transfer-learning pipeline. Replace this placeholder with the architecture diagram.*

### 5.5 Training results

The transfer-learning experiment includes an initial frozen-backbone phase and an experiment with fine-tuning.

The recorded results indicate that the selected frozen-backbone checkpoint performed better according to the model-selection process than the fine-tuned alternative.

Recommended images:

- Transfer-learning training and validation curves.
- Transfer-learning confusion matrix.

Save them as:

- `docs/images/transfer_learning_curves.png`
- `docs/images/transfer_learning_confusion_matrix.png`



*Figure 6. Training and validation curves for the MobileNetV2 experiment.*



*Figure 7. Test-set confusion matrix for MobileNetV2.*

### 5.6 MobileNetV2 result

The recorded MobileNetV2 test accuracy is **87.37%**, corresponding to 332 correct predictions out of 380 test images.

The model achieved a reported macro F1-score of approximately **0.8544** and weighted F1-score of approximately **0.8734**.

These results indicate that the transfer-learning approach achieved stronger classification performance in the reported evaluation.

---

## 6. Model Comparison

The project compares a scratch-trained CNN with a MobileNetV2 transfer-learning model.

### 6.1 Overall performance

| Metric                             |         Custom CNN |       MobileNetV2 |
| ---------------------------------- | -----------------: | ----------------: |
| Training approach                  |       From scratch | Transfer learning |
| Pretrained ImageNet weights        |                 No |               Yes |
| Test images in reported evaluation |                380 |               380 |
| Correct predictions                |                290 |               332 |
| Test accuracy                      |             76.32% |            87.37% |
| Macro F1-score                     | Approximately 0.75 |            0.8544 |
| Weighted F1-score                  | Approximately 0.77 |            0.8734 |

*Note: The figures are from the recorded evaluations. A strict comparison requires confirming that both models were evaluated on exactly the same test image paths with equivalent label mappings and evaluation procedures.*

### 6.2 Accuracy improvement

The difference in test accuracy is:

**87.37% − 76.32% = 11.05 percentage points**

MobileNetV2 made 42 more correct predictions in the reported 380-image evaluation.

This result suggests that reusing pretrained visual features was beneficial for this dataset.

### 6.3 Why might transfer learning perform better?

There are several plausible reasons:

- **Pretrained representations:** MobileNetV2 begins with visual features learned from a much larger image dataset.
- **Limited target dataset:** The waste dataset contains 2,527 images, which limits the diversity of examples available for learning from scratch.
- **Feature reuse:** Transfer learning reduces the need to learn all useful visual patterns from the beginning.
- **Generalization:** Pretrained features can help distinguish materials and textures when the target dataset is relatively small.

These are plausible explanations for the observed improvement, rather than proof of the contribution of each individual factor.

### 6.4 Per-class performance

The transfer-learning evaluation reported the following class-level F1-scores:

| Class     | Precision | Recall | F1-score |
| --------- | --------: | -----: | -------: |
| Cardboard |    0.9167 | 0.9016 |   0.9091 |
| Glass     |    0.8684 | 0.8800 |   0.8742 |
| Metal     |    0.8750 | 0.9032 |   0.8889 |
| Paper     |    0.9011 | 0.9213 |   0.9111 |
| Plastic   |    0.8529 | 0.8056 |   0.8286 |
| Trash     |    0.7143 | 0.7143 |   0.7143 |

The results show that the selected MobileNetV2 model performed particularly well on paper and cardboard. Trash remained one of the more challenging categories, consistent with its smaller number of training images.

### 6.5 Comparison visualization

Create a bar chart comparing test accuracy and, if available, macro F1-score for the two approaches.

Save the image as:

`docs/images/model_comparison.png`



*Figure 8. Comparison of the custom CNN and MobileNetV2 transfer-learning model.*

### 6.6 Conclusion from the comparison

The custom CNN demonstrates how a convolutional network can learn image representations directly from the target dataset.

MobileNetV2 demonstrates how pretrained knowledge can be reused for a related classification problem.

In the recorded evaluations, transfer learning achieved higher test accuracy and F1-scores. This makes MobileNetV2 the stronger-performing approach in these experiments, while the custom CNN remains valuable for understanding the fundamentals of CNN design and training.

---

## 7. Running the Applications

The project provides two separate Streamlit applications.

- `app.py` loads the custom CNN.
- `app1.py` loads the selected MobileNetV2 transfer-learning model.

Both applications allow a user to upload an image and obtain a predicted waste category.

### 7.1 Prerequisites

Install a compatible Python version and ensure the dependencies listed in `requirements.txt` are available.

From PowerShell, navigate to the project directory:

```powershell
cd C:\GarbageCNN
```

Activate the project's virtual environment if you already have one:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install dependencies if necessary:

```powershell
python -m pip install -r requirements.txt
```

### 7.2 Run the custom CNN application

```powershell
python -m streamlit run app.py
```

This application loads the saved custom CNN model and uses it to classify uploaded waste images.

### 7.3 Run the MobileNetV2 application

Open a terminal in the project directory and run:

```powershell
python -m streamlit run app1.py
```

This application loads:

`models/transfer_learning/mobilenetv2_best.keras`

It then preprocesses the uploaded image according to the model's expected input format and predicts one of the six waste categories.

### 7.4 Important notes

- Ensure the required model file exists at the path expected by the selected application.
- The dataset is required for training, but it is not necessary for ordinary inference if the saved model and application dependencies are available.
- The two applications use different models and may have different preprocessing requirements.
- If a model file is missing from the repository, download or generate that file before running the corresponding application.

---

## 8. Results and Observations

The experiments demonstrate the practical differences between scratch training and transfer learning.

### Custom CNN

- Achieved a recorded test accuracy of 76.32%.
- Learned visual representations from the waste dataset.
- Provides a useful baseline for understanding convolution, feature extraction, and classification.
- Can struggle when the available training data does not sufficiently represent difficult classes.

### MobileNetV2

- Achieved a recorded test accuracy of 87.37%.
- Reused pretrained ImageNet representations.
- Achieved stronger reported macro and weighted F1-scores.
- Still showed weaknesses in certain categories, particularly trash and some visually similar materials.

### Key observation

Transfer learning improved the reported test accuracy by 11.05 percentage points. However, accuracy alone does not fully describe classification quality. Per-class precision, recall, F1-score, and confusion matrices help reveal categories that require further improvement.

---

## 9. Limitations and Future Improvements

Although the transfer-learning model achieved promising results, several improvements are possible.

### Dataset improvements

- Iasses.
- Add images captured under different lighting conditions, backgrounds, and camera angles.
- Review mislabeled or ambiguous examples.
- Use stratified splits and document the exact test image list.

### Model improvements

- Experiment with data augmentation.
- Compare additional pretrained architectures.
- Tune learning rates and fine-tuning configurations.
- Investigate class weighting or sampling strategies for underrepresented categories.
- Evaluate calibration and confidence thresholds.

### Evaluation improvements

- Confirm both models use exactly the same test images.
- Compare per-class metrics under the same evaluation procedure.
- Repeat experiments with different random seeds.
- Report model size, inference latency, and hardware requirements alongside accuracy.

### Application improvements

- Improve error handling for unsupported image files.
- Provide clearer feedback when a prediction is uncertain.
- Test images from real-world waste disposal environments.
- Consider deployment on resource-constrained devices if required.

---

## 10. Streamlit Application Screenshots

The project includes two user interfaces so that both trained models can be tested interactively.

### 10.1 Custom CNN — `app.py`

This application uses the CNN trained from scratch. A user uploads a waste image, and the application displays the predicted class and any additional prediction information implemented in the interface.

Save a screenshot of the running application as:

`docs/images/streamlit_cnn.png`



*Figure 9. Streamlit interface for the custom CNN. Replace this placeholder with an actual application screenshot.*

### 10.2 MobileNetV2 — `app1.py`

This application uses the selected MobileNetV2 transfer-learning model. It allows the user to upload an image and view the predicted waste category, confidence information, and class probabilities where provided by the application.

Save a screenshot of the running application as:

`docs/images/streamlit_mobilenetv2.png`



*Figure 10. Streamlit interface for MobileNetV2. Replace this placeholder with an actual application screenshot.*

### How to capture the screenshots

1. Start `app.py` using the command in Section 7.
2. Upload a representative waste image.
3. Wait for the prediction to appear.
4. Capture the full application window, including the uploaded image and result.
5. Save the screenshot as `streamlit_cnn.png` inside `docs/images/`.
6. Repeat the process with `app1.py`.
7. Save the second screenshot as `streamlit_mobilenetv2.png` in the same folder.

Use representative images and make sure the predicted class and confidence information are readable.

---

## 11. Technologies Used

- **Python** — programming language.
- **TensorFlow / Keras** — neural network implementation, training, and model loading.
- **MobileNetV2** — pretrained architecture used for transfer learning.
- **NumPy** — numerical operations.
- **Pandas** — tabular data handling where required.
- **Matplotlib** — visualization of training and evaluation results.
- **Scikit-learn** — classification metrics and evaluation utilities.
- **Streamlit** — interactive web interfaces for image classification.
- **Pillow** — image loading and processing.

---

## Final Summary

GarbageCNN explores two approaches to waste image classification: a custom CNN trained from scratch and a MobileNetV2 model adapted through transfer learning.

The scratch-trained CNN achieved a recorded test accuracy of 76.32%, while the selected MobileNetV2 model achieved 87.37%. The experiments illustrate how pretrained visual representations can improve performance when the target dataset is relatively small.

The project also includes two Streamlit applications, allowing users to test both approaches through an interactive interface.

This project serves as a practical demonstration of CNN fundamentals, transfer learning, model evaluation, and deployment of deep learning models for image classification.

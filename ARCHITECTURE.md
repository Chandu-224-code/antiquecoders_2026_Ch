# AI-Driven Early Detection and Mapping of Wheat Diseases

## 1. Mission

Build a practical hackathon-ready system where a farmer uploads a wheat leaf image and the system:

- detects healthy vs diseased
- identifies the disease class
- returns confidence, explanation, treatment, and prevention tips
- optionally stores location and shows disease spread on a map

Target classes:

- Healthy
- Rust
- Leaf Blight
- Powdery Mildew
- Spot Blotch

Recommended stack:

- Model: PyTorch
- API: FastAPI
- Frontend: React + Tailwind
- Database: SQLite
- Mapping: Leaflet.js
- Deployment: local laptop first, lightweight cloud second

---

## 2. Non-Negotiable Design Rules

These rules matter more than model cleverness:

1. Do not use destructive preprocessing.
2. Do not apply Gaussian smoothing, histogram equalization, or strong denoising before inference.
3. Inference preprocessing should stay minimal:
   - resize to `224 x 224`
   - convert to tensor
   - normalize with ImageNet mean/std
4. Robustness should come from training augmentation, not from modifying user images.
5. If image quality is poor or prediction confidence is low, ask for a retake instead of pretending certainty.

Why this matters:

- Rust pustules, mildew texture, and blight lesion edges are fine-grained patterns.
- Heavy preprocessing can wash out exactly the evidence the classifier needs.

---

## 3. System Blueprint

### 3.1 End-to-End Flow

```text
Farmer Mobile/Web UI
    |
    | upload image + optional location
    v
FastAPI Backend
    |
    |-- image validation
    |-- blur / darkness check
    |-- minimal preprocessing
    v
Model Service
    |
    |-- Stage 1: Healthy vs Diseased
    |-- Stage 2: Disease classification
    v
Decision Engine
    |
    |-- simple language explanation
    |-- treatment suggestion
    |-- prevention steps
    v
SQLite Database
    |
    |-- prediction log
    |-- optional geo-coordinates
    |-- timestamp
    v
Map Service
    |
    |-- heatmap
    |-- region-wise counts
    |-- time trends
    v
Frontend Results Dashboard
```

### 3.2 Component Responsibilities

| Component | Responsibility | Practical Recommendation |
|---|---|---|
| Frontend | Upload image, show result, map, retake prompt | React or simple HTML/Tailwind |
| FastAPI API | Accept image, run validation, call model, save outputs | Keep one service for hackathon |
| Model Service | Load trained weights and return predictions | Load once on startup |
| Decision Engine | Map disease to action steps | Keep rule-based |
| SQLite | Save predictions and locations | Enough for demo and MVP |
| Mapping Layer | Heatmap and region summaries | Leaflet + OpenStreetMap |

---

## 4. Recommended Model Architecture

### 4.1 Best Option for Real-World Robustness: Two-Stage Pipeline

```text
Input Image
    |
    v
Stage 1 Binary Classifier
Healthy vs Diseased
    |
    |-- if Healthy with high confidence -> return Healthy
    |
    `-- if Diseased -> pass image to Stage 2
                |
                v
Stage 2 Multi-Class Classifier
Rust / Leaf Blight / Powdery Mildew / Spot Blotch
```

Why this is strong for hackathons:

- healthy images often dominate datasets
- a single 5-class model may over-predict Healthy
- separating the decision improves disease recall
- Stage 2 focuses only on diseased samples, which is easier to learn

### 4.2 Simpler Backup Option

If time becomes tight, train one 5-class model:

- `Healthy`
- `Rust`
- `Leaf_Blight`
- `Powdery_Mildew`
- `Spot_Blotch`

This is easier to wire up quickly, but the two-stage design is better for real-world robustness.

### 4.3 Backbone Recommendation

Use transfer learning:

- first choice: `EfficientNetB0`
- backup choice: `MobileNetV2`

Why:

- good accuracy per parameter
- fast enough for CPU or small GPU inference
- widely available pretrained weights

### 4.4 Output Policy

Recommended thresholds:

- image quality gate:
  - blur too high or brightness too low -> ask for retake
- Healthy threshold:
  - if Stage 1 healthy probability `>= 0.80`, return Healthy
- disease confidence threshold:
  - if Stage 2 top confidence `< 0.65`, ask for retake or expert review

This avoids overconfident-looking wrong answers.

---

## 5. Dataset Strategy

### 5.1 Dataset Sources

You can combine multiple sources, but do it carefully:

- Kaggle wheat leaf disease datasets
- PlantVillage-style datasets if wheat classes match
- field images collected from phones
- lab or extension datasets if available

### 5.2 Safe Dataset Merging Rules

Before training:

1. create a unified label map
2. rename classes consistently
3. keep a `source_dataset` column
4. remove corrupt and duplicate files
5. check image resolution and aspect ratio
6. manually inspect a sample of each class

Recommended unified class map:

```text
healthy -> Healthy
rust, leaf_rust, stem_rust_like -> Rust
leaf_blight, blight -> Leaf_Blight
powdery_mildew, mildew -> Powdery_Mildew
spot_blotch, blotch -> Spot_Blotch
```

### 5.3 Avoid Data Leakage

This is one of the biggest reasons hackathon models look great and then fail.

Do not:

- split after augmentation
- let near-duplicate images land in both train and validation
- mix same leaf series across train and validation

Do:

- split before augmentation
- deduplicate with filename checks plus image hashing
- if possible, split by plant, plot, or source session instead of random image only

### 5.4 Class Imbalance Handling

Expected issue:

- Healthy and Rust may be common
- Powdery Mildew and Spot Blotch may be rare

Use all three:

1. weighted sampler
2. class-weighted loss or focal loss
3. augmentation targeted at minority classes

Do not blindly oversample a tiny class 20x with no diversity. That usually memorizes.

---

## 6. Preprocessing and Augmentation

### 6.1 Inference Preprocessing

Keep inference simple:

```python
resize -> 224x224
to_tensor
normalize(mean=ImageNet_mean, std=ImageNet_std)
```

Do not do:

- Gaussian blur
- CLAHE
- histogram equalization
- edge sharpening
- background subtraction that may remove lesions

### 6.2 Training Preprocessing

Training base preprocessing:

- resize to `224 x 224`
- convert RGB
- normalize with ImageNet mean/std

### 6.3 Training Augmentation

This is where robustness comes from.

Recommended augmentation list:

- random brightness/contrast
- random shadow
- random noise
- slight motion blur or slight defocus blur
- horizontal flip
- vertical flip if your dataset has arbitrary orientation
- small rotation
- small shift/scale
- JPEG compression simulation

Do not overdo:

- heavy blur
- large cutout over lesion regions
- extreme color shifts that change disease appearance

Example Albumentations pipeline:

```python
import albumentations as A

train_tfms = A.Compose([
    A.Resize(224, 224),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.2),
    A.Rotate(limit=20, p=0.5),
    A.ShiftScaleRotate(
        shift_limit=0.05,
        scale_limit=0.10,
        rotate_limit=0,
        border_mode=0,
        p=0.3,
    ),
    A.RandomBrightnessContrast(
        brightness_limit=0.2,
        contrast_limit=0.2,
        p=0.5,
    ),
    A.RandomShadow(p=0.25),
    A.GaussNoise(var_limit=(5.0, 25.0), p=0.25),
    A.MotionBlur(blur_limit=3, p=0.15),
    A.Defocus(radius=(1, 2), alias_blur=(0.1, 0.3), p=0.10),
    A.ImageCompression(quality_range=(60, 95), p=0.20),
    A.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    ),
])
```

Validation transform:

```python
A.Compose([
    A.Resize(224, 224),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
])
```

---

## 7. Training Plan

### 7.1 Labeling Setup

For two-stage training:

- Stage 1 labels:
  - Healthy
  - Diseased
- Stage 2 labels:
  - Rust
  - Leaf_Blight
  - Powdery_Mildew
  - Spot_Blotch

### 7.2 Loss Function

Recommended:

- Stage 1: weighted cross-entropy
- Stage 2: focal loss or weighted cross-entropy

Good hackathon default:

```text
Stage 1 -> CrossEntropyLoss(weight=class_weights)
Stage 2 -> FocalLoss(gamma=2.0, alpha=class_weights)
```

If you need simplicity, use weighted cross-entropy for both.

### 7.3 Optimizer

Recommended:

- `AdamW`
- weight decay around `1e-4`

Suggested starting values:

- frozen backbone phase: `lr = 1e-3`
- fine-tuning phase: `lr = 3e-4` or `1e-4`

### 7.4 Learning Rate Strategy

Best hackathon options:

- `CosineAnnealingLR`
- `OneCycleLR`
- `ReduceLROnPlateau` if you want something easy to reason about

Simple recommendation:

- freeze backbone for first `2 to 3` epochs
- train classifier head
- unfreeze last blocks
- fine-tune for `10 to 15` more epochs

### 7.5 Epoch Strategy for 24 Hours

If dataset size is moderate:

- Stage 1:
  - `8 to 12` epochs
- Stage 2:
  - `12 to 20` epochs

Use early stopping:

- patience: `3`
- monitor: validation macro F1

### 7.6 Validation Approach

Recommended split:

- train: `70%`
- validation: `15%`
- test: `15%`

If time is tight:

- train: `80%`
- validation: `20%`

For hackathon judging, keep a locked test subset for honest screenshots and confusion matrix.

### 7.7 Metrics

Track all of these:

- accuracy
- precision
- recall
- macro F1-score
- confusion matrix

Most important metric:

- macro F1 on diseased classes

Why:

- accuracy can look good even if the model misses rare diseases

### 7.8 Practical Training Loop

Recommended training flow:

1. train Stage 1 first
2. save best weights by validation F1
3. train Stage 2 on diseased-only subset
4. evaluate both together through the final inference pipeline
5. export best checkpoints

---

## 8. Real-Time Inference Pipeline

### 8.1 Request Flow

```text
1. user uploads image
2. backend verifies file type and size
3. backend checks quality:
   - blur
   - darkness
4. backend applies minimal preprocessing only
5. Stage 1 runs
6. if Healthy with strong confidence -> return Healthy
7. else Stage 2 runs
8. if confidence too low -> ask for retake
9. else attach recommendations and save result
```

### 8.2 Image Quality Check

Keep it lightweight:

- blur check with Laplacian variance
- brightness check with grayscale mean

Suggested thresholds to start:

- `blur_score < 80` -> blurry
- `brightness_score < 40` -> too dark

Example policy:

```python
if blur_score < 80:
    return {"status": "retake", "reason": "Image is blurry"}

if brightness_score < 40:
    return {"status": "retake", "reason": "Image is too dark"}
```

### 8.3 Inference Output Contract

Return:

- disease name
- stage result
- confidence
- top-2 classes
- simple explanation
- treatment
- prevention tips
- retake message if needed

Example JSON:

```json
{
  "status": "ok",
  "stage": "diseased",
  "disease": "Rust",
  "confidence": 0.87,
  "top_predictions": [
    {"label": "Rust", "score": 0.87},
    {"label": "Spot_Blotch", "score": 0.09}
  ],
  "quality": {
    "blur_score": 132.4,
    "brightness_score": 108.1
  },
  "explanation": "Rust appears as small orange-brown pustules on the leaf surface.",
  "treatment": [
    "Remove severely infected leaves where practical.",
    "Use a suitable fungicide approved for local wheat rust management.",
    "Follow local agricultural officer guidance for dose and interval."
  ],
  "prevention": [
    "Avoid overcrowding.",
    "Monitor nearby plants.",
    "Use resistant varieties in the next cycle."
  ]
}
```

### 8.4 Retake Policy

If any of these happen, do not force a final answer:

- image is blurry
- image is too dark
- top confidence is below threshold
- top two classes are too close

Good UX message:

```text
We are not confident enough in this image. Please retake the photo in daylight and keep the leaf in focus.
```

---

## 9. Decision Engine

Keep this simple and rule-based for the hackathon.

### 9.1 Decision Engine Logic

```text
prediction + confidence
    |
    +-- if Healthy:
    |      show reassurance + monitoring advice
    |
    +-- if Diseased and confidence >= threshold:
    |      map disease -> explanation + treatment + prevention
    |
    `-- if confidence < threshold:
           ask for retake / expert verification
```

### 9.2 Disease Knowledge Base

Use a plain Python dictionary or JSON file.

Example shape:

```python
DISEASE_INFO = {
    "Healthy": {
        "explanation": "The leaf does not show strong signs of the target diseases.",
        "treatment": ["No immediate treatment needed."],
        "prevention": [
            "Continue weekly monitoring.",
            "Maintain balanced nutrition and irrigation."
        ]
    },
    "Rust": {
        "explanation": "Rust commonly appears as orange-brown pustules on wheat leaves.",
        "treatment": [
            "Isolate highly affected areas if possible.",
            "Use a locally approved fungicide for rust management.",
            "Consult local agronomy guidance for exact product and dose."
        ],
        "prevention": [
            "Use resistant varieties where available.",
            "Avoid excessive nitrogen application.",
            "Scout surrounding fields early."
        ]
    },
    "Leaf_Blight": {
        "explanation": "Leaf blight often causes elongated brown lesions and drying tissue.",
        "treatment": [
            "Remove heavily damaged leaves if feasible.",
            "Use a locally approved fungicide based on extension advice."
        ],
        "prevention": [
            "Improve field sanitation.",
            "Avoid prolonged leaf wetness when possible."
        ]
    },
    "Powdery_Mildew": {
        "explanation": "Powdery mildew appears as white powder-like growth on leaf surfaces.",
        "treatment": [
            "Use a locally approved fungicide if spread is active.",
            "Reduce dense canopy conditions where possible."
        ],
        "prevention": [
            "Improve airflow in the crop canopy.",
            "Monitor early-stage patches."
        ]
    },
    "Spot_Blotch": {
        "explanation": "Spot blotch often shows dark brown lesions that expand across the leaf.",
        "treatment": [
            "Apply locally recommended disease management practices.",
            "Remove severely affected material where practical."
        ],
        "prevention": [
            "Use healthy seed and resistant varieties if available.",
            "Avoid carrying infected residue between cycles."
        ]
    }
}
```

### 9.3 Important Safety Note

For the hackathon:

- provide treatment categories and practical next steps
- avoid claiming exact pesticide dosage unless verified for the farmer's region
- add a note: `Confirm final product choice with local agriculture extension guidance`

That keeps the system useful without pretending legal agronomy authority.

---

## 10. Disease Mapping System

### 10.1 What to Store

If location permission is granted, store:

- predicted disease
- confidence
- latitude
- longitude
- timestamp
- image id
- optional district or village name

### 10.2 Simple Database Tables

Recommended SQLite tables:

```sql
CREATE TABLE predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    image_path TEXT NOT NULL,
    disease TEXT NOT NULL,
    confidence REAL NOT NULL,
    stage TEXT NOT NULL,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE locations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    prediction_id INTEGER NOT NULL,
    latitude REAL NOT NULL,
    longitude REAL NOT NULL,
    region_name TEXT,
    FOREIGN KEY (prediction_id) REFERENCES predictions(id)
);
```

### 10.3 Mapping Features for MVP

Build only these first:

- point markers for predictions
- heatmap layer
- count by region
- last 7 days trend chart

### 10.4 Heatmap Logic

Use simple intensity mapping:

```text
Healthy -> 0.0
Rust -> 0.8
Leaf_Blight -> 0.7
Powdery_Mildew -> 0.6
Spot_Blotch -> 0.7
final_intensity = disease_severity_weight * confidence
```

### 10.5 Leaflet MVP

Frontend map flow:

1. fetch `/map/points`
2. fetch `/map/heatmap`
3. plot markers
4. overlay heatmap
5. show region summary cards

This is enough for a strong demo.

---

## 11. Backend Design

### 11.1 Recommended FastAPI Routes

Minimum routes:

- `POST /predict`
- `GET /health`
- `GET /map/points`
- `GET /map/heatmap`
- `GET /map/stats`

### 11.2 Backend Service Layout

Keep the API thin:

- route handles request validation
- service loads image and runs model
- decision layer enriches response
- repository layer writes to SQLite

### 11.3 Startup Behavior

On API startup:

1. load model weights once
2. warm up one dummy inference
3. keep models in memory
4. initialize SQLite connection

This avoids slow first prediction during demo.

### 11.4 Prediction Service Pseudocode

```python
def predict(image_bytes, latitude=None, longitude=None):
    image = load_image(image_bytes)

    quality = run_quality_checks(image)
    if not quality["ok"]:
        return build_retake_response(quality)

    tensor = preprocess_minimal(image)

    stage1_probs = stage1_model(tensor)
    healthy_score = stage1_probs["Healthy"]

    if healthy_score >= 0.80:
        result = build_healthy_response(healthy_score, quality)
    else:
        stage2_probs = stage2_model(tensor)
        top_label, top_score = argmax(stage2_probs)

        if top_score < 0.65:
            return build_low_confidence_response(stage2_probs, quality)

        decision = decision_engine.get(top_label, top_score)
        result = build_disease_response(top_label, top_score, decision, quality)

    save_prediction(result, latitude, longitude)
    return result
```

---

## 12. Frontend Design

### 12.1 MVP Screens

You only need four screens:

1. upload page
2. result card
3. disease advice section
4. map page

### 12.2 Upload UX

The upload page should:

- allow image upload from phone or desktop
- request optional location permission
- show loading state
- show retake message clearly

### 12.3 Result Card

Display:

- disease label
- confidence bar
- plain-language explanation
- treatment suggestions
- prevention tips
- image quality warning if relevant

### 12.4 Farmer-Friendly UX Rules

- use large buttons
- use simple language
- avoid jargon like "softmax confidence"
- prefer color coding
- support English first, then Telugu/Hindi labels as optional bonus

### 12.5 Bonus Features

Optional if time remains:

- voice guidance for result playback
- multilingual output
- offline cached last predictions
- district filter on the map

---

## 13. Recommended Folder Structure

This is the target structure I would build toward.

```text
code_verse/
|
|-- ARCHITECTURE.md
|-- frontend/
|   |-- package.json
|   |-- tailwind.config.js
|   |-- src/
|   |   |-- components/
|   |   |   |-- ImageUploader.jsx
|   |   |   |-- PredictionCard.jsx
|   |   |   |-- DiseaseAdvice.jsx
|   |   |   `-- DiseaseMap.jsx
|   |   |-- pages/
|   |   |   |-- Home.jsx
|   |   |   `-- MapPage.jsx
|   |   |-- services/
|   |   |   `-- api.js
|   |   `-- styles/
|   `-- public/
|
|-- backend/
|   |-- app/
|   |   |-- main.py
|   |   |-- api/
|   |   |   `-- routes_predict.py
|   |   |-- services/
|   |   |   |-- model_service.py
|   |   |   |-- decision_service.py
|   |   |   `-- map_service.py
|   |   |-- db/
|   |   |   |-- models.py
|   |   |   `-- repository.py
|   |   `-- schemas/
|   |       `-- prediction.py
|   `-- requirements.txt
|
|-- training/
|   |-- data/
|   |   |-- raw/
|   |   |-- processed/
|   |   `-- splits/
|   |-- configs/
|   |   `-- train.yaml
|   |-- scripts/
|   |   |-- prepare_data.py
|   |   |-- train_stage1.py
|   |   |-- train_stage2.py
|   |   `-- evaluate.py
|   |-- src/
|   |   |-- data/
|   |   |   |-- dataset.py
|   |   |   `-- augmentations.py
|   |   |-- models/
|   |   |   |-- efficientnet.py
|   |   |   `-- two_stage.py
|   |   |-- training/
|   |   |   |-- engine.py
|   |   |   |-- losses.py
|   |   |   `-- metrics.py
|   |   `-- inference/
|   |       |-- preprocess.py
|   |       `-- quality_checks.py
|   |-- checkpoints/
|   `-- notebooks/
|
|-- shared/
|   |-- disease_knowledge.py
|   `-- constants.py
|
`-- database/
    `-- wheat_disease.db
```

### 13.1 Small-Team Shortcut

If your team is very small, simplify:

- keep React optional
- use a single FastAPI app with Jinja templates
- keep SQLite local
- keep mapping as one Leaflet page

That is still a strong hackathon solution.

---

## 14. 24-Hour Hackathon Execution Plan

### 0 to 6 Hours

Goal: data and model setup

- collect and merge datasets
- finalize class mapping
- remove obvious duplicates/corrupt images
- create train/validation split
- implement augmentation pipeline
- fine-tune Stage 1 first

Deliverables:

- cleaned dataset table
- first working binary classifier
- baseline metrics screenshot

### 6 to 12 Hours

Goal: disease classifier and API

- train Stage 2 disease classifier
- evaluate confusion matrix
- export best weights
- build FastAPI `/predict`
- implement blur and darkness checks
- connect decision engine

Deliverables:

- working inference endpoint
- prediction JSON with recommendations

### 12 to 18 Hours

Goal: frontend and integration

- build upload UI
- connect API from frontend
- show result card and confidence bar
- add retake flow
- save predictions to SQLite
- add map page

Deliverables:

- end-to-end demo from upload to result
- first map view working

### 18 to 24 Hours

Goal: polish and presentation

- test on real phone images
- tune thresholds
- add screenshots and confusion matrix
- record demo flow
- prepare problem, solution, architecture, and impact slides
- add multilingual labels if time remains

Deliverables:

- stable demo
- presentation deck
- short explanation of why preprocessing is minimal and augmentation is heavy

---

## 15. Common Mistakes and How to Avoid Them

### Mistake 1: Heavy Preprocessing

Problem:

- smoothing and equalization remove disease texture

Fix:

- keep inference preprocessing minimal
- put robustness into augmentation

### Mistake 2: Reporting Accuracy Only

Problem:

- rare disease recall may be terrible even if accuracy looks good

Fix:

- track macro F1, recall per class, and confusion matrix

### Mistake 3: Ignoring Low Confidence

Problem:

- system gives a strong-looking answer when it is guessing

Fix:

- use confidence threshold and retake flow

### Mistake 4: Leakage Between Train and Validation

Problem:

- near-duplicate images make validation unrealistically high

Fix:

- split first
- deduplicate
- keep source metadata

### Mistake 5: Overbuilding the Map Too Early

Problem:

- team spends hours on mapping while model is not reliable

Fix:

- first get a stable upload -> predict -> advice loop
- then add storage and heatmap

### Mistake 6: Too Many Features

Problem:

- voice, multilingual, offline mode, and map all at once can break the MVP

Fix:

- ship core prediction first
- treat everything else as stretch

---

## 16. Recommended Demo Story

Use this during judging:

1. upload a healthy leaf image
2. show healthy result
3. upload a diseased leaf image
4. show disease label, confidence, advice
5. show low-quality image and retake message
6. show map with saved disease points
7. explain why minimal preprocessing preserves disease texture

This tells a strong real-world story.

---

## 17. What to Build First

If you start implementation right now, build in this order:

1. dataset merge and label cleanup
2. Stage 1 classifier
3. Stage 2 classifier
4. FastAPI `/predict`
5. quality check gate
6. decision engine
7. frontend upload and result UI
8. SQLite storage
9. map page

That order reduces risk and keeps the demo shippable.

---

## 18. Final Recommendation

For a 24-hour hackathon, the strongest practical version is:

- two-stage EfficientNetB0 pipeline
- minimal inference preprocessing
- heavy training augmentation
- FastAPI backend
- simple React frontend
- SQLite prediction logging
- Leaflet heatmap
- rule-based recommendation engine

If time becomes tight, cut complexity in this order:

1. keep map simple
2. skip multilingual support
3. skip voice support
4. keep two-stage if possible
5. if needed, fall back to one 5-class model

Do not cut:

- quality checks
- confidence threshold
- class balancing
- validation discipline

Those are what make the system believable in the real world.

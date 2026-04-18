# AlgoPulse AI

AI-assisted breast ultrasound analysis with:

- 3-class prediction: `Benign`, `Malignant`, `Normal`
- Grad-CAM heatmap for visual explanation
- Similar case retrieval with Qdrant
- Doctor and patient dashboards
- Voice output for patients using Vapi or browser speech fallback

This project is designed as an assistive tool for screening support and patient communication. It is not a replacement for a qualified doctor.

## Overview

AlgoPulse takes a breast ultrasound image, sends it to a Flask backend, runs a trained PyTorch model, generates an explanation heatmap, finds similar past cases, and returns the result to a React frontend.

The app has two views:

- Doctor view: detailed result, probabilities, heatmap, similar cases
- Patient view: simplified explanation with voice playback

## Features

- EfficientNet-B0 based image classification
- Grad-CAM overlay on the original uploaded image
- Similar-image search using Qdrant vector storage
- Click-to-open similar case images from the doctor dashboard
- Patient-friendly text explanation
- Voice assistant integration through Vapi
- Browser speech fallback when Vapi is not configured

## Tech Stack

### Frontend

- React
- JavaScript
- Axios
- CSS
- `@vapi-ai/web`

Frontend code lives in [algopulse-ui](/abs/path/c:/Users/Mithali satish/ethicare/algopulse-ui).

### Backend

- Flask
- Flask-CORS
- Python

Backend code lives in [backend](/abs/path/c:/Users/Mithali satish/ethicare/backend).

### AI / ML

- PyTorch
- Torchvision
- EfficientNet-B0
- Grad-CAM
- Pillow
- NumPy
- OpenCV

### Similar Case Search

- Qdrant

By default, Qdrant is used in local persistent mode through `backend/qdrant_data`. If `QDRANT_URL` and `QDRANT_API_KEY` are set, the project can use Qdrant Cloud instead.

### Voice

- Vapi Web SDK on the frontend
- Browser `speechSynthesis` fallback

## Project Structure

```text
ethicare/
├── algopulse-ui/      # Main React frontend
├── backend/           # Flask API, model, Grad-CAM, Qdrant integration
├── dataset/           # Additional dataset folder in repo root
├── requirements.txt   # Python dependencies for deployment
├── runtime.txt        # Python runtime version
└── README.md
```

Important backend files:

- [backend/app.py](/abs/path/c:/Users/Mithali satish/ethicare/backend/app.py:1): Flask API routes
- [backend/model.py](/abs/path/c:/Users\Mithali satish\ethicare\backend\model.py:1): model definition and inference
- [backend/gradcam.py](/abs/path/c:/Users/Mithali satish/ethicare/backend/gradcam.py:1): Grad-CAM implementation
- [backend/qdrant_db.py](/abs/path/c:/Users/Mithali satish/ethicare/backend/qdrant_db.py:1): Qdrant client and search
- [backend/store_dataset.py](/abs/path/c:/Users/Mithali satish/ethicare/backend/store_dataset.py:1): embed dataset images into Qdrant
- [backend/train.py](/abs/path/c:/Users/Mithali satish/ethicare/backend/train.py:1): training script

## How It Works

1. User uploads an ultrasound image in the React app.
2. The frontend sends the image to `POST /predict`.
3. The backend preprocesses the image.
4. The trained model predicts one of the 3 classes.
5. Grad-CAM highlights the region the model focused on.
6. The backend extracts an embedding and searches Qdrant for similar cases.
7. The frontend displays:
   - predicted label
   - confidence and class probabilities
   - heatmap
   - similar cases
   - recommendation and disclaimer
8. On the patient side, the result can also be read aloud.

## Model

The current model is a custom classifier built on top of EfficientNet-B0.

- Backbone: pretrained `efficientnet_b0`
- Output classes: `Benign`, `Malignant`, `Normal`
- Custom classifier head: `1280 -> 512 -> 128 -> 3`

The trained weights file used by the backend is:

- [backend/best_model.pth](/abs/path/c:/Users/Mithali satish/ethicare/backend/best_model.pth:1)

## Grad-CAM

Grad-CAM is implemented manually in the backend.

In simple terms:

- the model makes a prediction
- the backend computes which feature maps were most important
- those important regions are converted into a heatmap
- the heatmap is resized to the original image size
- the final overlay is sent back to the frontend as a base64 image

## Qdrant Similar Cases

Qdrant stores embeddings extracted from the model.

For each dataset image, the system stores:

- label
- image path
- vector embedding

During prediction:

- the uploaded image is embedded
- Qdrant returns the nearest stored vectors
- duplicate filenames are filtered out
- the frontend shows the top unique similar cases
- clicking a similar case opens the matched image through the backend

## Voice Assistant

The patient dashboard supports voice playback.

If these frontend environment variables are set:

- `REACT_APP_VAPI_API_KEY`
- `REACT_APP_VAPI_ASSISTANT_ID`

the app uses Vapi.

If they are missing, the app falls back to browser speech synthesis.

## Setup

### Prerequisites

- Python 3.11.x recommended
- Node.js and npm

### 1. Install Python Dependencies

From the repo root:

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

If you prefer using the backend virtual environment folder:

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\activate
pip install -r ..\requirements.txt
```

### 2. Start the Backend

```powershell
cd backend
python app.py
```

Backend runs on:

- `http://localhost:5000`

### 3. Install Frontend Dependencies

```powershell
cd algopulse-ui
npm install
```

### 4. Configure Optional Vapi

Create or update [algopulse-ui/.env](/abs/path/c:/Users/Mithali satish/ethicare/algopulse-ui/.env:1):

```env
REACT_APP_VAPI_API_KEY=YOUR_VAPI_PUBLIC_KEY
REACT_APP_VAPI_ASSISTANT_ID=YOUR_VAPI_ASSISTANT_ID
```

### 5. Start the Frontend

```powershell
cd algopulse-ui
npm start
```

Frontend runs on:

- `http://localhost:3000`

## Qdrant Dataset Indexing

If you want to build or rebuild the similar-case index:

```powershell
cd backend
python store_dataset.py
```

This reads dataset images, computes embeddings, and stores them in Qdrant.

## Model Performance

The old README listed:

- Accuracy: `94.2%`
- Precision: `93.8%`
- Recall: `94.5%`

Those exact numbers are not reproducible from the current repo as-is because the repository does not include the original evaluation script or test split that produced them.

A reproducible evaluation run on the current shipped model over the labeled dataset folders in `backend/Dataset`, excluding mask files, gave:

- Accuracy: `85.64%`
- Precision:
  - Macro: `86.55%`
  - Weighted: `85.86%`
- Recall:
  - Macro: `81.52%`
  - Weighted: `85.64%`
- F1:
  - Macro: `83.62%`
  - Weighted: `85.38%`

Per-class results from that run:

- Benign: Precision `84.39%`, Recall `94.05%`
- Malignant: Precision `88.07%`, Recall `73.81%`
- Normal: Precision `87.18%`, Recall `76.69%`

Because the original evaluation pipeline is not preserved in the repo, these should be treated as the current reproducible metrics for the shipped model, not a direct reproduction of the older README claims.

## Current Limitations

- The original README metrics are not fully traceable from the repo
- There are duplicate dataset folders (`Dataset` and `dataset`) in `backend`
- The included `test` folder is not organized as labeled class subfolders
- The app is assistive only and not suitable as a final medical decision system

## Future Improvements

- Train on a larger and more diverse dataset
- Add a proper fixed test set and reproducible evaluation pipeline
- Add confusion matrix and metrics reporting scripts
- Improve malignant recall
- Add multilingual voice support
- Add stronger authentication and patient history tracking
- Support production cloud deployment
- Integrate with hospital record systems

## Disclaimer

This project is for educational and assistive use only.

It does not replace a radiologist, oncologist, or any licensed medical professional. Always confirm results through proper clinical evaluation.

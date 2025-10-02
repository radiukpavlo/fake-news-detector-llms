# Fake News Detector

A containerized demo that trains and serves a transformer-based fake-news classifier. The stack is composed of:

- **frontend** – React SPA for training and inference
- **backend** – ASP.NET Core API that proxies requests to the ML service
- **ml-service** – FastAPI application wrapping Hugging Face models

## How to Launch

This project is designed to be run with Docker. Follow these steps to get the application up and running.

### 1. Prerequisites

- **Docker Desktop:** Ensure you have Docker Desktop installed and running, with support for Compose v2.

### 2. Clone the Repository

Clone this repository to your local machine:

```bash
git clone <repository-url>
cd <repository-directory>
```

### 3. Set Up Kaggle API (Optional)

The `ml-service` can automatically download the LIAR dataset from Kaggle. You have two options for this step:

- **Option A: Skip Dataset Download**
  If you do not want to download the dataset, you can create a dummy `kaggle.json` file. This will allow the application to run without requiring real Kaggle credentials.

  ```bash
  # Create the directory and an empty file with dummy credentials
  mkdir -p $HOME/.config/kaggle
  echo '{"username":"testuser","key":"testkey"}' > $HOME/.config/kaggle/kaggle.json
  ```

- **Option B: Use Your Kaggle Credentials**
  If you want to download the dataset, place your `kaggle.json` file (containing your Kaggle API username and key) in the `ml-service/.kaggle` directory.

### 4. Build and Run the Application

From the root of the repository, run the following command to build the Docker images and start the services:

```bash
docker compose up --build -d
```

The `-d` flag runs the containers in detached mode.

### 5. Access the Application

Once the containers are running, you can access the services at the following URLs:

| Service    | URL                     |
|------------|-------------------------|
| Frontend   | http://localhost:3000   |
| Backend    | http://localhost:5000   |
| ML Service | http://localhost:8000   |

## Application Usage

### Training a Model

1.  Open the frontend application at `http://localhost:3000`.
2.  Choose a model from the dropdown menu (e.g., `roberta-base`).
3.  If you have provided valid Kaggle credentials, you can check the "Download dataset from Kaggle" box.
4.  Click **Start Training** and monitor the progress.

Metrics and plots will be saved to the `ml-service/output/<model-name>/` directory.

### Predicting

Once a model is trained, you can enter a news headline or snippet into the text box and click **Analyse** to get a prediction and confidence score.

## Development Workflow

If you prefer to run the services without Docker, follow these steps:

### Backend

```bash
# From the backend directory
cd backend
dotnet run --project src/RealFakeNews.csproj
```

### ML Service

```bash
# From the ml-service directory
cd ml-service
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

### Frontend

```bash
# From the frontend/app directory
cd frontend/app
npm install
npm start
```

Remember to update `REACT_APP_API_URL` in `frontend/app/.env` if your backend is not running on `http://localhost:5000`.

## Project Structure

```
.
├── backend/
│   ├── src/              # ASP.NET Core source code
│   └── Dockerfile
├── frontend/
│   ├── app/              # React application source code
│   └── Dockerfile
├── ml-service/           # FastAPI and Hugging Face ML service
└── docker-compose.yml
```

## Known Limitations

- Training large models on a CPU can be slow. A GPU is recommended but not required.
- The React application has basic styling. Feel free to replace it with a design system of your choice.

## Enhancements (2025-10-01)

- XAI endpoints: `/explain` (SHAP, LIME, IG)
- Projection endpoint: `/project` (UMAP/t-SNE)
- Export endpoint: `/report/download/{model}`
- Frontend Explain & Projection panels rendered in `App.jsx`
- Backend proxy wired via `ML_BASE_URL`
- Docker Compose with healthchecks
- Tests & pinned dependencies
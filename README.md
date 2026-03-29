# Sleep Disorder Prediction API

A REST API built with Flask that predicts sleep disorders using a pre-trained Logistic Regression machine learning model.

## Table of Contents

- [Overview](#overview)
- [Tech Stack](#tech-stack)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Running the Application](#running-the-application)
- [API Reference](#api-reference)
  - [Predict Sleep Disorder](#predict-sleep-disorder)
- [Example Usage](#example-usage)
- [License](#license)

## Overview

This API accepts health and lifestyle metrics as input and returns a binary prediction indicating whether a sleep disorder is present. It uses a pre-trained Logistic Regression model (`LR_Model.joblib`) along with standard preprocessing (feature scaling and label encoding) before inference.

## Tech Stack

| Component        | Details                          |
|------------------|----------------------------------|
| Language         | Python 3.x                       |
| Framework        | Flask 3.0.3                      |
| CORS             | Flask-CORS                       |
| ML Model         | Logistic Regression (scikit-learn 1.4.2) |
| Model Storage    | Joblib 1.4.2                     |
| Data Processing  | Pandas 2.2.2, NumPy 1.26.4       |
| Production Server| Gunicorn 22.0.0                  |

## Getting Started

### Prerequisites

- Python 3.x
- pip

### Installation

```bash
# Clone the repository
git clone https://github.com/tamim-shadman/sleep_disorder_prediction_api.git
cd sleep_disorder_prediction_api

# Install dependencies
pip install -r requirements.txt
```

### Running the Application

**Development mode:**

```bash
python app.py
```

The server starts at `http://localhost:5000` with debug mode enabled.

**Production mode (Gunicorn):**

```bash
gunicorn app:app
```

## API Reference

### Predict Sleep Disorder

**Endpoint:** `POST /predict_sleep_disorder`

**Description:** Predicts whether a sleep disorder is present based on the provided health and lifestyle data.

**Request Body (JSON):**

| Field                    | Type   | Description                              |
|--------------------------|--------|------------------------------------------|
| `age`                    | number | Age of the individual                    |
| `sleep_duration`         | number | Average sleep duration in hours          |
| `quality_of_sleep`       | number | Self-reported sleep quality score        |
| `physical_activity_level`| number | Physical activity level score            |
| `stress_level`           | number | Self-reported stress level score         |
| `heart_rate`             | number | Resting heart rate (bpm)                 |
| `daily_steps`            | number | Average number of daily steps            |
| `gender`                 | string | Gender (e.g., `"Male"`, `"Female"`)      |
| `occupation`             | string | Occupation (e.g., `"Engineer"`)          |
| `bmi_category`           | string | BMI category (e.g., `"Normal"`, `"Overweight"`) |
| `blood_pressure`         | string | Blood pressure reading (e.g., `"120/80"`) |

**Response (JSON):**

| Field            | Type    | Description                                    |
|------------------|---------|------------------------------------------------|
| `sleep_disorder` | integer | `1` if a sleep disorder is predicted, `0` otherwise |

**Error Response (JSON):**

| Field   | Type   | Description          |
|---------|--------|----------------------|
| `error` | string | Error message detail |

## Example Usage

**Request:**

```bash
curl -X POST http://localhost:5000/predict_sleep_disorder \
  -H "Content-Type: application/json" \
  -d '{
    "age": 30,
    "sleep_duration": 7,
    "quality_of_sleep": 8,
    "physical_activity_level": 6,
    "stress_level": 3,
    "heart_rate": 70,
    "daily_steps": 8000,
    "gender": "Male",
    "occupation": "Engineer",
    "bmi_category": "Normal",
    "blood_pressure": "120/80"
  }'
```

**Response:**

```json
{
  "sleep_disorder": 0
}
```

## License

This project is licensed under the [Apache License 2.0](LICENSE).


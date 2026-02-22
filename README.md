# Housing Price Predictor — Model + FastAPI + HTML

A simple end-to-end project that trains a regression model to predict house prices and serves predictions via a FastAPI backend with a minimal HTML frontend.

- Model: Scikit-learn Linear Regression trained on Housing_cleaned.csv
- API: FastAPI with a /predict endpoint that accepts JSON input and returns a numeric price
- UI: index.html that posts form data to the API and displays the predicted price


## Project Structure

```
Project_2_Housing_price_predictor_project/
├─ Housing_cleaned.csv                 # Cleaned dataset used for training
├─ Housing_price_predictor.py          # Training script (scikit-learn + pickle)
├─ Housing_price_predictor.pkl         # Pickled trained model (expected by API)
├─ Housing_price_predict.pkl           # Alternate/older pickled model filename
├─ main.py                             # FastAPI application
├─ index.html                          # Frontend UI (served at "/")
├─ Housing_Price_Predictor.ipynb       # Notebook version of training
├─ Other_models/                       # Additional notebooks (Linear Regression, Random Forest, etc.)
└─ README.md                           # This file
```

Note about model filename: main.py expects the model file Housing_price_predictor.pkl. If you train the model with Housing_price_predictor.py as-is, it currently writes Housing_price_predict.pkl. Either:
- Rename Housing_price_predict.pkl to Housing_price_predictor.pkl, or
- Update the training script to save as Housing_price_predictor.pkl to match main.py.


## Setup

Requirements
- Python 3.8+
- Packages: fastapi, uvicorn, scikit-learn, numpy, pandas, matplotlib, pydantic

Recommended installation
```
# Create and activate a virtual environment (Windows)
python -m venv .venv
.venv\Scripts\activate

# Install dependencies
pip install fastapi uvicorn scikit-learn numpy pandas matplotlib
```


## Training the Model

The training pipeline (Housing_price_predictor.py):
1. Loads Housing_cleaned.csv
2. Drops the column Unnamed: 0
3. Reorders columns to set price as the target and the following as features:
   - area, bedrooms, bathrooms, stories, mainroad, guestroom, basement,
     hotwaterheating, airconditioning, parking, prefare, furnishingstatus
4. Splits data (train/test = 80/20)
5. Trains a LinearRegression model
6. Evaluates with MSE and R²
7. Serializes the model to a pickle file

Run training:
```
python Housing_price_predictor.py
```
This will create a pickle file in the project root. Ensure the final file is named Housing_price_predictor.pkl so the API can load it at startup.


## Running the API Server

Start the FastAPI app (development, with hot-reload):
```
uvicorn main:app --reload
```
- The HTML UI is served at: http://127.0.0.1:8000/
- Interactive API docs: http://127.0.0.1:8000/docs

On startup, main.py loads the pickle from Housing_price_predictor.pkl located in the project root. If it is missing, the app will raise a FileNotFoundError.


## API Details

Endpoint
- POST /predict

Request body (JSON)
```
{
  "area": number (> 0),
  "bedrooms": integer (0–4),
  "bathrooms": integer (0–4),
  "stories": integer (0–4),
  "mainroad": integer (0–4),
  "guestroom": integer (0–4),
  "basement": integer (0–4),
  "hotwaterheating": integer (0–4),
  "airconditioning": integer (0–4),
  "parking": integer (0–4),
  "prefare": integer (0–4),
  "furnishingstatus": integer (0–4)
}
```

Example
```
POST /predict
Content-Type: application/json

{
  "area": 1200,
  "bedrooms": 3,
  "bathrooms": 2,
  "stories": 1,
  "mainroad": 1,
  "guestroom": 0,
  "basement": 0,
  "hotwaterheating": 0,
  "airconditioning": 1,
  "parking": 1,
  "prefare": 0,
  "furnishingstatus": 2
}
```

Response
```
{
  "prediction": 123456.78
}
```

Validation
- Server-side validation uses Pydantic (see InputModel in main.py)
- area must be > 0; integer fields are constrained between 0 and 4


## Frontend (index.html)

- Served at the root path "/" by main.py
- Contains a form with the 12 model features and minimal client-side validation
- Submits JSON via fetch to POST /predict
- Displays the predicted price; currency formatting in the UI uses CAD by default

To use the UI:
1. Ensure the API is running (uvicorn main:app --reload)
2. Open http://127.0.0.1:8000/
3. Enter inputs and click "Estimate Price"


## Re-training or Updating the Model

- Modify the training script or notebooks as needed (e.g., feature engineering, algorithm changes)
- Re-run the script to produce an updated pickle
- Ensure the pickle filename is Housing_price_predictor.pkl in the project root
- Restart the API server to load the new model


## Troubleshooting

- Error: "Model pickle not found at .../Housing_price_predictor.pkl"
  - Cause: The expected pickle file is missing or differently named
  - Fix: Train the model and save/rename the file to Housing_price_predictor.pkl

- API returns 500 on /predict
  - Cause: Model predict error or invalid input types
  - Fix: Verify request JSON fields match the expected schema and are numeric; confirm the pickle matches the training features order

- UI does not update
  - Cause: JS error or server not reachable
  - Fix: Open browser dev tools console for errors; confirm the server runs at http://127.0.0.1:8000/


## Notes

- This project is for demonstration/educational purposes only; predictions are not intended for production use.
- Consider normalizing/standardizing features and performing more robust evaluation for real-world deployment.
- If serving the HTML from another origin, configure CORS in FastAPI.

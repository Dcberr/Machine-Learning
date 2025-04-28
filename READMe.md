# CO(GT) Prediction Model

## Overview
This project implements a Linear Regression model to predict Carbon Monoxide ground truth (CO(GT)) concentrations using the Air Quality UCI dataset. The model processes environmental sensor data, handles missing values, engineers features, and provides accurate CO predictions based on various air quality parameters.

## Features
- Data preprocessing and cleaning for the UCI Air Quality dataset
- Advanced feature engineering including interaction terms
- Outlier detection and removal using IQR method
- Missing value imputation with KNN
- Model evaluation with comprehensive metrics and visualizations
- Ready-to-use prediction function for new data

## Requirements
- Python 3.6+
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- joblib

## Installation

Clone this repository or download the source code. Then install the required dependencies:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn joblib
```

## Dataset

This project uses the Air Quality UCI dataset. Download the dataset from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Air+Quality) and place the `AirQualityUCI.csv` file in the project directory.

## Usage

### Running the Model Training

1. Ensure the `AirQualityUCI.csv` file is in the same directory as the script
2. Execute the main script:

```bash
python co_prediction_model.py
```

### Understanding the Output

The script will:
1. Load and preprocess the data
2. Train a Linear Regression model
3. Evaluate the model performance
4. Generate visualization plots
5. Save the trained model and necessary components

### Output Files

After running the script, the following files will be generated:

- **Model Files**:
  - `co_prediction_model.pkl`: The trained Linear Regression model
  - `co_scaler.pkl`: Feature standardization scaler
  - `co_model_features.pkl`: List of features used by the model

- **Visualization Files**:
  - `correlation_matrix.png`: Heatmap showing correlations between features
  - `actual_vs_predicted.png`: Scatter plot comparing actual vs. predicted CO values
  - `residuals_histogram.png`: Distribution of prediction errors
  - `residuals_scatter.png`: Residual analysis plot
  - `feature_importance.png`: Bar chart of feature importance

## Making Predictions with the Trained Model

To use the trained model for predictions on new data:

```python
import pandas as pd
import joblib

# Load the saved model components
model = joblib.load('co_prediction_model.pkl')
scaler = joblib.load('co_scaler.pkl')
features = joblib.load('co_model_features.pkl')

# Load your new data (must have the same feature columns)
new_data = pd.read_csv('your_new_data.csv', sep=';')

# Use the provided prediction function 
from co_prediction_model import predict_co
predictions = predict_co(new_data)
print(predictions)
```

## Model Performance

The model is evaluated using multiple metrics:
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- Mean Absolute Error (MAE)
- R-squared (R²)

Typical performance metrics on the test set include an R² value of approximately 0.7-0.8 and RMSE of around 0.3-0.5 ppm, depending on the specific data split.

## Feature Engineering

The model leverages several engineered features:
- Time-based features extracted from Date and Time columns
- Polynomial terms (e.g., squared values of key parameters)
- Interaction terms between related pollution measures
- Weather-pollution interaction features

## Troubleshooting

Common issues:
- **File not found**: Ensure `AirQualityUCI.csv` is in the correct directory
- **Missing values**: The script handles missing values, but excessive missing data may affect performance
- **Format issues**: The script addresses European decimal format (commas instead of dots)

If you encounter errors, check that your dataset matches the expected format with semicolon separators.

## Contributing

Contributions to improve the model are welcome. Please feel free to submit a pull request with your enhancements.

## License

This project is available under the MIT License.
  # Diabetes Prediction App
Welcome to the Diabetes Prediction project ! This project aims to develop a machine learning model to predict diabetes based on various health-related attributes. The project involves several stages, including data exploration, preprocessing, model training, evaluation, and deployment via a Streamlit application.


## Features

- 🔍 Instant diabetes risk assessment
- 📊 Visualization of input parameters
- 🧪 Based on clinically relevant features
- 📱 Mobile-friendly interface
- 🔄 User-friendly input validation

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/diabetes-prediction.git
   cd diabetes-prediction
   ```

2. Create a virtual environment and activate it:
   ```bash
   python -m venv venv
   
   # On Windows
   venv\Scripts\activate
   
   # On macOS/Linux
   source venv/bin/activate
   ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

2. Open your browser and go to `http://localhost:8501`

3. Enter the required health parameters and click on the "Predict" button to get your diabetes risk assessment.

## Project Overview

### 1. Introduction
Diabetes is a chronic disease impacting millions globally. This project focuses on developing a machine learning model to predict diabetes based on health-related attributes. The project involves data exploration, preprocessing, model training, and deployment through a Streamlit application.

### 2. Dataset Description
- **Source and Content**: The dataset contains 100,000 records with 9 features, including both numerical and categorical variables. The target variable indicates whether an individual has diabetes (1) or not (0).
- **Features**:
  - `gender`: Categorical (Male, Female)
  - `age`: Numerical (Years)
  - `hypertension`: Binary (0 = No, 1 = Yes)
  - `heart_disease`: Binary (0 = No, 1 = Yes)
  - `smoking_history`: Categorical (e.g., never, current, formerly, No Info, ever, not current)
  - `bmi`: Numerical (Body Mass Index)
  - `HbA1c_level`: Numerical (Hemoglobin A1c Level)
  - `blood_glucose_level`: Numerical (Blood Glucose Level)
  - `diabetes`: Binary (Target variable, 0 = No, 1 = Yes)

### 3. Data Exploration and Preprocessing
- **Data Exploration**: Initial inspection and summary statistics were generated to understand the dataset.
- **Preprocessing**:
  - **Categorical Encoding**: Encoded categorical variables.
  - **Feature Scaling**: Scaled numerical features using StandardScaler.
  - **Saved Data**: Preprocessed data saved as a CSV file.

### 4. Feature Selection
- **Correlation Analysis**: Identified features with strong relationships to the target variable.
- **Feature Importance**: Used Random Forest to rank features based on importance.
- **Variance Threshold**: Considered but all features were retained.

### 5. Model Building and Evaluation
- **Data Preparation**: Split dataset into training and testing sets.
- **Class Imbalance Handling**: Used SMOTE to oversample the minority class.
- **Model Training**: Random Forest classifier was trained and tuned using GridSearchCV.
- **Evaluation**: Assessed model performance using accuracy, precision, recall, F1-score, and confusion matrix.
- **Threshold Adjustment**: Adjusted decision threshold to improve recall for diabetic cases.

### 6. Model Deployment using Streamlit
- **Setup**: Instructions for setting up and running the Streamlit application.
- **Application Workflow**:
  - Import libraries and load the trained model.
  - Create a form to capture input features.
 

The prediction model was trained on the [Pima Indians Diabetes Dataset](https://www.kaggle.com/uciml/pima-indians-diabetes-database) using various machine learning algorithms. The best performing model was saved as a `.sav` file using Python's `pickle` module.

### Model Features

The model uses the following features for prediction:
- Pregnancies
- Glucose Level
- Blood Pressure
- Skin Thickness
- Insulin
- BMI (Body Mass Index)
- Diabetes Pedigree Function
- Age

## Project Structure

```
diabetes-prediction/
│
├── app.py                  # Streamlit application
├── model/
│   └── diabetes_model.sav  # Saved model file
├── data/
│   └── diabetes.csv        # Dataset used for training (optional)
├── requirements.txt        # Project dependencies
└── README.md               # Project documentation
```

## How It Works

1. The user inputs their health parameters through the Streamlit interface
2. The application preprocesses and validates the input data
3. The pre-trained model (loaded from the `.sav` file) makes a prediction
4. Results are displayed with appropriate visualizations and explanations


## Requirements

The application requires the following Python packages:
- streamlit>=1.18.0
- numpy>=1.20.0
- pandas>=1.3.0
- scikit-learn>=1.0.0
- matplotlib>=3.5.0
- seaborn>=0.11.0


## Disclaimer

This application is for educational purposes only and should not be used as a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of your physician or other qualified health provider with any questions you may have regarding a medical condition.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.



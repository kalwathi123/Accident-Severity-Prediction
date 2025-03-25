# Car Accident Severity Prediction

## Project Overview
A machine learning project to predict the severity of car accidents in the US using environmental and temporal factors. The model uses characteristics like weather conditions, road features, and geographic data to classify accidents into different severity categories.

## Problem Statement
To create a sophisticated predictive model that can effectively classify accident severity using a dataset containing various environmental and temporal factors. The goal is to improve proactive risk mitigation strategies, optimize resource allocation, and facilitate prompt emergency response across the country.

## Background & Motivation
- High incidence of car accidents in the US leading to significant loss of life, injuries, and economic losses
- Need for better prediction and prevention mechanisms
- Economic and social impact of accidents on individuals, families, and society
- Opportunity to improve emergency response through better severity prediction

## Data Source
- Dataset hosted by "Sobhan Moosavi"
- Collected from various APIs broadcasting traffic events
- Sources include US and state departments of transportation, law enforcement agencies, traffic cameras, and traffic sensors
- Original dataset contains over 7 million records
- Used a sampled version containing 500k records for this project

  ## Project Structure
- `src/`: Contains source code files including Streamlit app and Jupyter notebooks
- `App_Screenshots/`: Contains application screenshots showing various features and functionalities

## Required Files Download
Due to size limitations of GitHub, please download the following essential folders from OneDrive:

### Data and Model Files
1. Download 'data' folder: [Download Data Folder](https://buffalo.box.com/s/g8qhfenuf0dn0b2h8yvzd54567bhf0l4)
  - Contains the US accidents dataset (us_accidents_data.csv)
  
2. Download 'pickle' folder: [Download Pickle Folder](https://buffalo.box.com/s/2uo1g97adg8f3uv9v98cae1u9ukqt4z3)
  - Contains trained models and preprocessed data files
  - Includes Random Forest, KNN, and other model files

### Installation Steps
1. Clone this repository
2. Download both folders from the OneDrive links above  
3. Extract and place both folders in the root directory of the project


## Methodology

### Data Preprocessing
1. Dropping unwanted columns
2. Column merging
3. Removing records with erroneous values
4. Handling null values
5. Standardizing data format
6. Removing outliers
7. Handling datetime features
8. Feature modification
9. Feature encoding
10. Feature scaling

### Machine Learning Models Implemented
- Multiclass Logistic Regression
- Gaussian Naive Bayes
- K-Nearest Neighbors
- Decision Tree
- Random Forest
- AdaBoost

### Model Performance
Best performing model: Random Forest
- Accuracy: 73%
- Precision: 85%
- Recall: 73%
- F1-Score: 0.76

Hyperparameters:
- n_estimators: 500
- max_depth: 30

## Web Application
Built using Streamlit framework, the application allows users to:
- Input accident-related parameters
- Get severity predictions
- Visualize predictions on a map
- View severity distribution through graphs

### Required Input Features
- Location (Address)
- Distance
- Temperature
- Humidity
- Pressure
- Visibility
- Wind Speed
- Amenity
- Bump
- Event Date and Time
- Weather Condition
- Civil Twilight
- Wind Direction

## Installation & Usage

### Prerequisites
- Python 3.x
- Required libraries (install using pip):
  - pandas
  - numpy
  - scikit-learn
  - streamlit
  - plotly
  - geopy

### Running the Application
```bash
streamlit run streamlit_app.py
```

## Future Recommendations
1. Implement real-time monitoring and alerts
2. Scale the solution for larger datasets using distributed computing
3. Deploy continuous monitoring of model performance metrics
4. Implement automated retraining pipelines
5. Integrate with insurance company systems
6. Develop traffic regulation impact measurements

## Contributors
- Ajay Vijayakumar
- Shivaramakrishnan Rajendran 
- Mohammed Abdul Rahman Kalwathi Jahir Hussain 

## References
1. Dealing with Imbalanced Data - Towards Data Science
2. SMOTE for high-dimensional class-imbalanced data
3. scikit-learn Documentation
4. Comprehensive Guide to Multiclass Classification Metrics


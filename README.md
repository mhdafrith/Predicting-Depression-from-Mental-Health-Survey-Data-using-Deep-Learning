# Predicting Depression from Mental Health Survey Data using Deep Learning
https://drpvau5tc5o6dhruzza62m.streamlit.app

## Project Overview
This project aims to predict whether an individual is experiencing depression based on their responses to a mental health survey. It leverages deep learning techniques to analyze survey responses and determine the likelihood of depression.

## Features
- **Data Preprocessing & EDA:** Includes preprocessing steps and exploratory data analysis (EDA) performed in `preprs_eda_train.ipynb`.
- **Deep Learning Model:** The trained model is saved as `model.pth`.
- **Web Application:** A Streamlit-based web application (`web_app.py`) allows users to input survey responses and receive depression predictions.
- **Scalability:** The model is deployed using a `scaler.pkl` file to standardize input data.
- **Experiment Tracking:** `mlruns/` is used for model tracking.

## File Structure
```
.
├── .devcontainer/             # Dev container settings
├── Desktop/mini_project_6/
│   ├── mlruns/               # MLflow experiment tracking directory
│   ├── .gitignore            # Ignore unnecessary files
│   ├── model.pth             # Trained deep learning model
│   ├── preprs_eda_train.ipynb # Data preprocessing & EDA
│   ├── requirements.txt       # Required Python libraries
│   ├── scaler.pkl             # Scaler for data preprocessing
│   ├── web_app.py             # Streamlit web application
├── README.md                  # Project documentation
```

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/your-repo.git
   cd your-repo
   ```
2. Create a virtual environment and install dependencies:
   ```bash
   python -m venv env
   source env/bin/activate  # On Windows use `env\Scripts\activate`
   pip install -r requirements.txt
   ```

## Usage
### Running the Web Application
To launch the Streamlit web application, run:
```bash
streamlit run web_app.py
```
This will open a browser window where users can input survey responses and receive depression predictions.

### Training the Model
To retrain the model, execute `preprs_eda_train.ipynb` in Jupyter Notebook.

## Requirements
- Python 3.8+
- Streamlit
- PyTorch
- Scikit-learn
- Pandas
- NumPy
- Matplotlib

Install dependencies using:
```bash
pip install -r requirements.txt
```

## Contributions
Contributions are welcome! Feel free to fork the repository and submit pull requests.

## License
This project is open-source and available under the [MIT License](LICENSE).

## Author
Mohamed Afrith S 
GitHub:[https://github.com/mhdafrith]


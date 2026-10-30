# AI-Based Predictive Maintenance for Simulated Turbofan Engine Failure Prediction

**NASA C-MAPSS FD001 | Remaining Useful Life (RUL) Prediction | LSTM**

## 1. Project Overview

Predictive maintenance helps estimate when a machine or component may require maintenance before it fails. Instead of performing maintenance only after a failure, machine learning models can analyse historical sensor readings and identify degradation patterns.

This project focuses on predicting the **Remaining Useful Life (RUL)** of simulated turbofan engines using the NASA C-MAPSS FD001 dataset.

RUL represents the estimated number of operational cycles remaining before an engine reaches its end of life.

The project uses sensor readings collected over multiple operating cycles to train and evaluate machine learning and deep learning models. The main objective is to compare different approaches and identify a suitable model for RUL prediction.

> **Note:** This is an academic machine learning project based on simulated turbofan engine data. It is an experimental decision-support system and is not a certified aviation safety or real-world aircraft maintenance system.

## 2. Objectives

- Understand and explore the NASA C-MAPSS FD001 dataset.
- Perform data preprocessing and sensor selection.
- Calculate Remaining Useful Life (RUL) labels.
- Convert sequential sensor readings into time-series windows.
- Apply Principal Component Analysis (PCA) for dimensionality reduction.
- Implement machine learning and deep learning models.
- Compare model performance using RMSE, MAE, and NASA scoring function.
- Evaluate predictions on training, validation, and official test data.

## 3. Dataset Description

This project uses the **NASA C-MAPSS (Commercial Modular Aero-Propulsion System Simulation) dataset**, specifically the FD001 subset.

The dataset simulates the degradation of turbofan engines under different operating conditions.

### FD001 Dataset Characteristics

| Property | Description |
|---|---|
| Dataset | NASA C-MAPSS |
| Subset | FD001 |
| Training engines | 100 |
| Test engines | 100 |
| Operating conditions | 1 |
| Fault modes | 1 |
| Fault type | High-Pressure Compressor (HPC) degradation |
| Total columns | 26 |
| Sensor measurements | 21 |
| Operating settings | 3 |

### Dataset Files

- `train_FD001.txt` – Contains run-to-failure sensor readings for training engines.
- `test_FD001.txt` – Contains sensor readings for test engines up to their final recorded cycles.
- `RUL_FD001.txt` – Contains the actual remaining useful life values for the test engines.

### Dataset Sources

- [NASA C-MAPSS Dataset](https://data.nasa.gov/dataset/cmapss-jet-engine-simulated-data)
- [NASA Turbofan Engine Degradation Simulation Data](https://phm-datasets.s3.amazonaws.com/NASA/6.+Turbofan+Engine+Degradation+Simulation+Data+Set.zip)

The raw dataset is not included in this repository. Download it from the source and place the FD001 files inside `data/raw/`.

## 4. Project Workflow

The project follows these main steps:

1. Dataset loading
2. Exploratory Data Analysis (EDA)
3. Data preprocessing
4. RUL label generation
5. Sensor selection
6. Train-validation splitting
7. Feature scaling
8. Time-series window generation
9. PCA experiments
10. Model training
11. Model evaluation
12. Actual vs predicted RUL comparison

## 5. Data Preprocessing

### 5.1 Understanding the Dataset

Each row represents sensor readings for an engine at a particular operating cycle.

The dataset contains:
- Engine ID
- Operational cycle
- Three operational settings
- Twenty-one sensor measurements

### 5.2 RUL Calculation

For the training dataset, RUL is calculated using the last recorded cycle of each engine.

**Formula:**

`RUL = Maximum Engine Cycle - Current Engine Cycle`

For example, if an engine has a maximum recorded cycle of 200 and the current cycle is 150:

`RUL = 200 - 150 = 50 cycles`

### 5.3 RUL Clipping

The RUL values are capped at 125 cycles.

`RUL = min(RUL, 125)`

This is done because very early engine life may contain relatively stable sensor readings. Capping RUL limits the influence of large early-life labels and focuses the model more on the degradation period.

### 5.4 Sensor Selection

The original 14-sensor experiment uses the following sensor measurements:

`(2, 3, 4, 7, 8, 9, 11, 12, 13, 14, 15, 17, 20, 21)`

Some sensors have constant or nearly constant readings and provide limited useful information for the selected experiment. Sensor selection helps reduce unnecessary input features.

### 5.5 Train-Validation Split

The training engines are divided into:
- 80% training engines
- 20% validation engines

The split is performed by engine ID rather than randomly splitting individual rows. This prevents windows from the same engine from appearing in both training and validation sets.

### 5.6 Feature Scaling

MinMaxScaler is used to scale sensor features.

The scaler is fitted only on the training-engine data and then applied to validation and test data. This prevents information from validation and test data from influencing the training preprocessing.

### 5.7 Time-Series Window Generation

The sensor readings are converted into sequences of 30 consecutive cycles.

- Window size: 30 cycles
- Stride: 1 cycle
- Number of selected sensors: 14

Each training sample therefore has the shape:

`(30, 14)`

The model uses the previous 30 cycles of sensor readings to predict RUL at the target cycle.

For the official test dataset, the final 30-cycle window of each engine is used to predict its RUL.

## 6. Why We Used LSTM Instead of a Simple RNN

A Recurrent Neural Network (RNN) is designed to process sequential data. It uses information from previous time steps to help make predictions at the current time step.

However, a basic RNN can suffer from the **vanishing gradient problem** when learning long sequences. This can make it difficult to retain important information from earlier time steps.

In our project, turbofan engine degradation happens gradually over multiple operating cycles. Therefore, information from previous cycles can be useful for predicting the engine's Remaining Useful Life.

We selected **Long Short-Term Memory (LSTM)** because it is a special type of RNN designed to learn long-term dependencies more effectively.

### LSTM Gates

LSTM uses a memory cell and three main gates:

- **Forget Gate:** Decides which information from the previous memory should be removed.
- **Input Gate:** Controls which new information should be stored in memory.
- **Output Gate:** Controls which information should be used to produce the current output.

These gates help the network manage information across time steps and reduce the vanishing gradient problem.

### RNN vs LSTM

| Feature | Simple RNN | LSTM |
|---|---|---|
| Memory | Basic hidden state | Hidden state and cell state |
| Gates | No gates | Forget, input, and output gates |
| Long-term dependencies | Difficult to learn | Better at learning |
| Vanishing gradient | More affected | Helps reduce the problem |
| Complexity | Simpler | More complex |

### Why LSTM is Suitable for This Project

1. **Sequential data:** It can learn patterns from sensor readings across multiple engine cycles.
2. **Long-term dependencies:** It can retain useful information about gradual engine degradation.
3. **Memory mechanism:** Its gates control what information should be remembered, updated, or forgotten.
4. **RUL prediction:** It can learn temporal degradation patterns that help estimate the engine's Remaining Useful Life.

**Conclusion:** We used LSTM because RUL prediction depends on changes in sensor readings over time. Its memory cells and gates make it more suitable than a basic RNN for learning long-term degradation patterns. However, LSTM is not guaranteed to perform better on every dataset, so model performance must be evaluated experimentally.

## 7. Models Implemented

The project compares traditional machine learning and deep learning approaches.

### 7.1 Ridge Regression – 14 Sensors

Ridge Regression is a linear regression model with L2 regularization. It is used as a baseline model to understand how well a relatively simple approach predicts RUL.

### 7.2 Ridge Regression – PCA

PCA is applied to reduce the number of input features. Ridge Regression is then trained using the reduced feature representation.

### 7.3 LSTM – 14 Sensors

The main LSTM model uses the original 14 selected sensor features.

**Architecture:**

`Input (30, 14) → LSTM (64) → Dropout (0.2) → LSTM (32) → Dense (32, ReLU) → Dense (1)`

The model receives 30 cycles of sensor readings and predicts one RUL value.

### 7.4 SimpleRNN – PCA

A SimpleRNN model is trained using PCA-transformed sensor features. This provides a comparison between a basic recurrent network and LSTM.

### 7.5 LSTM – PCA

An LSTM model is trained using PCA-transformed sensor features to evaluate whether dimensionality reduction improves or affects prediction performance.

## 8. Principal Component Analysis (PCA)

Principal Component Analysis is a dimensionality reduction technique. It transforms the original features into a smaller number of new features called principal components.

The purpose of PCA in this project is to:
- Reduce the number of input features.
- Remove redundant information.
- Retain as much data variation as possible.
- Compare models trained on original and reduced feature representations.

In the controlled PCA experiment, 15 non-constant sensor and operating-setting features are transformed into 10 principal components.

The 10 components retain approximately **95.16% of the explained variance**.

Explained variance represents how much of the original data's variation is retained. It is not the prediction accuracy of the model.

PCA is fitted only using training-engine data to avoid data leakage.

## 9. Experimental Setup

The controlled comparison follows these conditions:

- Dataset: NASA C-MAPSS FD001
- Training-validation split: 80/20 by engine ID
- Window size: 30 cycles
- RUL cap for training labels: 125
- Neural network random seeds: 42, 43, and 44
- Ridge Regression: deterministic evaluation
- Official test set: 100 engines
- Scaling and PCA: fitted using training-engine data only

Neural network results are reported as mean ± sample standard deviation across three random seeds. Ridge Regression results are deterministic.

## 10. Evaluation Metrics

Three evaluation metrics are used to compare model performance.

### 10.1 Root Mean Squared Error (RMSE)

RMSE measures the square root of the average squared difference between actual and predicted RUL values.

`RMSE = sqrt(mean((Actual RUL - Predicted RUL)^2))`

A lower RMSE indicates that predictions are closer to the actual values, with larger errors receiving greater weight.

### 10.2 Mean Absolute Error (MAE)

MAE measures the average absolute difference between actual and predicted RUL values.

`MAE = mean(abs(Actual RUL - Predicted RUL))`

A lower MAE indicates better average prediction accuracy.

### 10.3 NASA Scoring Function

The NASA scoring function is an asymmetric error metric used in the PHM08 turbofan engine benchmark.

It penalizes late predictions more heavily than early predictions because predicting a longer remaining life than the actual value can delay maintenance.

Let:

`d = Predicted RUL - Actual RUL`

The score for each prediction is:

- If `d < 0`: `score = exp(-d / 13) - 1`
- If `d >= 0`: `score = exp(d / 10) - 1`

The total NASA score is the sum of the individual scores.

**A lower NASA score is better.**

The score is calculated in this project using the benchmark scoring function; it is not an award or certification from NASA.

## 11. Controlled Official Test Results

The following results are from the controlled comparison on the official FD001 test set.

Neural network results are mean ± sample standard deviation across seeds 42, 43, and 44. Ridge Regression results are deterministic.

| Model | RMSE | MAE | NASA Score |
|---|---:|---:|---:|
| 14-Sensor Ridge | 21.6072 | 17.3268 | 1087.1635 |
| PCA Ridge | 21.7092 | 17.4440 | 1079.8589 |
| 14-Sensor LSTM | **14.8876 ± 0.1792** | **11.1554 ± 0.1757** | **380.3457 ± 13.2517** |
| PCA SimpleRNN | 19.1310 ± 1.1944 | 14.0696 ± 1.0689 | 988.6472 ± 227.4474 |
| PCA LSTM | 15.4767 ± 0.6264 | 11.7092 ± 0.2079 | 494.5570 ± 159.9626 |

### Results Interpretation

- The 14-sensor LSTM achieved the best RMSE, MAE, and NASA score among the compared models.
- LSTM performed better than SimpleRNN in the PCA-based comparison.
- PCA-based LSTM performed better than PCA-based SimpleRNN in this experiment.
- Ridge Regression provides a baseline for comparing the deep learning models.
- The results indicate that the original 14-sensor LSTM performed better than the PCA-based models in this controlled experiment.

These results apply to the specified FD001 dataset and experimental setup. They should not be treated as a guarantee of performance on real aircraft engines.

## 12. Training, Validation, and Test Evaluation

The project evaluates model predictions across three splits:

- **Training:** Used to fit the model. Training metrics measure performance on data used during model fitting.
- **Validation:** Used to evaluate model performance on separate training engines and support model comparison.
- **Test:** Used for final evaluation on the official 100-engine FD001 test set with the provided RUL labels.

Actual vs Predicted RUL comparisons and split-level metrics are saved in the `reports/` directory.

Training results are in-sample and should not be compared directly with validation or test results as evidence of generalization.

## 13. Project Structure

The main project structure is:

```text
mini_project/
│
├── app/
│
├── data/
│   ├── raw/
│   │   ├── train_FD001.txt
│   │   ├── test_FD001.txt
│   │   └── RUL_FD001.txt
│   │
│   └── processed/
│
├── models/
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_baseline_lstm.ipynb
│   ├── 03_pca_experiments.ipynb
│   └── 04_fair_model_comparison.ipynb
│
├── reports/
│   ├── figures/
│   ├── fd001_fair_comparison.csv
│   ├── fd001_fair_comparison.md
│   ├── fd001_rul_predictions_train.csv
│   ├── fd001_rul_predictions_validation.csv
│   ├── fd001_rul_predictions_test.csv
│   ├── fd001_rul_split_metrics.csv
│   ├── pca_model_comparison.csv
│   └── results.md
│
├── scripts/
│   ├── audit_compare_fd001.py
│   ├── compare_rul_splits.py
│   ├── run_pca_experiments.py
│   ├── step1_label_train.py
│   └── verify_step1.py
│
├── src/
│   └── predict.py
│
├── tests/
│   ├── test_fd001_experiment.py
│   └── test_predict.py
│
└── README.md
```

Some generated files and additional folders may not be shown in this simplified structure.

## 14. Notebook Description

| Notebook | Purpose |
|---|---|
| `01_data_exploration.ipynb` | Dataset exploration, RUL calculation, preprocessing, and initial baseline experiments |
| `02_baseline_lstm.ipynb` | Baseline LSTM implementation and training |
| `03_pca_experiments.ipynb` | PCA-based feature reduction and model experiments |
| `04_fair_model_comparison.ipynb` | Controlled comparison of models and split-level prediction analysis |

`04_fair_model_comparison.ipynb` is the main notebook for the controlled model comparison.

## 15. Technologies Used

- **Python** – Main programming language
- **NumPy** – Numerical operations
- **Pandas** – Data manipulation and analysis
- **Matplotlib** – Data visualization
- **Seaborn** – Statistical visualization
- **Scikit-learn** – Preprocessing, PCA, and Ridge Regression
- **TensorFlow / Keras** – LSTM and SimpleRNN implementation
- **Jupyter Notebook** – Experimentation and analysis
- **Git and GitHub** – Version control

## 16. Installation and Setup

### Prerequisites

- Python 3.12
- Git
- VS Code or another Python-compatible IDE
- Jupyter Notebook

### Step 1: Clone the Repository

```bash
git clone https://github.com/SamyaArora/mini_project.git
cd mini_project
```

### Step 2: Create a Virtual Environment

On Windows PowerShell:

```powershell
py -3.12 -m venv .venv
```

### Step 3: Activate the Environment

```powershell
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, use:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

### Step 4: Install Dependencies

If a `requirements.txt` file is available:

```bash
pip install -r requirements.txt
```

Otherwise, install the main libraries:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn tensorflow jupyter
```

### Step 5: Download the Dataset

Download the NASA C-MAPSS dataset and extract the FD001 files into:

```text
data/raw/
```

The directory should contain:

```text
train_FD001.txt
test_FD001.txt
RUL_FD001.txt
```

### Step 6: Run the Notebooks

Start Jupyter Notebook:

```bash
jupyter notebook
```

Open the notebooks in the `notebooks/` directory.

For the controlled model comparison, open:

```text
04_fair_model_comparison.ipynb
```

Run the notebook cells in order.

## 17. Limitations

- The project uses simulated turbofan engine data rather than live aircraft sensor data.
- The experiments focus only on FD001, which contains one operating condition and one fault mode.
- The dataset is relatively small compared with many real-world industrial datasets.
- The results depend on the selected preprocessing steps, model architecture, and hyperparameters.
- LSTM predictions are estimates and should not be used as the sole basis for real-world maintenance decisions.
- The project is an academic experimental system and has not been certified for aviation use.

## 18. Future Improvements

Possible future improvements include:

- Experimenting with FD002, FD003, and FD004 datasets.
- Comparing BiLSTM, GRU, and CNN-LSTM architectures.
- Hyperparameter tuning.
- Testing additional window sizes.
- Developing a user interface for entering sensor readings and displaying RUL predictions.
- Exploring explainable AI methods to understand sensor contributions.
- Evaluating the system on additional datasets.

## 19. References

1. Saxena, A., Goebel, K., Simon, D., and Eklund, N. (2008). *Damage Propagation Modeling for Aircraft Engine Run-to-Failure Simulation*. International Conference on Prognostics and Health Management (PHM).

2. NASA Prognostics Center of Excellence. *C-MAPSS Turbofan Engine Degradation Simulation Data*. NASA.

3. Hochreiter, S., and Schmidhuber, J. (1997). *Long Short-Term Memory*. Neural Computation, 9(8), 1735–1780.

4. [NASA C-MAPSS Dataset – Google Scholar](https://scholar.google.com/scholar?q=NASA+C-MAPSS+turbofan+engine+degradation+dataset)

5. [LSTM for Remaining Useful Life Prediction – Google Scholar](https://scholar.google.com/scholar?q=LSTM+remaining+useful+life+prediction+turbofan+engine)

---

## 20. Author

**Project:** AI-Based Predictive Maintenance for Simulated Turbofan Engine Failure Prediction

**Dataset:** NASA C-MAPSS FD001

**Approach:** Machine Learning, Deep Learning, LSTM, and PCA

**Institution:** GLA University, Mathura

---

*This project was developed for academic and learning purposes.*
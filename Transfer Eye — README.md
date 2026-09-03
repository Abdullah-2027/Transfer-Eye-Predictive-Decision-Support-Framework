# ⚽ Transfer Eye

**A Data-Driven Football Transfer Analysis and Success Prediction System**

Transfer Eye is a machine learning-based football analytics system designed to evaluate player performance and analyze the likelihood of transfer success.

The project combines historical transfer records, player performance statistics, injury history, loan history, and club context to create position-aware player evaluations and data-driven transfer insights.

Rather than evaluating every footballer using the same criteria, Transfer Eye uses **role-specific performance metrics and models** for goalkeepers, defenders, midfielders, and attackers.

---

## 🎯 Project Objectives

Football transfers involve significant financial and sporting risk. A player's performance at one club does not necessarily translate directly to success at another.

Transfer Eye aims to support transfer analysis by combining multiple aspects of a player's history:

- Player performance statistics
- Transfer and market-value information
- Injury history
- Loan history and career stability
- Playing time
- Player age
- Club and league context
- Position and tactical role

The system processes these factors to provide a structured, data-driven assessment of players and historical transfers.

---

## 🔍 What Transfer Eye Does

Transfer Eye provides several layers of football analytics:

### Transfer Success Prediction
Machine learning models are trained on historical transfers to identify patterns associated with successful and unsuccessful transfers.

### Position-Specific Player Evaluation
Players are evaluated differently depending on their position and role:

- 🧤 **Goalkeepers**
- 🛡️ **Defenders**
- 🎯 **Midfielders**
- ⚡ **Attackers**

Each group uses performance indicators appropriate to its responsibilities.

### Performance Index
Player statistics are transformed into role-aware performance scores using feature weights calibrated from historical data.

### Injury Analysis
Historical injury records are converted into a **Season Injury Index**, incorporating:

- Days unavailable
- Matches missed
- Injury severity

### Loan Stability Analysis
Loan history is transformed into a **Weighted Loan Index**, considering:

- Number of loans
- Player age
- Loan destination quality
- Loans outside major European leagues

### Explainable AI
SHAP (SHapley Additive exPlanations) is used to identify which performance features contribute most strongly to model predictions.

---

## 🏗️ System Pipeline

```text
Transfer / Loan / Injury Data
            │
            ▼
      Data Collection
            │
            ▼
     Data Cleaning &
     Standardization
            │
            ▼
   Missing Data Handling
            │
            ▼
     Feature Engineering
       /            \
      ▼              ▼
Injury Index      Loan Index
      \              /
       └──────┬─────┘
              ▼
     Position & Role
       Segmentation
              │
              ▼
     Performance Index
              │
              ▼
      Machine Learning
       Model Comparison
              │
              ▼
      Gradient Boosting
              │
              ▼
      SHAP Explainability
              │
              ▼
       Transfer Eye App
```

---

## 📊 Data Collection

The project combines football data from sources including:

- **Transfermarkt** — transfers, loans, market values, and injury history
- **Stathead** — player performance statistics

Automated scraping pipelines were developed using Selenium and BeautifulSoup, with multithreading used to process multiple leagues and players concurrently.

The transfer collection covers the major European leagues, including:

- Premier League
- LaLiga
- Bundesliga
- Serie A
- Ligue 1

Collected transfer information includes fields such as:

`PlayerName`, `Position`, `Season`, `LeftClub`, `JoinedClub`, `MarketValue`, and `Fee`.

---

## 🧠 Feature Engineering

Raw football statistics are converted into higher-level indicators that better represent player performance and transfer risk.

### Injury Index

The injury model combines three components:

```text
Season Injury Index =
    0.50 × Days Lost
  + 0.20 × Games Missed
  + 0.30 × Injury Severity
```

This gives greater importance to long-duration and severe injuries.

### Weighted Loan Index

The loan index measures career stability based on:

```text
Weighted Loan Index =
    Age Component
  + Number of Loans
  + Loans Outside Top Leagues
  + Average Destination Rank
```

The resulting score ranges from **0 to 1**, where higher values indicate greater instability.

---

## ⚽ Position-Specific Analytics

A central design decision in Transfer Eye is that players should not be evaluated using identical criteria.

### 🧤 Goalkeepers

Key metrics include:

- Shots on Target Against
- Save Percentage
- Clean Sheet Percentage

Bootstrap calibration is used to estimate the relative importance of these features.

### 🛡️ Defenders

Defenders are evaluated using metrics related to:

- Defensive actions
- Mistake rate
- Passing
- Ball carrying
- Progressive actions
- Attacking contribution
- Duel efficiency

### 🎯 Midfielders

Midfielders are further segmented into role cohorts such as:

- Central
- Defensive
- Attacking

Their evaluation considers possession control, ball progression, defensive contribution, creativity, attacking contribution, and mistake rate.

### ⚡ Attackers

Attackers are divided into:

- Centre-Forwards
- Wingers

Their role-specific evaluation includes metrics such as:

- Goals
- Assists
- Key passes
- Progressive passes
- Crossing productivity

This allows a striker and winger to be evaluated according to their actual responsibilities rather than using a single universal scoring formula.

---

## 🤖 Machine Learning

Transfer Eye evaluates multiple machine learning algorithms rather than relying on a single model.

Models evaluated include:

- Gradient Boosting
- Random Forest
- AdaBoost
- Logistic Regression
- L1 Logistic Regression
- SVM (RBF)
- SVM (Linear)
- Neural Networks
- XGBoost
- LightGBM

Models are evaluated using repeated bootstrap experiments and metrics including:

- Accuracy
- Precision
- Recall
- AUC-ROC

### Example — Defender Model Results

| Model | Accuracy | Precision | Recall | AUC-ROC |
|---|---:|---:|---:|---:|
| **Gradient Boosting** | **0.823** | **0.814** | **0.798** | **0.887** |
| XGBoost | 0.821 | 0.809 | 0.795 | 0.884 |
| Random Forest | 0.815 | 0.801 | 0.789 | 0.878 |
| Neural Network | 0.808 | 0.794 | 0.782 | 0.871 |
| SVM (RBF) | 0.792 | 0.775 | 0.768 | 0.856 |
| Logistic Regression | 0.758 | 0.742 | 0.751 | 0.823 |

Gradient Boosting demonstrated the strongest overall performance in the model comparison and is used for subsequent explainability analysis.

---

## 🔬 Explainable AI with SHAP

Prediction accuracy alone is not sufficient for a football decision-support system.

Transfer Eye therefore uses **SHAP** to explain how individual performance features influence transfer-success predictions.

For example, the defender analysis identified features such as:

1. Defensive Actions
2. Mistake Rate
3. Passes Completed per 90
4. Total Carry Distance
5. Progressive Actions
6. Attacking Contributions
7. Duel Efficiency

This provides insight into **why** the model reaches a prediction rather than presenting only a probability.

---

## 💻 Transfer Eye Application

The project includes an interactive application for exploring the final Transfer Eye analysis.

> **Add screenshots of the application here.**

Example:

```markdown
![Transfer Eye Application](assets/app_screenshot.png)
```

The interface can be used to present player information, performance indicators, and model outputs in a more accessible format than the raw datasets.

---

## 📁 Repository Structure

```text
Transfer-Eye/
│
├── README.md
├── requirements.txt
│
├── app/
│   └── football_transfer_app.py
│
├── src/
│   ├── data_collection/
│   │   └── transfermarkt_loan_scraper.py
│   │
│   └── modeling/
│       └── defender_model_pipeline_example.py
│
├── data/
│   ├── raw/
│   └── processed/
│
├── results/
│   ├── model_comparison.csv
│   ├── defenders/
│   ├── midfielders/
│   ├── attackers/
│   └── goalkeepers/
│
├── assets/
│   └── app_screenshot.png
│
└── docs/
    └── Transfer-Eye_Documentation.pdf
```

---

## Code Examples

This repository contains representative examples of the main Transfer Eye pipeline rather than every position-specific implementation.

The complete project applies the same overall methodology across goalkeepers, defenders, midfielders, and attackers, with position-specific features and role definitions.

### `src/data_collection/loansStatsTransfersThreads.py`

Example of the data-collection pipeline used in Transfer Eye.

It demonstrates:

- Automated Transfermarkt scraping using Selenium and BeautifulSoup
- Multi-league and multi-season data collection
- Parallel processing
- Cookie handling and browser automation
- Retry and timeout handling
- Loan-history extraction
- Structured CSV generation

### `src/modeling/pipeline_example.py`

Representative machine-learning pipeline using the defender dataset.

It demonstrates:

- Position/role-aware bootstrap resampling
- Train/test splitting with stratification
- Feature standardization
- Comparison of multiple machine-learning algorithms
- Repeated bootstrap evaluation
- Accuracy, Precision, Recall, and AUC-ROC evaluation
- Aggregation and ranking of model performance

The same modeling framework was adapted for other playing positions using their corresponding position-specific features.

For full methodology, feature definitions, position-specific pipelines, and experimental results, see:

`docs/Transfer-Eye_Documentation.pdf`


## 🛠️ Technologies

**Programming & Data Processing**

- Python
- Pandas
- NumPy

**Machine Learning**

- Scikit-learn
- XGBoost
- LightGBM

**Explainable AI**

- SHAP

**Data Collection**

- Selenium
- BeautifulSoup
- WebDriver Manager

**Data Visualization & Analysis**

- Matplotlib
- Statistical bootstrapping

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone <repository-url>
cd Transfer-Eye
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the application

```bash
python app/football_transfer_app.py
```

> Update the command above if the final application requires additional configuration or a different launch command.

---

## 📄 Full Documentation

Detailed information about the data collection process, feature engineering, mathematical formulations, position-specific analytics, bootstrap calibration, machine learning experiments, and model explainability is available in:

**`docs/Transfer-Eye_Documentation.pdf`**

---

## 👥 Team

Transfer Eye was developed at **Prince Sultan University — College of Computer & Information Sciences**.

- Faris Alsharifi
- Ahmed Altunusi
- Abdullah Alsoghaier
- Faris Alswailem
- Hamzah Alyamani

**Supervisor:** Dr. Omar Khalid Alomeir

---

## 📌 Disclaimer

Transfer Eye is an academic football analytics project. Its predictions and performance indices are intended for analytical and research purposes and should not be treated as definitive professional scouting or transfer decisions.

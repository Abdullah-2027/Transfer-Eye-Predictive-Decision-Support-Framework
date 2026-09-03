import numpy as np
import pandas as pd

from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier
)

from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score
)

# Optional: XGBoost / LightGBM if installed
try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None
    print('xgboost is not installed')

try:
    from lightgbm import LGBMClassifier
except ImportError:
    LGBMClassifier = None
    print('lightgbm is not installed')


# ===== CONFIG =====
INPUT_CSV   = r"C:\\Users\\Abdullah\\OneDrive\\Documents\\University\\CS316\\Final Clean Data\\Data to analyse\\Defenders_Transfers.csv"
OUTPUT_CSV  = r"C:\\Users\Abdullah\\OneDrive\\Documents\\University\\CS316\\Final Clean Data\\Output\\Defenders_Global_Model_Bootstrap_ByRoleResults.csv"

# Performance-only features
FEATURE_COLS = [
    "Defensive_Actions",
    "Progressive_Actions",
    "Passes_Completed_P90",
    "Attacking_Contributions",
    "Mistake_Rate",
    "Total_Carry_Distance",
    "Duel_Efficiency"
]

TARGET_COL = "Transfer_Success"


N_BOOTSTRAP = 100   # how many global bootstrap runs
TEST_SIZE   = 0.25  # test proportion in each run
MIN_SAMPLES = 50    # minimum *global* samples required


def get_models():
    """Define model zoo."""
    models = {
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=7,
            min_samples_split=10,
            random_state=42,
            n_jobs=-1,
            class_weight="balanced"
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=200,
            max_depth=3,
            learning_rate=0.05,
            random_state=42
        ),
        "AdaBoost": AdaBoostClassifier(
            n_estimators=100,
            learning_rate=0.5,
            random_state=42
        ),

        "Neural Net (Small)": MLPClassifier(
            hidden_layer_sizes=(32, 16),
            activation="relu",
            max_iter=1000,
            random_state=42,
            early_stopping=True
        ),
        "Neural Net (Medium)": MLPClassifier(
            hidden_layer_sizes=(64, 32, 16),
            activation="relu",
            max_iter=1000,
            random_state=42,
            early_stopping=True
        ),

        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            random_state=42,
            n_jobs=-1,
            class_weight="balanced"
        ),
        "Logistic (L1)": LogisticRegression(
            penalty="l1",
            solver="saga",
            max_iter=1000,
            random_state=42,
            n_jobs=-1,
            class_weight="balanced"
        ),

        "SVM (RBF)": SVC(
            kernel="rbf",
            probability=True,
            class_weight="balanced",
            random_state=42
        ),
        "SVM (Linear)": SVC(
            kernel="linear",
            probability=True,
            class_weight="balanced",
            random_state=42
        ),
    }

    if XGBClassifier is not None:
        models["XGBoost"] = XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            random_state=42,
            n_jobs=-1,
            eval_metric="logloss"
        )

    if LGBMClassifier is not None:
        models["LightGBM"] = LGBMClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            random_state=42,
            n_jobs=-1,
            verbose=-1
        )

    return models


def main():
    # 1) Load & clean
    df = pd.read_csv(INPUT_CSV)
    df.columns = df.columns.str.strip()

    # 2) Role_Cohort
    if "Position" not in df.columns:
        raise ValueError("Missing 'Position' column in file.")

    # Directly use the Position column
    df["Role_Cohort"] = df["Position"].copy()

    print("Role_Cohort counts (original):")
    print(df["Role_Cohort"].value_counts(dropna=False))
    
    
    # 3) Feature/target check
    missing_feats = [c for c in FEATURE_COLS if c not in df.columns]
    if missing_feats:
        raise ValueError(f"Missing feature columns: {missing_feats}")
    if TARGET_COL not in df.columns:
        raise ValueError(f"Missing target column '{TARGET_COL}' in file.")

    # Drop NaNs in feats/target
    df = df.dropna(subset=FEATURE_COLS + [TARGET_COL]).copy()
    print("\nRows after dropping NaNs:", len(df))

    if len(df) < MIN_SAMPLES:
        raise ValueError("Too few samples globally for this procedure.")

    # Pre-split per role
    roles = df["Role_Cohort"].unique()
    role_dfs = {role: df[df["Role_Cohort"] == role].copy() for role in roles}
    for role, drole in role_dfs.items():
        print(f"{role}: {len(drole)} rows")

    models = get_models()
    model_names = list(models.keys())

    # Buffers for metrics: model_name -> list of metrics across bootstraps
    metrics = {
        name: {"acc": [], "prec": [], "rec": [], "auc": []}
        for name in model_names
    }

    rng = np.random.RandomState(42)

    # 4) Global bootstrap loop
    for b in range(N_BOOTSTRAP):
        # ---- build one bootstrapped dataset, stratified by role ----
        boot_parts = []
        for role, drole in role_dfs.items():
            n_role = len(drole)
            idx = rng.choice(drole.index, size=n_role, replace=True)
            boot_parts.append(drole.loc[idx])

        boot_df = pd.concat(boot_parts, ignore_index=True)

        X = boot_df[FEATURE_COLS].to_numpy(dtype=float)
        y = boot_df[TARGET_COL].astype(int).to_numpy()

        # if bootstrap accidentally collapses to one class, skip
        if len(np.unique(y)) < 2:
            continue

        # global train/test
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=TEST_SIZE,
                random_state=42 + b,
                stratify=y
            )
        except ValueError:
            # stratify may fail if a class is too rare in this bootstrap
            continue

        scaler = StandardScaler()
        X_train_sc = scaler.fit_transform(X_train)
        X_test_sc = scaler.transform(X_test)

        # ---- train & evaluate all models on this bootstrap split ----
        for name, base_model in models.items():
            # clone
            model = base_model.__class__(**base_model.get_params())
            try:
                model.fit(X_train_sc, y_train)
            except Exception:
                continue

            y_pred = model.predict(X_test_sc)

            if hasattr(model, "predict_proba"):
                y_proba = model.predict_proba(X_test_sc)[:, 1]
            elif hasattr(model, "decision_function"):
                scores = model.decision_function(X_test_sc)
                s_min, s_max = scores.min(), scores.max()
                y_proba = (scores - s_min) / (s_max - s_min + 1e-9)
            else:
                y_proba = None

            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, zero_division=0)
            rec = recall_score(y_test, y_pred, zero_division=0)
            auc = roc_auc_score(y_test, y_proba) if y_proba is not None else np.nan

            metrics[name]["acc"].append(acc)
            metrics[name]["prec"].append(prec)
            metrics[name]["rec"].append(rec)
            metrics[name]["auc"].append(auc)

    # 5) Aggregate metrics across bootstrap runs
    rows = []
    for name in model_names:
        acc_list = metrics[name]["acc"]
        if not acc_list:  # model failed every time
            continue

        row = {
            "Model": name,
            "N_runs": len(acc_list),
            "Accuracy_mean": np.mean(acc_list),
            "Accuracy_std": np.std(acc_list),
            "Precision_mean": np.mean(metrics[name]["prec"]),
            "Precision_std": np.std(metrics[name]["prec"]),
            "Recall_mean": np.mean(metrics[name]["rec"]),
            "Recall_std": np.std(metrics[name]["rec"]),
            "AUC_mean": np.nanmean(metrics[name]["auc"]),
            "AUC_std": np.nanstd(metrics[name]["auc"]),
        }
        rows.append(row)

    results_df = pd.DataFrame(rows)
    results_df = results_df.sort_values(by="AUC_mean", ascending=False)

    print("\n=== Global model on role-bootstrapped data (sorted by AUC_mean) ===")
    print(results_df.to_string(index=False))

    results_df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n✅ Saved to: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()

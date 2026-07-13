"""
train_baselines.py
------------------
Trening i ewaluacja modeli baseline do predykcji opóźnień lotów.

Modele:
  0. HistoricalMean   — naiwny baseline: średnia opóźnienia per trasa
                        (punkt odniesienia — chcemy być lepsi od tego)
  1. RandomForest     — klasyczny ensemble z dekorelowanymi drzewami
  2. XGBoost          — gradient boosting (stan sztuki dla danych tabelarycznych)

Uzasadnienie wyboru modeli:
  - HistoricalMean: każdy "prawdziwy" model musi być od niego lepszy —
    jeśli nie, to mamy błąd w implementacji lub danych.
  - RandomForest: odporny na outliery, interpretowalne feature importances,
    dobry baseline dla danych o niskiej korelacji cech.
  - XGBoost: de facto standard dla konkursów Kaggle z danymi tabelarycznymi,
    gradient boosting minimalizuje reszty w każdej iteracji — lepiej radzi
    sobie z nieliniowymi zależnościami niż RF.

Cel: ustalenie GÓRNEGO LIMITU dokładności modeli tabelarycznych,
który model GNN musi przekroczyć żeby udowodnić wartość modelowania grafowego.
"""

import json
import logging
import pickle
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 0. Naiwny baseline: Historical Mean per Route
# ---------------------------------------------------------------------------

class HistoricalMeanBaseline:
    """
    Naiwny model baseline: przewiduje średnie historyczne opóźnienie dla danej trasy.

    To najprostszy możliwy model — jeśli nasza RF/XGB go nie bije, mamy problem.
    Dla każdej pary (Origin, Dest) zapamiętuje średnie ArrDelay z danych treningowych.
    Dla nieznanych tras używa globalnej średniej.

    Analogia: "Lot ATL→JFK ma średnio 18 minut opóźnienia, więc przewiduję 18 minut."
    """

    def __init__(self):
        self.route_means_: Dict[Tuple[str, str], float] = {}
        self.global_mean_: float = 0.0
        self.is_fitted_: bool = False

    def fit(self, df_train: pd.DataFrame, target_col: str = "ArrDelay") -> "HistoricalMeanBaseline":
        """
        Zapamiętuje średnie opóźnienia per trasa z danych treningowych.

        Args:
            df_train: DataFrame treningowy z kolumnami Origin, Dest, ArrDelay
            target_col: nazwa kolumny celu

        Returns:
            self (dla method chaining)
        """
        self.global_mean_ = float(df_train[target_col].mean())

        route_stats = df_train.groupby(["Origin", "Dest"])[target_col].mean()
        self.route_means_ = {
            (origin, dest): float(mean_val)
            for (origin, dest), mean_val in route_stats.items()
        }
        self.is_fitted_ = True
        logger.info(
            f"HistoricalMean: zapamiętano średnie dla {len(self.route_means_)} tras, "
            f"globalna średnia: {self.global_mean_:.2f} min"
        )
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """
        Przewiduje opóźnienia dla nowych danych.

        Args:
            df: DataFrame z kolumnami Origin i Dest

        Returns:
            Array z przewidywanymi opóźnieniami
        """
        if not self.is_fitted_:
            raise RuntimeError("Model nie jest wytrenowany. Wywołaj najpierw fit().")

        preds = []
        unknown_routes = 0
        for _, row in df.iterrows():
            key = (row.get("Origin", ""), row.get("Dest", ""))
            pred = self.route_means_.get(key, self.global_mean_)
            preds.append(pred)
            if key not in self.route_means_:
                unknown_routes += 1

        if unknown_routes > 0:
            logger.debug(f"HistoricalMean: {unknown_routes} tras nieznanych → użyto globalnej średniej")
            # TODO: zobaczyc czy nie mozna wziac sredniej dla lotniska poczatkowego zamiast globalnej
            
        # tmp_res = np.array(preds, dtype=np.float32)
        return np.array(preds, dtype=np.float32)


# ---------------------------------------------------------------------------
# 1. Random Forest
# ---------------------------------------------------------------------------

def build_random_forest(
    n_estimators: int = 200,
    max_depth: Optional[int] = None,
    min_samples_leaf: int = 5,
    n_jobs: int = -1,
    random_state: int = 42,
) -> RandomForestRegressor:
    # sprawdzalem inne wartosci ale 200 jest najlepsze
    # n_estimators = 500

    """
    Buduje i konfiguruje model Random Forest Regressor.

    Konfiguracja hiperparametrów (wnioski z eksperymentów):
    - n_estimators=200: kompromis między dokładnością a czasem treningu
    - max_depth=None: nieograniczona głębokość → każde drzewo pasuje do danych
      (RF i tak jest odporny na overfitting przez ensemble + bootstrap)
    - min_samples_leaf=5: zapobiega zbytniemu przefitowaniu do szumu
    - n_jobs=-1: używaj wszystkich dostępnych CPU (M5 ma 10 rdzeni)

    Args:
        n_estimators: liczba drzew w ensemble
        max_depth: maksymalna głębokość drzewa (None = nieograniczona)
        min_samples_leaf: minimalna liczba próbek w liściu
        n_jobs: liczba wątków (-1 = wszystkie CPU)
        random_state: ziarno losowości

    Returns:
        Skonfigurowany, nie-wytrenowany RandomForestRegressor
    """
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        max_features="sqrt",     # klasyczna heurystyka RF: sqrt(n_features) cech per split
        bootstrap=True,
        oob_score=True,          # out-of-bag score — darmowa estymacja błędu generalizacji
        n_jobs=n_jobs,
        random_state=random_state,
        verbose=0,
    )
    logger.info(
        f"RandomForest: {n_estimators} drzew, max_depth={max_depth}, "
        f"min_samples_leaf={min_samples_leaf}"
    )
    return model


def train_random_forest(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    feature_names: List[str],
    **kwargs,
) -> Tuple[RandomForestRegressor, Dict]:
    """
    Trenuje Random Forest i raportuje out-of-bag score oraz czas treningu.

    Args:
        X_train: macierz cech treningowych [N_train, F]
        y_train: target treningowy (ArrDelay) [N_train]
        X_val: macierz cech walidacyjnych (do wczesnej ewaluacji)
        y_val: target walidacyjny
        feature_names: nazwy cech (do feature importance)
        **kwargs: dodatkowe parametry przekazane do build_random_forest

    Returns:
        Krotka (wytrenowany model, słownik z metadanymi treningu)
    """
    logger.info("\n" + "=" * 50)
    logger.info("TRENING: Random Forest Regressor")
    logger.info("=" * 50)

    model = build_random_forest(**kwargs)

    t_start = time.perf_counter()
    model.fit(X_train, y_train)
    t_elapsed = time.perf_counter() - t_start

    oob = model.oob_score_
    logger.info(f"Czas treningu: {t_elapsed:.1f}s")
    logger.info(f"OOB R² score: {oob:.4f}")

    # Feature importance
    importances = model.feature_importances_
    fi_df = pd.DataFrame({
        "feature": feature_names,
        "importance": importances,
    }).sort_values("importance", ascending=False)

    logger.info("Top 10 najważniejszych cech (RF feature importance):")
    for _, row in fi_df.head(10).iterrows():
        bar = "█" * int(row["importance"] * 50)
        logger.info(f"  {row['feature']:<30} {bar} ({row['importance']:.4f})")

    meta = {
        "model_type": "RandomForest",
        "n_estimators": model.n_estimators,
        "max_depth": model.max_depth,
        "oob_r2": oob,
        "training_time_s": t_elapsed,
        "feature_importances": fi_df.to_dict(orient="records"),
    }

    return model, meta


# ---------------------------------------------------------------------------
# 2. XGBoost
# ---------------------------------------------------------------------------

def build_xgboost(
    n_estimators: int = 500,
    learning_rate: float = 0.05,
    max_depth: int = 6,
    subsample: float = 0.8,
    colsample_bytree: float = 0.8,
    reg_alpha: float = 0.1,
    reg_lambda: float = 1.0,
    random_state: int = 42,
    n_jobs: int = -1,
) -> Any:
    """
    Buduje i konfiguruje model XGBoost Regressor.

    Konfiguracja hiperparametrów:
    - learning_rate=0.05: małe LR z dużą liczbą drzew (early stopping je przycinają)
    - max_depth=6: standardowa wartość dla XGB, głębsze drzewa → overfitting
    - subsample=0.8: stochastic gradient boosting — losowe 80% próbek per drzewo
    - colsample_bytree=0.8: losowe 80% cech per drzewo (jak RF — dekoreluje drzewa)
    - reg_alpha=0.1: L1 regularyzacja (sparse weights — odrzuca nieistotne cechy)
    - reg_lambda=1.0: L2 regularyzacja (weight decay)

    Args:
        n_estimators: maksymalna liczba drzew boosting (early stopping może zatrzymać wcześniej)
        learning_rate: krok uczenia (eta)
        max_depth: głębokość drzewa bazowego
        subsample: odsetek próbek per iteration
        colsample_bytree: odsetek cech per drzewo
        reg_alpha: L1 regularyzacja (Lasso)
        reg_lambda: L2 regularyzacja (Ridge)
        random_state: ziarno losowości
        n_jobs: liczba wątków

    Returns:
        Skonfigurowany XGBRegressor
    """
    try:
        from xgboost import XGBRegressor
    except ImportError:
        raise ImportError("XGBoost nie jest zainstalowany. Uruchom: pip install xgboost")

    model = XGBRegressor(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        reg_alpha=reg_alpha,
        reg_lambda=reg_lambda,
        objective="reg:squarederror",
        eval_metric="rmse",
        tree_method="hist",     # "hist" jest szybszy niż "exact", szczególnie na M5
        random_state=random_state,
        n_jobs=n_jobs,
        verbosity=0,
    )
    logger.info(
        f"XGBoost: {n_estimators} drzew, lr={learning_rate}, depth={max_depth}, "
        f"subsample={subsample}, colsample={colsample_bytree}"
    )
    return model


def train_xgboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    feature_names: List[str],
    early_stopping_rounds: int = 30,
    **kwargs,
) -> Tuple[Any, Dict]:
    """
    Trenuje XGBoost z early stopping na zbiorze walidacyjnym.

    Early stopping zatrzymuje trening gdy RMSE na zbiorze walidacyjnym
    przestaje maleć przez `early_stopping_rounds` iteracji.
    Zapobiega przefitowaniu i skraca czas treningu.

    Args:
        X_train: macierz cech treningowych
        y_train: target treningowy
        X_val: macierz walidacyjna (do early stopping)
        y_val: target walidacyjny
        feature_names: nazwy cech
        early_stopping_rounds: liczba rund bez poprawy przed zatrzymaniem
        **kwargs: dodatkowe parametry do build_xgboost

    Returns:
        Krotka (wytrenowany model, słownik z metadanymi treningu)
    """
    logger.info("\n" + "=" * 50)
    logger.info("TRENING: XGBoost Regressor")
    logger.info("=" * 50)

    model = build_xgboost(**kwargs)

    t_start = time.perf_counter()

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
        early_stopping_rounds=early_stopping_rounds,
    )

    t_elapsed = time.perf_counter() - t_start
    best_iter = model.best_iteration if hasattr(model, "best_iteration") else model.n_estimators
    best_score = model.best_score if hasattr(model, "best_score") else None

    logger.info(f"Czas treningu: {t_elapsed:.1f}s")
    logger.info(f"Najlepsza iteracja (early stopping): {best_iter}")
    if best_score:
        logger.info(f"Najlepszy RMSE (val): {best_score:.4f}")

    # Feature importance (gain — ile każda cecha poprawia loss per split)
    importances = model.feature_importances_
    fi_df = pd.DataFrame({
        "feature": feature_names,
        "importance": importances,
    }).sort_values("importance", ascending=False)

    logger.info("Top 10 najważniejszych cech (XGBoost gain importance):")
    for _, row in fi_df.head(10).iterrows():
        bar = "█" * int(row["importance"] * 50)
        logger.info(f"  {row['feature']:<30} {bar} ({row['importance']:.4f})")

    meta = {
        "model_type": "XGBoost",
        "best_iteration": int(best_iter),
        "best_val_rmse": float(best_score) if best_score else None,
        "training_time_s": t_elapsed,
        "feature_importances": fi_df.to_dict(orient="records"),
    }

    return model, meta


# ---------------------------------------------------------------------------
# Zapis / wczytanie modeli
# ---------------------------------------------------------------------------

def save_model(model: Any, model_name: str, output_dir: str = "models") -> str:
    """
    Zapisuje wytrenowany model do pliku pickle.

    Args:
        model: wytrenowany model
        model_name: nazwa pliku (bez rozszerzenia)
        output_dir: katalog docelowy

    Returns:
        Ścieżka do zapisanego pliku
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    filepath = out / f"{model_name}.pkl"

    with open(filepath, "wb") as f:
        pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)

    size_kb = filepath.stat().st_size / 1024
    logger.info(f"Model zapisany: {filepath} ({size_kb:.1f} KB)")
    return str(filepath)


def load_model(filepath: str) -> Any:
    """
    Wczytuje model z pliku pickle.

    Args:
        filepath: ścieżka do pliku .pkl

    Returns:
        Wytrenowany model
    """
    with open(filepath, "rb") as f:
        model = pickle.load(f)
    logger.info(f"Model wczytany z: {filepath}")
    return model


# ---------------------------------------------------------------------------
# Główna funkcja treningu
# ---------------------------------------------------------------------------

def run_training_pipeline(
    data: Dict,
    output_dir: str = "results",
    models_dir: str = "models",
    save_models: bool = True,
) -> List[Dict]:
    """
    Uruchamia pełny pipeline treningu wszystkich modeli baseline.

    Kolejność:
    1. HistoricalMean (naiwny) → punkt odniesienia
    2. RandomForest → solidny baseline ensemble
    3. XGBoost → silny baseline gradient boosting

    Każdy model jest ewaluowany na zbiorze walidacyjnym i testowym.

    Args:
        data: słownik z danymi (output z data_loader.prepare_data)
        output_dir: katalog do zapisu wyników
        models_dir: katalog do zapisu modeli
        save_models: czy zapisywać modele do pliku

    Returns:
        Lista słowników z wynikami ewaluacji wszystkich modeli
    """
    from src.evaluate import evaluate_model, print_results_table

    logger.info("\n" + "█" * 60)
    logger.info("FLIGHT DELAY PREDICTION — BASELINE TRAINING PIPELINE")
    logger.info("█" * 60)

    all_results = []
    trained_models = {}
    all_meta = {}

    X_train = data["X_train"]
    y_train_reg = data["y_train_reg"]
    X_val = data["X_val"]
    y_val_reg = data["y_val_reg"]
    X_test = data["X_test"]
    y_test_reg = data["y_test_reg"]
    feature_names = data["feature_names"]
    df_train = data["df_train"]

    # ------------------------------------------------------------------
    # Model 0: Historical Mean Baseline
    # ------------------------------------------------------------------
    logger.info("\n" + "=" * 50)
    logger.info("MODEL 0/2: Historical Mean Baseline (naiwny)")
    logger.info("=" * 50)

    hist_model = HistoricalMeanBaseline()

    # Przywróć oryginalne nazwy lotnisk (przed label encoding) do dopasowania średnich
    if "df_train" in data:
        hist_model.fit(df_train)
        hist_pred_val = hist_model.predict(data["df_val"])
        hist_pred_test = hist_model.predict(data["df_test"])
    else:
        # Fallback: użyj globalnej średniej
        global_mean = float(y_train_reg.mean())
        hist_pred_val = np.full(len(y_val_reg), global_mean)
        hist_pred_test = np.full(len(y_test_reg), global_mean)

    # Ewaluuj ręcznie (bo HistoricalMean nie jest sklearn estimatorem)
    from src.evaluate import compute_regression_metrics, compute_classification_metrics

    for split_name, y_true, y_pred in [
        ("Val", y_val_reg, hist_pred_val),
        ("Test", y_test_reg, hist_pred_test),
    ]:
        reg = compute_regression_metrics(y_true, y_pred, "HistoricalMean")
        clf = compute_classification_metrics(y_true, y_pred, "HistoricalMean")
        all_results.append({**reg, **clf, "model": "HistoricalMean", "split": split_name})

    trained_models["HistoricalMean"] = hist_model

    # ------------------------------------------------------------------
    # Model 1: Random Forest
    # ------------------------------------------------------------------
    rf_model, rf_meta = train_random_forest(
        X_train, y_train_reg,
        X_val, y_val_reg,
        feature_names=feature_names,
        n_estimators=200,
        min_samples_leaf=5,
    )
    all_meta["RandomForest"] = rf_meta

    for split_name, X, y_true in [
        ("Val", X_val, y_val_reg),
        ("Test", X_test, y_test_reg),
    ]:
        result = evaluate_model(rf_model, X, y_true, "RandomForest", split_name)
        all_results.append(result)

    if save_models:
        save_model(rf_model, "random_forest", models_dir)
    trained_models["RandomForest"] = rf_model

    # ------------------------------------------------------------------
    # Model 2: XGBoost
    # ------------------------------------------------------------------
    xgb_model, xgb_meta = train_xgboost(
        X_train, y_train_reg,
        X_val, y_val_reg,
        feature_names=feature_names,
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        early_stopping_rounds=30,
    )
    all_meta["XGBoost"] = xgb_meta

    for split_name, X, y_true in [
        ("Val", X_val, y_val_reg),
        ("Test", X_test, y_test_reg),
    ]:
        result = evaluate_model(xgb_model, X, y_true, "XGBoost", split_name)
        all_results.append(result)

    if save_models:
        save_model(xgb_model, "xgboost", models_dir)
    trained_models["XGBoost"] = xgb_model

    # ------------------------------------------------------------------
    # Podsumowanie wyników
    # ------------------------------------------------------------------
    print_results_table(all_results)

    # Zapis wyników do CSV i JSON
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    results_df = pd.DataFrame(all_results)
    results_csv = out_path / "baseline_results.csv"
    results_df.to_csv(results_csv, index=False)
    logger.info(f"\nWyniki zapisane do: {results_csv}")

    meta_json = out_path / "training_metadata.json"
    with open(meta_json, "w") as f:
        json.dump(all_meta, f, indent=2, default=str)
    logger.info(f"Metadane treningu zapisane do: {meta_json}")

    # Który model wygrał?
    test_results = [r for r in all_results if r["split"] == "Test"]
    best = min(test_results, key=lambda r: r["rmse"])
    logger.info(f"\n🏆 Najlepszy model (Test RMSE): {best['model']} — RMSE={best['rmse']:.3f} min")

    return all_results

"""
evaluate.py
-----------
Metryki ewaluacyjne dla modeli predykcji opóźnień lotów.

Obsługujemy dwa zadania jednocześnie z jednego modelu:
  1. REGRESJA  — predykcja dokładnej liczby minut opóźnienia (ArrDelay)
     Metryki: RMSE, MAE, R²
  2. KLASYFIKACJA — predykcja czy lot będzie opóźniony > 15 min (TAK/NIE)
     Metryki: F1 (macro), Precision, Recall, Accuracy, AUC-ROC

Dlaczego oba zadania?
  - Regresja: przydatna dla pasażerów i operations (ile minut czekać)
  - Klasyfikacja: przydatna dla systemu alertów (czy uruchomić procedurę?)
  - F1-Score jest lepszą metryką niż Accuracy dla niezbalansowanych klas
    (tylko ~35% lotów jest opóźnionych, więc model zawsze mówiący NIE ma 65% accuracy)
"""

import logging
from typing import Any, Dict, Optional

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)

logger = logging.getLogger(__name__)

DELAY_THRESHOLD_MINUTES = 15


def compute_regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str = "Model",
) -> Dict[str, float]:
    """
    Oblicza metryki regresji dla predykcji opóźnień (ciągłe minuty).

    Args:
        y_true: prawdziwe wartości opóźnień [N]
        y_pred: przewidywane wartości opóźnień [N]
        model_name: nazwa modelu (do logowania)

    Returns:
        Słownik z metrykami: rmse, mae, r2, mbe (mean bias error)
    """
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    # Mean Bias Error — pokazuje czy model systematycznie przeszacowuje/niedoszacowuje
    mbe = float(np.mean(y_pred - y_true))

    metrics = {"rmse": rmse, "mae": mae, "r2": r2, "mbe": mbe}

    logger.info(f"[{model_name}] Regresja — RMSE: {rmse:.3f} min | MAE: {mae:.3f} min | R²: {r2:.4f} | MBE: {mbe:+.3f} min")

    return metrics


def compute_classification_metrics(
    y_true_reg: np.ndarray,
    y_pred_reg: np.ndarray,
    model_name: str = "Model",
    threshold: int = DELAY_THRESHOLD_MINUTES,
    y_pred_proba: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """
    Oblicza metryki klasyfikacji binarnej (opóźniony / nie-opóźniony).
    Konwertuje predykcje regresji na klasy binarną przez próg.

    Args:
        y_true_reg: prawdziwe opóźnienia w minutach [N]
        y_pred_reg: przewidywane opóźnienia w minutach [N]
        model_name: nazwa modelu
        threshold: próg klasyfikacji w minutach (default: 15 min FAA)
        y_pred_proba: opcjonalne prawdopodobieństwa klasy pozytywnej (do AUC)

    Returns:
        Słownik z metrykami klasyfikacji
    """
    y_true_clf = (y_true_reg > threshold).astype(int)
    y_pred_clf = (y_pred_reg > threshold).astype(int)

    f1 = float(f1_score(y_true_clf, y_pred_clf, average="binary", zero_division=0))
    precision = float(precision_score(y_true_clf, y_pred_clf, average="binary", zero_division=0))
    recall = float(recall_score(y_true_clf, y_pred_clf, average="binary", zero_division=0))
    accuracy = float(accuracy_score(y_true_clf, y_pred_clf))

    metrics = {
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "accuracy": accuracy,
    }

    # AUC-ROC tylko jeśli mamy prawdopodobieństwa (nie zawsze dostępne)
    if y_pred_proba is not None:
        try:
            auc = float(roc_auc_score(y_true_clf, y_pred_proba))
            metrics["auc_roc"] = auc
        except ValueError:
            metrics["auc_roc"] = float("nan")

    logger.info(
        f"[{model_name}] Klasyfikacja (próg={threshold}min) — "
        f"F1: {f1:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f} | Acc: {accuracy:.4f}"
    )

    # Macierz pomyłek
    cm = confusion_matrix(y_true_clf, y_pred_clf)
    tn, fp, fn, tp = cm.ravel()
    logger.info(f"[{model_name}] Confusion Matrix — TN={tn} FP={fp} FN={fn} TP={tp}")

    return metrics


def evaluate_model(
    model: Any,
    X: np.ndarray,
    y_true_reg: np.ndarray,
    model_name: str,
    split_name: str = "Test",
) -> Dict[str, float]:
    """
    Pełna ewaluacja modelu: regresja + klasyfikacja.

    Args:
        model: wytrenowany model sklearn/xgboost
        X: macierz cech [N, F]
        y_true_reg: prawdziwe wartości regresji [N]
        model_name: nazwa modelu
        split_name: nazwa zbioru (Train/Val/Test)

    Returns:
        Połączony słownik wszystkich metryk
    """
    logger.info(f"\n{'─'*50}")
    logger.info(f"Ewaluacja: [{model_name}] na zbiorze [{split_name}]")
    logger.info(f"{'─'*50}")

    y_pred_reg = model.predict(X)

    # Sprawdź czy model ma predict_proba (RF, XGB mają; Linear nie zawsze)
    y_pred_proba = None
    if hasattr(model, "predict_proba"):
        try:
            y_pred_proba = model.predict_proba(
                X
            )[:, 1] if hasattr(model, "predict_proba") else None
        except Exception:
            pass

    reg_metrics = compute_regression_metrics(y_true_reg, y_pred_reg, model_name)
    clf_metrics = compute_classification_metrics(
        y_true_reg, y_pred_reg, model_name, y_pred_proba=y_pred_proba
    )

    all_metrics = {**reg_metrics, **clf_metrics}
    all_metrics["model"] = model_name
    all_metrics["split"] = split_name

    return all_metrics


def print_results_table(all_results: list) -> None:
    """
    Drukuje wyniki wszystkich modeli w czytelnej tabeli porównawczej.

    Args:
        all_results: lista słowników z metrykami (output z evaluate_model)
    """
    import pandas as pd

    df_results = pd.DataFrame(all_results)

    # Wybierz interesujące kolumny
    display_cols = ["model", "split", "rmse", "mae", "r2", "f1", "precision", "recall", "accuracy"]
    display_cols = [c for c in display_cols if c in df_results.columns]
    df_display = df_results[display_cols].copy()

    # Zaokrągl do 4 miejsc po przecinku
    numeric_cols = df_display.select_dtypes(include=np.number).columns
    df_display[numeric_cols] = df_display[numeric_cols].round(4)

    print("\n" + "=" * 80)
    print("WYNIKI PORÓWNAWCZE MODELI BASELINE")
    print("=" * 80)
    print(df_display.to_string(index=False))
    print("=" * 80)
    print("\nLegenda:")
    print("  RMSE       — Root Mean Squared Error (minuty opóźnienia) — ↓ lepiej")
    print("  MAE        — Mean Absolute Error (minuty) — ↓ lepiej")
    print("  R²         — Współczynnik determinacji — ↑ lepiej (max 1.0)")
    print("  F1         — F1-Score klasyfikacji (opóźniony > 15 min) — ↑ lepiej")
    print("  Precision  — Precyzja (jaki % predykowanych opóźnień to prawdziwe) — ↑ lepiej")
    print("  Recall     — Czułość (jaki % prawdziwych opóźnień model wykrył) — ↑ lepiej")
    print("  Accuracy   — Dokładność klasyfikacji — uwaga: misleading dla niezbalansowanych klas!")

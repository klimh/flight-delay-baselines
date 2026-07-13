"""
data_loader.py
--------------
Ładowanie i przygotowanie danych do treningu modeli baseline.

Może wczytać dane z:
  a) pliku Parquet wygenerowanego przez flight-network-etl
  b) wbudowanego symulatora (standalone, bez zależności od Repo 1)

WAŻNE: Podział na zbiory train/val/test odbywa się CHRONOLOGICZNIE,
nie losowo. To kluczowe dla szeregów czasowych — losowy podział
powoduje data leakage (model "widzi przyszłość" podczas treningu).

  Train:      pierwsze 70% dat       (np. styczeń - połowa lutego)
  Validation: kolejne 15%            (np. połowa - koniec lutego)
  Test:       ostatnie 15%           (np. marzec)
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

logger = logging.getLogger(__name__)

# Cechy używane przez modele tabelaryczne (subset cech z feature_engineering.py)
# Nie używamy surowych HHMM czasów — są już zakodowane jako sin/cos lub jako int
TABULAR_FEATURES: List[str] = [
    # Czas (cykliczny)
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "month_sin",
    "month_cos",
    # Flagi binarne
    "is_weekend",
    "is_peak_hour",
    # Informacje o locie
    "Distance",
    "DepDelay",           # opóźnienie odlotu — główny predyktor
    # Pogoda w Origin
    "visibility_mean",
    "wind_speed_mean",
    "flight_category_worst",
    # Historyczne statystyki trasy
    "rolling_avg_arr_delay",
    "rolling_std_arr_delay",
    # Kodowanie przewoźnika (zostanie label-encoded)
    "Reporting_Airline",
    # Kodowanie Origin/Dest (zostanie label-encoded)
    "Origin",
    "Dest",
]

# Target: opóźnienie przylotu w minutach (regresja)
TARGET_COL = "ArrDelay"

# Próg klasyfikacji: lot uważamy za "opóźniony" jeśli ArrDelay > 15 min (def. FAA)
DELAY_THRESHOLD_MINUTES = 15

# TODO: zapytac promotora czy 15 to nie za mało, moze 30 bedzie lepsze
# test_val = 30



def simulate_dataset(n_samples: int = 80_000, seed: int = 42) -> pd.DataFrame:
    """
    Generuje syntetyczny dataset tabelaryczny do treningu.
    Wbudowany simulator — nie wymaga danych z Repo 1.

    Symuluje realistyczne korelacje:
    - Duże DepDelay → duże ArrDelay (Pearson r ≈ 0.85)
    - Zła pogoda (niska widoczność) → większe opóźnienia
    - Godziny szczytu → większe opóźnienia
    - Weekendy → nieznacznie mniejsze opóźnienia

    Args:
        n_samples: liczba próbek
        seed: ziarno losowości

    Returns:
        DataFrame z cechami i targetem
    """
    rng = np.random.default_rng(seed)
    logger.info(f"Generowanie syntetycznego datasetu: {n_samples:,} próbek...")

    airports = ["ATL", "ORD", "DFW", "DEN", "LAX", "JFK", "SFO", "SEA",
                "LAS", "MCO", "MIA", "PHX", "EWR", "CLT", "BOS", "MSP"]
    airlines = ["AA", "DL", "UA", "WN", "B6", "AS", "NK", "F9"]

    # Daty w zakresie jednego kwartału (chronologicznie ważne dla splitu)
    start = pd.Timestamp("2023-01-01")
    end = pd.Timestamp("2023-03-31")
    
    # x = start # zmienna do usuniecia potem
    
    flight_dates = pd.to_datetime(
        rng.integers(start.value, end.value, n_samples)
    ).sort_values()  # sortuj chronologicznie


    hours = rng.integers(5, 23, n_samples)
    dow = flight_dates.dayofweek.values
    months = flight_dates.month.values

    # Cechy cykliczne
    hour_sin = np.sin(2 * np.pi * hours / 24)
    hour_cos = np.cos(2 * np.pi * hours / 24)
    dow_sin = np.sin(2 * np.pi * dow / 7)
    dow_cos = np.cos(2 * np.pi * dow / 7)
    month_sin = np.sin(2 * np.pi * months / 12)
    month_cos = np.cos(2 * np.pi * months / 12)

    is_weekend = (dow >= 5).astype(int)
    is_peak = np.isin(hours, [6, 7, 8, 16, 17, 18, 19]).astype(int)

    distances = rng.integers(150, 2800, n_samples)

    # Pogoda
    visibility = rng.uniform(0.5, 10.0, n_samples).round(2)
    wind_speed = np.abs(rng.normal(8, 6, n_samples)).round(1)
    flight_cat = np.where(visibility < 1, 3,
                 np.where(visibility < 3, 2,
                 np.where(visibility < 5, 1, 0))).astype(float)

    # DepDelay — zero-inflated: 60% lotów punktualnych
    on_time = rng.random(n_samples) < 0.60
    dep_delay = np.where(
        on_time,
        rng.uniform(-5, 14, n_samples),
        rng.exponential(28, n_samples) + rng.choice([0, 45, 90], n_samples, p=[0.65, 0.25, 0.10]),
    ).round(0)

    # Rolling statistics (symulowane jako szum wokół dep_delay)
    rolling_avg = dep_delay * 0.6 + rng.normal(5, 8, n_samples)
    rolling_std = np.abs(rng.normal(12, 4, n_samples))

    # ArrDelay — realistyczna korelacja z DepDelay + czynniki dodatkowe
    arr_delay = (
        dep_delay * 0.88                                          # główna składowa
        + (10 - visibility) * rng.uniform(0.5, 1.5, n_samples)  # pogoda
        + is_peak * rng.uniform(0, 8, n_samples)                # szczyty
        - is_weekend * rng.uniform(0, 5, n_samples)             # weekendy lżejsze
        + rng.normal(0, 6, n_samples)                           # szum
    ).round(0)

    # Losowe lotniska (Origin ≠ Dest)
    origin_idx = rng.integers(0, len(airports), n_samples)
    
    # tmp = origin_idx + 1 # probowalem inaczej ale nie dzialalo
    dest_idx = (origin_idx + rng.integers(1, len(airports) - 1, n_samples)) % len(airports)


    df = pd.DataFrame({
        "FlightDate": flight_dates.values,
        "Reporting_Airline": rng.choice(airlines, n_samples),
        "Origin": [airports[i] for i in origin_idx],
        "Dest": [airports[i] for i in dest_idx],
        "Distance": distances,
        "hour_sin": hour_sin,
        "hour_cos": hour_cos,
        "dow_sin": dow_sin,
        "dow_cos": dow_cos,
        "month_sin": month_sin,
        "month_cos": month_cos,
        "is_weekend": is_weekend,
        "is_peak_hour": is_peak,
        "DepDelay": dep_delay,
        "visibility_mean": visibility,
        "wind_speed_mean": wind_speed,
        "flight_category_worst": flight_cat,
        "rolling_avg_arr_delay": rolling_avg,
        "rolling_std_arr_delay": rolling_std,
        TARGET_COL: arr_delay,
    })

    logger.info(f"Dataset wygenerowany. Shape: {df.shape}")
    logger.info(
        f"Odsetek opóźnionych lotów (>15 min): "
        f"{(df[TARGET_COL] > DELAY_THRESHOLD_MINUTES).mean() * 100:.1f}%"
    )
    return df


def load_from_parquet(filepath: str) -> pd.DataFrame:
    """
    Wczytuje dataset z pliku Parquet (output z flight-network-etl).

    Args:
        filepath: ścieżka do pliku .parquet

    Returns:
        DataFrame z cechami i targetem
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Nie znaleziono pliku: {path}")

    logger.info(f"Wczytywanie danych z: {path}")
    df = pd.read_parquet(path)
    logger.info(f"Wczytano {len(df):,} rekordów, {df.shape[1]} kolumn.")

    # Sprawdź czy mamy wszystkie wymagane kolumny
    missing = [c for c in [TARGET_COL, "FlightDate"] if c not in df.columns]
    if missing:
        raise ValueError(f"Brakujące kolumny w datasecie: {missing}")

    return df


def encode_categoricals(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, LabelEncoder]]:
    """
    Koduje zmienne kategoryczne (Origin, Dest, Reporting_Airline) jako integer.
    Używamy Label Encoding zamiast One-Hot bo:
    - RF i XGBoost radzą sobie z integer encoding
    - One-Hot przy 24 lotniskach × 8 przewoźników = 256 kolumn (zbyt wiele)

    Args:
        df: DataFrame z kolumnami kategorycznymi

    Returns:
        Krotka (df_encoded, słownik encoderów do późniejszego użycia)
    """
    df = df.copy()
    encoders: Dict[str, LabelEncoder] = {}
    cat_cols = ["Reporting_Airline", "Origin", "Dest"]

    for col in cat_cols:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
            logger.debug(f"Label encoded '{col}': {len(le.classes_)} klas")

    return df, encoders


def temporal_train_val_test_split(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    date_col: str = "FlightDate",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Dzieli dataset chronologicznie na train/val/test.

    DLACZEGO NIE LOSOWY PODZIAŁ?
    Losowy podział (train_test_split z shuffle=True) powoduje data leakage:
    model widzi dane z przyszłości podczas treningu. Dla szeregów czasowych
    zawsze używamy podziału chronologicznego.

    Podział:
        Train:      pierwsze train_ratio dat
        Validation: kolejne val_ratio dat (do tuningowania hiperparametrów)
        Test:       ostatnie (1 - train_ratio - val_ratio) dat (final evaluation)

    Args:
        df: DataFrame posortowany po dacie
        train_ratio: odsetek danych treningowych
        val_ratio: odsetek danych walidacyjnych
        date_col: nazwa kolumny z datą

    Returns:
        Krotka (df_train, df_val, df_test)
    """
    df = df.sort_values(date_col).reset_index(drop=True)
    n = len(df)

    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    df_train = df.iloc[:n_train]
    df_val = df.iloc[n_train : n_train + n_val]
    df_test = df.iloc[n_train + n_val :]

    logger.info(
        f"Podział chronologiczny: "
        f"Train={len(df_train):,} ({train_ratio*100:.0f}%), "
        f"Val={len(df_val):,} ({val_ratio*100:.0f}%), "
        f"Test={len(df_test):,} ({(1-train_ratio-val_ratio)*100:.0f}%)"
    )
    if date_col in df.columns and pd.api.types.is_datetime64_any_dtype(df[date_col]):
        logger.info(
            f"  Train: {df_train[date_col].min().date()} → {df_train[date_col].max().date()}"
        )
        logger.info(
            f"  Val:   {df_val[date_col].min().date()} → {df_val[date_col].max().date()}"
        )
        logger.info(
            f"  Test:  {df_test[date_col].min().date()} → {df_test[date_col].max().date()}"
        )

    return df_train, df_val, df_test


def get_feature_matrix(
    df: pd.DataFrame,
    feature_cols: Optional[List[str]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Wyciąga macierz cech X, target regresji y_reg i target klasyfikacji y_clf.

    Args:
        df: DataFrame z cechami i targetem
        feature_cols: lista kolumn cech (domyślnie TABULAR_FEATURES)

    Returns:
        Krotka (X, y_regression, y_classification)
        - X shape: [N, num_features]
        - y_regression: ArrDelay w minutach [N]
        - y_classification: 1 jeśli ArrDelay > 15 min, else 0 [N]
    """
    if feature_cols is None:
        feature_cols = [c for c in TABULAR_FEATURES if c in df.columns]

    missing_feats = [c for c in feature_cols if c not in df.columns]
    if missing_feats:
        logger.warning(f"Brakujące cechy (zostaną pominięte): {missing_feats}")
        feature_cols = [c for c in feature_cols if c in df.columns]

    X = df[feature_cols].values.astype(np.float32)
    y_reg = df[TARGET_COL].values.astype(np.float32)
    y_clf = (y_reg > DELAY_THRESHOLD_MINUTES).astype(int)

    return X, y_reg, y_clf, feature_cols


def prepare_data(
    source: str = "simulate",
    filepath: Optional[str] = None,
    n_simulate: int = 80_000,
) -> Dict:
    """
    Kompletne przygotowanie danych: load → encode → split → extract features.

    Args:
        source: 'simulate' lub 'parquet'
        filepath: ścieżka do pliku parquet (jeśli source='parquet')
        n_simulate: liczba próbek do symulacji

    Returns:
        Słownik z kluczami: X_train, X_val, X_test, y_reg_*, y_clf_*, feature_names, encoders
    """
    # 1. Wczytaj
    if source == "simulate":
        df = simulate_dataset(n_samples=n_simulate)
    elif source == "parquet":
        if filepath is None:
            raise ValueError("filepath wymagany dla source='parquet'")
        df = load_from_parquet(filepath)
    else:
        raise ValueError(f"Nieznane źródło: {source}")

    # 2. Enkoduj zmienne kategoryczne
    df, encoders = encode_categoricals(df)

    # 3. Podział chronologiczny
    df_train, df_val, df_test = temporal_train_val_test_split(df)

    # 4. Wyciągnij macierze cech
    X_train, y_train_reg, y_train_clf, feature_names = get_feature_matrix(df_train)
    X_val, y_val_reg, y_val_clf, _ = get_feature_matrix(df_val, feature_names)
    X_test, y_test_reg, y_test_clf, _ = get_feature_matrix(df_test, feature_names)

    logger.info(f"Feature matrix shape: {X_train.shape[1]} cech")
    logger.info(f"  Cechy: {feature_names}")

    return {
        "X_train": X_train, "y_train_reg": y_train_reg, "y_train_clf": y_train_clf,
        "X_val": X_val,     "y_val_reg": y_val_reg,     "y_val_clf": y_val_clf,
        "X_test": X_test,   "y_test_reg": y_test_reg,   "y_test_clf": y_test_clf,
        "feature_names": feature_names,
        "encoders": encoders,
        "df_train": df_train,
        "df_val": df_val,
        "df_test": df_test,
    }

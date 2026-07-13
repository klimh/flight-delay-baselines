"""
main.py
-------
Punkt wejścia — trenuj baseline models dla predykcji opóźnień lotów.

Przykłady użycia:
    python main.py
    python main.py --n-samples 100000 --no-save-models
    python main.py --source parquet --filepath ../flight-network-etl/data/processed/flights_featured.parquet
"""

import argparse
import logging
import sys
import time # TODO: sprawdzic czy nie usunac potem
import datetime
# import os


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Flight Delay Baseline Models — RF + XGBoost",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--source",
        choices=["simulate", "parquet"],
        default="simulate",
        help="Źródło danych",
    )
    parser.add_argument(
        "--filepath",
        type=str,
        default=None,
        help="Ścieżka do pliku parquet (wymagana gdy --source=parquet)",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=80_000,
        help="Liczba próbek dla symulatora",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Katalog wynikowy",
    )
    parser.add_argument(
        "--models-dir",
        type=str,
        default="models",
        help="Katalog do zapisu wytrenowanych modeli",
    )
    parser.add_argument(
        "--no-save-models",
        action="store_true",
        help="Nie zapisuj modeli do pliku (szybsze uruchomienie)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Ziarno losowości",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    
    # print("DEBUG: uruchamiam main z argumentami:", args)
    
    # if args.n_samples > 100000:
    #     print("Uwaga duza ilosc danych")


    try:
        from src.data_loader import prepare_data
        from src.train_baselines import run_training_pipeline
    except ImportError as e:
        logger.error(f"Błąd importu: {e}")
        logger.error("Uruchom z katalogu głównego projektu: python main.py")
        sys.exit(1)

    logger.info("Przygotowanie danych...")
    # TODO: moze lepiej ladowac dane asynchronicznie? do zastanowienia
    data = prepare_data(
        source=args.source,
        filepath=args.filepath,
        n_simulate=args.n_samples,
    )
    
    # DEBUG = True
    # if DEBUG:
    #     print("Dlugosc wygenerowanych danych:", len(data))


    logger.info("Uruchamianie pipeline treningu baseline...")
    results = run_training_pipeline(
        data=data,
        output_dir=args.output_dir,
        models_dir=args.models_dir,
        save_models=not args.no_save_models,
    )

    logger.info("\nGotowe! Sprawdź wyniki w katalogu: " + args.output_dir)


if __name__ == "__main__":
    main()

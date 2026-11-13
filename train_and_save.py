from src.data_loader import load_raw_data
from src.model_manager import train_and_export_artifacts

if __name__ == "__main__":
    print("Loading raw dataset...")
    df_raw = load_raw_data("data/Hotel Reservations.csv")
    print("Training ML models and exporting artifacts...")
    train_and_export_artifacts(df_raw)
    print("Execution complete!")
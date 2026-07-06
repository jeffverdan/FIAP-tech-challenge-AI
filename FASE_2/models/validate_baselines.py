from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.impute import SimpleImputer


TARGET_COLUMN = "PCOS (Y/N)"
DROP_COLUMNS = ["Sl. No", "Patient File No.", "Unnamed: 44"]


def resolve_paths() -> tuple[Path, Path]:
    # Define os caminhos a partir da raiz do repositório para que o script funcione a partir de qualquer diretório atual.
    repo_root = Path(__file__).resolve().parents[2]
    dataset_path = repo_root / "FASE_1" / "data" / "PCOS_data_without_infertility.xlsx"
    models_dir = repo_root / "FASE_2" / "models"
    return dataset_path, models_dir


def load_reference_features(dataset_path: Path) -> pd.DataFrame:
    # Recria o mesmo pré-processamento de recursos usado durante a exportação.
    df = pd.read_excel(dataset_path, sheet_name="Full_new")
    cols_to_drop = [col for col in DROP_COLUMNS if col in df.columns]
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)

    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Aplica a imputação mediana para que as entradas de previsão correspondam às expectativas de treinamento.
    null_columns = df.columns[df.isnull().any()].tolist()
    if null_columns:
        imputer = SimpleImputer(strategy="median")
        df[null_columns] = imputer.fit_transform(df[null_columns])

    return df.drop(columns=[TARGET_COLUMN])


def main() -> None:
    dataset_path, models_dir = resolve_paths()
    X = load_reference_features(dataset_path)
    sample = X.head(5)

    # Falha rapidamente se algum artefato necessário não tiver sido gerado.
    required_artifacts = [
        "baseline_lr.joblib",
        "baseline_knn.joblib",
        "baseline_dt.joblib",
        "baseline_metadata.json",
    ]
    missing = [name for name in required_artifacts if not (models_dir / name).exists()]
    if missing:
        raise FileNotFoundError(f"Artefatos ausentes: {missing}")

    # Carrega os modelos e executa um teste rápido de inferência.
    lr = joblib.load(models_dir / "baseline_lr.joblib")
    knn = joblib.load(models_dir / "baseline_knn.joblib")
    dt = joblib.load(models_dir / "baseline_dt.joblib")
    metadata = json.loads((models_dir / "baseline_metadata.json").read_text(encoding="utf-8"))

    outputs = {
        "metadata_best_k": metadata.get("best_k"),
        "metadata_best_depth": metadata.get("best_depth"),
        "sample_size": len(sample),
        "lr_preds": lr.predict(sample).tolist(),
        "knn_preds": knn.predict(sample).tolist(),
        "dt_preds": dt.predict(sample).tolist(),
    }
    print("Validação concluída com sucesso.")
    print(json.dumps(outputs, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

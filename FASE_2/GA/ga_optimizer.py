from __future__ import annotations

import json
import logging
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import joblib
import numpy as np
import pandas as pd
from deap import base, creator, tools
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score, recall_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


logger = logging.getLogger("fase2.ga")


def configure_logging() -> None:
    """Configura o logging da aplicação. Nível controlado pela env var LOG_LEVEL (default INFO)."""
    level_name = os.getenv("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )


RANDOM_STATE = 42
TARGET_COLUMN = "PCOS (Y/N)"
AGE_COLUMN = " Age (yrs)"
DROP_COLUMNS = ["Sl. No", "Patient File No.", "Unnamed: 44"]


@dataclass
class ExperimentConfig:
    exp_id: str
    population_size: int
    generations: int
    cxpb: float
    mutpb: float
    tournament_size: int
    crossover: str  # "uniform" | "one_point"
    mutation_indpb: float
    elitism_count: int = 2
    early_stop_patience: int = 4


EXPERIMENTS = [
    ExperimentConfig(
        exp_id="high_mut_uniform",
        population_size=100,
        generations=20,
        cxpb=0.7,
        mutpb=0.5,
        tournament_size=3,
        crossover="uniform",
        mutation_indpb=0.4,
        elitism_count=2,
        early_stop_patience=4,
    ),
    ExperimentConfig(
        exp_id="low_mut_onepoint",
        population_size=100,
        generations=10,
        cxpb=0.8,
        mutpb=0.2, # Taxa de mutação mais baixa para observar o impacto na convergência.
        tournament_size=3,
        crossover="one_point",
        mutation_indpb=0.2, # Menos genes mutados por indivíduo para uma exploração mais conservadora.
        elitism_count=2,
        early_stop_patience=5,
    ),
    ExperimentConfig(
        exp_id="balanced_config",
        population_size=100,
        generations=20,
        cxpb=0.75,
        mutpb=0.3,
        tournament_size=3,
        crossover="uniform",
        mutation_indpb=0.3,
        elitism_count=2,
        early_stop_patience=5,
    ),
]


def resolve_paths() -> dict[str, Path]:
    repo_root = Path(__file__).resolve().parents[2]
    return {
        "repo_root": repo_root,
        "dataset": repo_root / "FASE_1" / "data" / "PCOS_data_without_infertility.xlsx",
        "models_dir": repo_root / "FASE_2" / "models",
        "results_dir": repo_root / "FASE_2" / "results",
    }


def load_dataset(dataset_path: Path) -> pd.DataFrame:
    df = pd.read_excel(dataset_path, sheet_name="Full_new")
    cols = [col for col in DROP_COLUMNS if col in df.columns]
    if cols:
        df = df.drop(columns=cols)
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    null_columns = df.columns[df.isnull().any()].tolist()
    if null_columns:
        imputer = SimpleImputer(strategy="median")
        df[null_columns] = imputer.fit_transform(df[null_columns])
    return df


def specificity_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0


def demographic_disparity(y_true: np.ndarray, y_pred: np.ndarray, age_values: np.ndarray) -> float:
    median_age = np.median(age_values)
    group_a = age_values <= median_age
    group_b = age_values > median_age
    if group_a.sum() == 0 or group_b.sum() == 0:
        return 0.0

    recall_a = recall_score(y_true[group_a], y_pred[group_a], pos_label=1, zero_division=0)
    recall_b = recall_score(y_true[group_b], y_pred[group_b], pos_label=1, zero_division=0)
    return abs(float(recall_a - recall_b))


def decode_individual(individual: list[float]) -> dict[str, float | str | None]:
    log10_c, penalty_gene, class_weight_gene, threshold = individual
    c_value = float(10 ** log10_c) # Transforma o gene logarítmico de volta para o espaço de C real.
    penalty = "l1" if round(penalty_gene) == 0 else "l2"
    class_weight = None if round(class_weight_gene) == 0 else "balanced"
    threshold = float(np.clip(threshold, 0.3, 0.7))
    return {
        "C": c_value,
        "penalty": penalty,
        "class_weight": class_weight,
        "threshold": threshold,
    }
# type: ignore

def build_pipeline(params: dict[str, float | str | None]) -> Pipeline:
    model = LogisticRegression(
        C=float(params["C"]),
        penalty=str(params["penalty"]),
        class_weight=params["class_weight"], 
        solver="liblinear",
        max_iter=2000,
        random_state=RANDOM_STATE,
    )
    return Pipeline([("scaler", StandardScaler()), ("model", model)])


def create_fitness_evaluator(
    X: pd.DataFrame,
    y: pd.Series,
    age: pd.Series,
    metric_weights: dict[str, float],
    cv_splits: int = 5,
) -> Callable[[list[float]], tuple[float]]:
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=RANDOM_STATE)

    def evaluate(individual: list[float]) -> tuple[float]:
        params = decode_individual(individual)

        recalls, specificities, f1s, disparities = [], [], [], []
        for train_idx, val_idx in cv.split(X, y):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            age_val = age.iloc[val_idx].to_numpy()

            pipeline = build_pipeline(params)
            pipeline.fit(X_train, y_train)

            probs = pipeline.predict_proba(X_val)[:, 1]
            preds = (probs >= params["threshold"]).astype(int)

            recalls.append(recall_score(y_val, preds, pos_label=1, zero_division=0))
            specificities.append(specificity_score(y_val.to_numpy(), preds))
            f1s.append(f1_score(y_val, preds, pos_label=1, zero_division=0))
            disparities.append(demographic_disparity(y_val.to_numpy(), preds, age_val))

        mean_recall = float(np.mean(recalls))
        mean_specificity = float(np.mean(specificities))
        mean_f1 = float(np.mean(f1s))
        mean_disparity = float(np.mean(disparities))

        fitness = (
            metric_weights["recall"] * mean_recall
            + metric_weights["specificity"] * mean_specificity
            + metric_weights["f1"] * mean_f1
            - metric_weights["disparity"] * mean_disparity
        )
        return (fitness,)

    return evaluate


def mutation_operator(individual: list[float], indpb: float) -> tuple[list[float]]:
    if random.random() < indpb:
        individual[0] = float(np.clip(individual[0] + random.uniform(-0.4, 0.4), -2.0, 1.0))
    if random.random() < indpb:
        individual[1] = 1.0 - round(individual[1])
    if random.random() < indpb:
        individual[2] = 1.0 - round(individual[2])
    if random.random() < indpb:
        individual[3] = float(np.clip(individual[3] + random.uniform(-0.1, 0.1), 0.3, 0.7))
    return (individual,)


def evaluate_on_test(
    pipeline: Pipeline,
    threshold: float,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    age_test: pd.Series,
) -> dict[str, float]:
    probs = pipeline.predict_proba(X_test)[:, 1]
    preds = (probs >= threshold).astype(int)
    return {
        "recall": float(recall_score(y_test, preds, pos_label=1, zero_division=0)),
        "specificity": float(specificity_score(y_test.to_numpy(), preds)),
        "f1": float(f1_score(y_test, preds, pos_label=1, zero_division=0)),
        "disparity": float(demographic_disparity(y_test.to_numpy(), preds, age_test.to_numpy())),
    }


def run_single_experiment(
    cfg: ExperimentConfig,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    age_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    age_test: pd.Series,
    models_dir: Path,
) -> dict[str, float | int | str]:
    random.seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)

    metric_weights = {"recall": 0.4, "specificity": 0.2, "f1": 0.3, "disparity": 0.1}
    logger.info("=" * 88)
    logger.info("Iniciando experimento: %s", cfg.exp_id)
    logger.info(
        "Config -> pop=%s, gen=%s, cxpb=%s, mutpb=%s, crossover=%s, "
        "mutation_indpb=%s, tournament=%s, elitism=%s, early_stop_patience=%s",
        cfg.population_size, cfg.generations, cfg.cxpb, cfg.mutpb, cfg.crossover,
        cfg.mutation_indpb, cfg.tournament_size, cfg.elitism_count, cfg.early_stop_patience,
    )
    logger.info("Pesos fitness -> %s", metric_weights)

    if "FitnessMaxGA" not in creator.__dict__:
        creator.create("FitnessMaxGA", base.Fitness, weights=(1.0,))
    if "IndividualGA" not in creator.__dict__:
        creator.create("IndividualGA", list, fitness=creator.FitnessMaxGA)

    toolbox = base.Toolbox()
    toolbox.register("gene_log10_c", random.uniform, -2.0, 1.0)
    toolbox.register("gene_penalty", random.randint, 0, 1)
    toolbox.register("gene_class_weight", random.randint, 0, 1)
    toolbox.register("gene_threshold", random.uniform, 0.3, 0.7)
    toolbox.register(
        "individual",
        tools.initCycle,
        creator.IndividualGA,
        (
            toolbox.gene_log10_c,
            toolbox.gene_penalty,
            toolbox.gene_class_weight,
            toolbox.gene_threshold,
        ),
        n=1,
    )
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register(
        "evaluate",
        create_fitness_evaluator(
            X=X_train,
            y=y_train,
            age=age_train,
            metric_weights=metric_weights,
            cv_splits=5,
        ),
    )
    toolbox.register("select", tools.selTournament, tournsize=cfg.tournament_size)
    if cfg.crossover == "uniform":
        toolbox.register("mate", tools.cxUniform, indpb=0.5)
    else:
        toolbox.register("mate", tools.cxOnePoint)
    toolbox.register("mutate", mutation_operator, indpb=cfg.mutation_indpb)

    start = time.perf_counter()

    pop = toolbox.population(n=cfg.population_size)
    for ind in pop:
        ind.fitness.values = toolbox.evaluate(ind)
    initial_best = tools.selBest(pop, 1)[0]
    logger.info("Fitness inicial (melhor individuo): %.6f", initial_best.fitness.values[0])
    best_fitness_so_far = float(initial_best.fitness.values[0])
    stagnation_counter = 0

    for gen_idx in range(cfg.generations):
        elites = list(map(toolbox.clone, tools.selBest(pop, cfg.elitism_count)))
        offspring = toolbox.select(pop, len(pop))
        offspring = list(map(toolbox.clone, offspring))

        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < cfg.cxpb:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            if random.random() < cfg.mutpb:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        invalid = [ind for ind in offspring if not ind.fitness.valid]
        for ind in invalid:
            ind.fitness.values = toolbox.evaluate(ind)

        # Explicit elitism: preserve top-N previous-generation individuals.
        offspring_sorted = sorted(offspring, key=lambda ind: ind.fitness.values[0])
        for elite in elites:
            offspring_sorted[0] = elite
            offspring_sorted.sort(key=lambda ind: ind.fitness.values[0])
        offspring = offspring_sorted

        pop[:] = offspring
        gen_best = tools.selBest(pop, 1)[0]
        current_best_fitness = float(gen_best.fitness.values[0])
        logger.info(
            "Geração %02d/%s -> best_fitness=%.6f",
            gen_idx + 1, cfg.generations, current_best_fitness,
        )

        if current_best_fitness > best_fitness_so_far + 1e-12:
            best_fitness_so_far = current_best_fitness
            stagnation_counter = 0
        else:
            stagnation_counter += 1

        if stagnation_counter >= cfg.early_stop_patience:
            logger.warning(
                "Early stopping em %s: sem melhora por %s gerações.",
                cfg.exp_id, cfg.early_stop_patience,
            )
            break

    best = tools.selBest(pop, 1)[0]
    best_params = decode_individual(best)
    logger.info("Melhor indivíduo (%s) -> %s", cfg.exp_id, best_params)
    logger.info("Melhor fitness final (%s) -> %.6f", cfg.exp_id, best.fitness.values[0])
    best_pipeline = build_pipeline(best_params)
    best_pipeline.fit(X_train, y_train)

    test_metrics = evaluate_on_test(
        pipeline=best_pipeline,
        threshold=float(best_params["threshold"]),
        X_test=X_test,
        y_test=y_test,
        age_test=age_test,
    )
    elapsed = time.perf_counter() - start
    logger.info(
        "Métricas teste (%s) -> recall=%.4f, specificity=%.4f, f1=%.4f, disparity=%.4f",
        cfg.exp_id, test_metrics["recall"], test_metrics["specificity"],
        test_metrics["f1"], test_metrics["disparity"],
    )
    logger.info("Tempo total (%s) -> %.2fs", cfg.exp_id, elapsed)

    model_payload = {
        "pipeline": best_pipeline,
        "threshold": float(best_params["threshold"]),
        "best_params": best_params,
        "fitness": float(best.fitness.values[0]),
        "experiment": cfg.exp_id,
    }
    model_path = models_dir / f"ga_{cfg.exp_id}.joblib"
    joblib.dump(model_payload, model_path)

    return {
        "exp_id": cfg.exp_id,
        "population": cfg.population_size,
        "generations": cfg.generations,
        "mutation_rate": cfg.mutpb,
        "crossover_rate": cfg.cxpb,
        "crossover_type": cfg.crossover,
        "best_fitness": float(best.fitness.values[0]),
        "best_C": float(best_params["C"]),
        "best_penalty": str(best_params["penalty"]),
        "best_class_weight": str(best_params["class_weight"]),
        "best_threshold": float(best_params["threshold"]),
        "test_recall": test_metrics["recall"],
        "test_specificity": test_metrics["specificity"],
        "test_f1": test_metrics["f1"],
        "test_disparity": test_metrics["disparity"],
        "runtime_seconds": float(elapsed),
        "saved_model": str(model_path),
    }


def evaluate_baseline_lr(
    baseline_path: Path,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    age_test: pd.Series,
) -> dict[str, float]:
    baseline = joblib.load(baseline_path)
    preds = baseline.predict(X_test)
    return {
        "recall": float(recall_score(y_test, preds, pos_label=1, zero_division=0)),
        "specificity": float(specificity_score(y_test.to_numpy(), preds)),
        "f1": float(f1_score(y_test, preds, pos_label=1, zero_division=0)),
        "disparity": float(demographic_disparity(y_test.to_numpy(), preds, age_test.to_numpy())),
    }


def main() -> None:
    configure_logging()
    paths = resolve_paths()
    paths["models_dir"].mkdir(parents=True, exist_ok=True)
    paths["results_dir"].mkdir(parents=True, exist_ok=True)

    df = load_dataset(paths["dataset"])
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN].astype(int)
    age = df[AGE_COLUMN]

    X_train, X_test, y_train, y_test, age_train, age_test = train_test_split(
        X,
        y,
        age,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y, # Mantém proporção das classes.
    )

    rows = []
    for cfg in EXPERIMENTS:
        rows.append(
            run_single_experiment(
                cfg=cfg,
                X_train=X_train,
                y_train=y_train,
                age_train=age_train,
                X_test=X_test,
                y_test=y_test,
                age_test=age_test,
                models_dir=paths["models_dir"],
            )
        )
        logger.info("Artefato salvo -> models/ga_%s.joblib", cfg.exp_id)

    baseline_lr_path = paths["models_dir"] / "baseline_lr.joblib"
    if baseline_lr_path.exists():
        baseline_metrics = evaluate_baseline_lr(
            baseline_path=baseline_lr_path,
            X_test=X_test,
            y_test=y_test,
            age_test=age_test,
        )
    else:
        baseline_metrics = {"recall": np.nan, "specificity": np.nan, "f1": np.nan, "disparity": np.nan}

    for row in rows:
        row["baseline_recall"] = baseline_metrics["recall"]
        row["baseline_specificity"] = baseline_metrics["specificity"]
        row["baseline_f1"] = baseline_metrics["f1"]
        row["baseline_disparity"] = baseline_metrics["disparity"]
        row["delta_f1_vs_baseline"] = row["test_f1"] - baseline_metrics["f1"] if pd.notna(baseline_metrics["f1"]) else np.nan
        row["delta_recall_vs_baseline"] = (
            row["test_recall"] - baseline_metrics["recall"] if pd.notna(baseline_metrics["recall"]) else np.nan
        )
        if pd.notna(baseline_metrics["f1"]):
            logger.info(
                "Comparação vs baseline (%s) -> delta_f1=%+.4f, delta_recall=%+.4f",
                row["exp_id"], row["delta_f1_vs_baseline"], row["delta_recall_vs_baseline"],
            )

    df_results = pd.DataFrame(rows)
    results_csv = paths["results_dir"] / "ga_experiments.csv"
    df_results.to_csv(results_csv, index=False)

    summary = {
        "results_csv": str(results_csv),
        "experiments_run": len(rows),
        "best_experiment_by_fitness": df_results.sort_values("best_fitness", ascending=False).iloc[0]["exp_id"],
    }
    logger.info("CSV de resultados salvo em -> %s", results_csv)
    logger.info("Melhor experimento por fitness -> %s", summary["best_experiment_by_fitness"])
    (paths["results_dir"] / "ga_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    logger.info("Resumo final:\n%s", json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split


RANDOM_STATE = 42
TARGET_COLUMN = "PCOS (Y/N)"
AGE_COLUMN = " Age (yrs)"
DROP_COLUMNS = ["Sl. No", "Patient File No.", "Unnamed: 44"]


def resolve_paths() -> dict[str, Path]:
    repo_root = Path(__file__).resolve().parents[2]
    return {
        "repo_root": repo_root,
        "dataset": repo_root / "FASE_1" / "data" / "PCOS_data_without_infertility.xlsx",
        "models_dir": repo_root / "FASE_2" / "models",
        "llm_outputs": repo_root / "FASE_2" / "llm_outputs",
    }


def load_dataset(dataset_path: Path) -> pd.DataFrame:
    df = pd.read_excel(dataset_path, sheet_name="Full_new")
    cols = [c for c in DROP_COLUMNS if c in df.columns]
    if cols:
        df = df.drop(columns=cols)
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    null_columns = df.columns[df.isnull().any()].tolist()
    if null_columns:
        imputer = SimpleImputer(strategy="median")
        df[null_columns] = imputer.fit_transform(df[null_columns])
    return df


def get_test_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN].astype(int)
    _X_train, X_test, _y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    return X_test, y_test


def anonymize_patient_row(row: pd.Series) -> dict[str, Any]:
    # Remove identificadores diretos e mantém apenas contexto clínico/sociodemográfico.
    data = row.to_dict()
    data.pop("Patient File No.", None)
    data.pop("Sl. No", None)
    return {str(k).strip(): float(v) if isinstance(v, (int, float, np.number)) else v for k, v in data.items()}


def top_features_for_patient(
    model_payload: dict[str, Any], patient_df: pd.DataFrame, top_k: int = 5
) -> list[tuple[str, float]]:
    pipeline = model_payload["pipeline"]
    model = pipeline.named_steps["model"]
    scaler = pipeline.named_steps["scaler"]

    # Coeficientes da regressão logística indicam importância global das features;
    # multiplicamos pelo valor normalizado da paciente para obter direção/contribuição local.
    x_scaled = scaler.transform(patient_df)[0]
    coefs = model.coef_[0]
    contribution = x_scaled * coefs
    feature_names = patient_df.columns.tolist()

    ranked_idx = np.argsort(np.abs(contribution))[::-1][:top_k]
    return [(feature_names[i], float(contribution[i])) for i in ranked_idx]


def build_prompt(
    patient_id: str,
    anonymized_data: dict[str, Any],
    probability_pcos: float,
    predicted_label: int,
    top_features: list[tuple[str, float]],
) -> str:
    social_context = {
        "faixa_etaria": "adulta jovem" if anonymized_data.get("Age (yrs)", 30) < 35 else "adulta",
        "necessidade_linguagem": "clara, acolhedora e não alarmista",
        "confidencialidade": "não incluir identificadores diretos e evitar exposição desnecessária de dados",
    }

    return f"""
Você é um assistente clínico para saúde da mulher.

Contexto:
- Paciente anonimizada: {patient_id}
- Probabilidade de PCOS prevista pelo modelo: {probability_pcos:.3f}
- Predição binária do modelo (1=PCOS, 0=Não PCOS): {predicted_label}

Dados clínicos resumidos (anonimizados):
{json.dumps(anonymized_data, ensure_ascii=False)}

Principais features (contribuição local para a predição):
{json.dumps(top_features, ensure_ascii=False)}

Contexto social e de comunicação:
{json.dumps(social_context, ensure_ascii=False)}

Tarefas:
1) Explique a predição em linguagem natural para profissional de saúde.
2) Explique os fatores mais relevantes do caso.
3) Traga recomendações práticas e próximos passos clínicos sugeridos.
4) Inclua um aviso de confidencialidade e limitação do modelo.
5) Use linguagem sensível ao contexto de saúde da mulher.

Responda em português com 4 seções:
- Interpretação do caso
- Fatores relevantes
- Recomendações práticas
- Confidencialidade e limites
""".strip()


def fallback_local_explanation(
    probability_pcos: float, predicted_label: int, top_features: list[tuple[str, float]]
) -> str:
    risk = "elevada" if predicted_label == 1 else "baixa a moderada"
    top_txt = ", ".join([f"{name} ({value:+.2f})" for name, value in top_features])
    return (
        "Interpretação do caso:\n"
        f"- O modelo sugere probabilidade {probability_pcos:.2%} de PCOS, com risco {risk}.\n\n"
        "Fatores relevantes:\n"
        f"- Variáveis com maior contribuição local: {top_txt}.\n"
        "- Esses sinais devem ser interpretados junto ao histórico clínico e exame físico.\n\n"
        "Recomendações práticas:\n"
        "- Confirmar critérios clínicos/laboratoriais e ultrassonográficos de PCOS.\n"
        "- Revisar risco metabólico (glicemia, perfil lipídico, IMC, estilo de vida).\n"
        "- Planejar seguimento com abordagem multiprofissional e educação em saúde.\n\n"
        "Confidencialidade e limites:\n"
        "- Esta interpretação usa dados anonimizados e não substitui decisão médica.\n"
        "- O modelo pode errar; usar apenas como apoio à triagem e priorização."
    )


def call_openai(prompt: str, model_name: str) -> str:
    from openai import OpenAI

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    response = client.responses.create(
        model=model_name,
        input=prompt,
        temperature=0.2,
    )
    return response.output_text.strip()


def qualitative_score(text: str) -> dict[str, int]:
    t = text.lower()    
    # Rubrica heurística (1-5) para avaliar a qualidade textual da resposta da LLM e gerar uma nota rápida (rubrica) por critério.
    # Essas regras são simplificadas e podem ser aprimoradas com técnicas de NLP ou análise semântica mais avançada.
    medical_precision = 4 if ("critérios" in t or "clínic" in t) else 3
    cultural_sensitivity = 4 if ("saúde da mulher" in t or "acolhedora" in t or "sensível" in t) else 3
    language_adequacy = 4 if ("recomendações" in t and "limites" in t) else 3
    return {
        "medical_precision": medical_precision,
        "cultural_sensitivity": cultural_sensitivity,
        "language_adequacy": language_adequacy,
        "overall": int(round((medical_precision + cultural_sensitivity + language_adequacy) / 3)),
    }


def run(args: argparse.Namespace) -> None:
    paths = resolve_paths()
    paths["llm_outputs"].mkdir(parents=True, exist_ok=True)
    print("[LLM] Iniciando integração LLM...")
    print(f"[LLM] Dataset: {paths['dataset']}")
    print(f"[LLM] Diretório de saída: {paths['llm_outputs']}")

    df = load_dataset(paths["dataset"])
    print(f"[LLM] Dataset carregado com {len(df)} linhas e {len(df.columns)} colunas.")
    X_test, y_test = get_test_split(df)
    print(f"[LLM] Conjunto de teste preparado com {len(X_test)} amostras.")

    model_file = paths["models_dir"] / args.model_file
    print(f"[LLM] Carregando modelo otimizado: {model_file}")
    payload = joblib.load(model_file)
    pipeline = payload["pipeline"]
    threshold = float(payload.get("threshold", 0.5))
    print(f"[LLM] Threshold de decisão carregado: {threshold:.3f}")

    sample_count = min(args.sample_count, len(X_test))
    print(f"[LLM] sample_count solicitado={args.sample_count} | usado={sample_count}")
    samples = X_test.head(sample_count).copy()
    y_ref = y_test.head(sample_count).copy()

    results = []
    jsonl_records = []
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    mode = "openai_api" if os.getenv("OPENAI_API_KEY") and not args.force_local else "local_fallback"
    print(f"[LLM] Modo de execução: {mode}")
    if mode == "openai_api":
        print(f"[LLM] Modelo OpenAI selecionado: {args.openai_model}")
    else:
        print("[LLM] API desabilitada ou sem chave; usando fallback local.")

    for idx, (row_idx, row) in enumerate(samples.iterrows(), start=1):
        patient_df = row.to_frame().T
        probs = pipeline.predict_proba(patient_df)[:, 1]
        prob = float(probs[0])
        pred = int(prob >= threshold)

        patient_id = f"P_{idx:03d}"
        anonymized = anonymize_patient_row(row)
        top_feats = top_features_for_patient(payload, patient_df, top_k=5)
        prompt = build_prompt(patient_id, anonymized, prob, pred, top_feats)
        print(
            f"[LLM] {patient_id} -> prob={prob:.3f}, pred={pred}, "
            f"features_top1={top_feats[0][0] if top_feats else 'n/a'}"
        )
        print(f"[LLM] {patient_id} -> tamanho prompt: {len(prompt)} caracteres")

        if mode == "openai_api":
            try:
                print(f"[LLM] {patient_id} -> chamando OpenAI API...")
                llm_text = call_openai(prompt, args.openai_model)
                print(f"[LLM] {patient_id} -> resposta recebida da API.")
            except Exception as exc:  # fallback caso ocorram problemas de API/rede
                print(f"[LLM] {patient_id} -> falha API ({type(exc).__name__}), usando fallback.")
                llm_text = (
                    f"[fallback] Falha na chamada API ({type(exc).__name__}).\n\n"
                    + fallback_local_explanation(prob, pred, top_feats)
                )
        else:
            llm_text = fallback_local_explanation(prob, pred, top_feats)

        rubric = qualitative_score(llm_text)
        print(
            f"[LLM] {patient_id} -> scores: "
            f"medical={rubric['medical_precision']}, "
            f"cultural={rubric['cultural_sensitivity']}, "
            f"language={rubric['language_adequacy']}, overall={rubric['overall']}"
        )
        row_result = {
            "timestamp_utc": ts,
            "patient_id": patient_id,
            "sample_index": int(row_idx),
            "mode": mode,
            "model_file": args.model_file,
            "predicted_label": pred,
            "predicted_probability_pcos": prob,
            "ground_truth_label": int(y_ref.loc[row_idx]),
            "top_features": json.dumps(top_feats, ensure_ascii=False),
            "llm_response": llm_text,
            **rubric,
        }
        results.append(row_result)

        jsonl_records.append(
            {
                "metadata": {
                    "timestamp_utc": ts,
                    "patient_id": patient_id,
                    "mode": mode,
                    "model_file": args.model_file,
                    "predicted_label": pred,
                    "predicted_probability_pcos": prob,
                    "ground_truth_label": int(y_ref.loc[row_idx]),
                },
                "prompt": prompt,
                "response": llm_text,
                "qualitative_scores": rubric,
            }
        )
        print(f"[LLM] Caso {patient_id} processado (pred={pred}, p={prob:.3f}).")

    out_csv = paths["llm_outputs"] / f"llm_responses_{ts}.csv"
    out_jsonl = paths["llm_outputs"] / f"llm_responses_{ts}.jsonl"
    out_matrix = paths["llm_outputs"] / f"llm_qualitative_matrix_{ts}.csv"

    df_out = pd.DataFrame(results)
    print(f"[LLM] Salvando {len(df_out)} respostas...")
    df_out.to_csv(out_csv, index=False)
    df_out[
        ["patient_id", "medical_precision", "cultural_sensitivity", "language_adequacy", "overall"]
    ].to_csv(out_matrix, index=False)

    with out_jsonl.open("w", encoding="utf-8") as f:
        for rec in jsonl_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"[LLM] Arquivo CSV salvo em: {out_csv}")
    print(f"[LLM] Arquivo JSONL salvo em: {out_jsonl}")
    print(f"[LLM] Matriz qualitativa salva em: {out_matrix}")

    summary = {
        "mode": mode,
        "sample_count": sample_count,
        "csv": str(out_csv),
        "jsonl": str(out_jsonl),
        "qualitative_matrix": str(out_matrix),
        "avg_overall_score": float(df_out["overall"].mean()) if len(df_out) else None,
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Etapa 4 - Integração com LLM para explicações clínicas.")
    parser.add_argument("--model-file", default="ga_balanced_config.joblib", help="Arquivo .joblib em FASE_2/models")
    parser.add_argument("--sample-count", type=int, default=8, help="Quantidade de casos para gerar explicações")
    parser.add_argument("--openai-model", default="gpt-4.1-mini", help="Modelo OpenAI (se usar API)")
    parser.add_argument(
        "--force-local",
        action="store_true",
        help="Força modo local (sem chamada de API), útil para estudo/offline.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())

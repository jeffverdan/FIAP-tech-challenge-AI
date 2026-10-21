"""Baixa os subsets dos datasets públicos usados na FASE 4.

Uso:
    python scripts/download_data.py                 # tudo
    python scripts/download_data.py --only vitaldb mimic
    python scripts/download_data.py --only rehab --rehab-exercicios 1 6

Os arquivos vão para ``data/raw/<dataset>/`` (ignorado pelo git). O script é idempotente:
arquivos já baixados são pulados.

Datasets e licenças:
  * VitalDB (PhysioNet / api.vitaldb.net) — CC BY-NC-SA 4.0 / ODC-By (open access)
  * MIMIC-IV Clinical Database Demo v2.2 (PhysioNet) — ODbL (open access, sem credenciamento)
  * PriMock57 (Babylon Health, GitHub) — consultas simuladas de atenção primária
  * TORGO (Univ. Toronto) — subset espelhado em huggingface.co/datasets/birgermoell/dysarthria
  * Coswara (IISc Bangalore, GitHub) — CC BY 4.0; voz + sintomas respiratórios autorrelatados
  * REHAB24-6 (Zenodo 13305826) — CC BY-NC 4.0, uso acadêmico
  * UR Fall Detection Dataset (Univ. Rzeszów) — uso acadêmico
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from monitor.config import CONFIG  # noqa: E402

RAW = CONFIG.raw_dir
SESSAO = requests.Session()
SESSAO.headers["User-Agent"] = "FIAP-TechChallenge-Fase4/1.0 (academic)"


def baixar(url: str, destino: Path, minimo_bytes: int = 1, tentativas: int = 5) -> Path:
    """Download idempotente com novas tentativas (arquivos grandes às vezes caem no meio)."""
    import time

    if destino.exists() and destino.stat().st_size >= minimo_bytes:
        return destino
    destino.parent.mkdir(parents=True, exist_ok=True)
    tmp = destino.with_suffix(destino.suffix + ".part")
    for tentativa in range(1, tentativas + 1):
        try:
            with SESSAO.get(url, stream=True, timeout=120) as r:
                r.raise_for_status()
                with open(tmp, "wb") as f:
                    for bloco in r.iter_content(1 << 20):
                        f.write(bloco)
            tmp.replace(destino)
            return destino
        except requests.HTTPError:
            raise
        except requests.RequestException as e:
            if tentativa == tentativas:
                raise
            print(f"  {destino.name}: tentativa {tentativa} falhou ({type(e).__name__}); tentando de novo…")
            time.sleep(5 * tentativa)
    return destino


# --------------------------------------------------------------------------------------
# Sinais vitais
# --------------------------------------------------------------------------------------
VITALDB_TRILHAS = [
    "Solar8000/HR", "Solar8000/PLETH_SPO2", "Solar8000/ART_SBP", "Solar8000/ART_DBP",
    "Solar8000/ART_MBP", "Solar8000/NIBP_SBP", "Solar8000/NIBP_DBP", "Solar8000/NIBP_MBP",
    "Solar8000/RR_CO2", "Solar8000/BT",
]
VITALDB_TRILHAS_OBRIGATORIAS = {"Solar8000/ART_MBP", "Solar8000/ART_SBP", "Solar8000/PLETH_SPO2", "Solar8000/HR"}


def selecionar_casos_vitaldb(n: int = 100) -> list[int]:
    """Critério reprodutível: anestesia geral, ≥ 2 h de duração e monitorização completa
    (PA invasiva + SpO2 + FC). Pega os ``n`` primeiros por ``caseid``."""
    import pandas as pd

    trks = pd.read_csv("https://api.vitaldb.net/trks")
    cases = pd.read_csv("https://api.vitaldb.net/cases")
    cobertos = trks[trks["tname"].isin(VITALDB_TRILHAS_OBRIGATORIAS)].groupby("caseid")["tname"].nunique()
    cobertos = set(cobertos[cobertos == len(VITALDB_TRILHAS_OBRIGATORIAS)].index)
    dur_min = (cases["caseend"] - cases["casestart"]) / 60
    ok = cases[cases["caseid"].isin(cobertos) & (cases["ane_type"] == "General") & (dur_min >= 120)]
    return sorted(ok["caseid"].tolist())[:n]


def baixar_vitaldb(n_casos: int = 100, intervalo_s: float = 2.0) -> None:
    import json

    import numpy as np
    import pandas as pd
    import vitaldb

    destino = RAW / "vitaldb"
    destino.mkdir(parents=True, exist_ok=True)
    baixar("https://api.vitaldb.net/cases", destino / "clinical_info.csv")
    lista = destino / "casos.json"
    if lista.exists():
        casos = json.loads(lista.read_text())
    else:
        casos = selecionar_casos_vitaldb(n_casos)
        lista.write_text(json.dumps(casos))
    for i, caso in enumerate(casos, 1):
        arq = destino / f"case_{caso:04d}.csv.gz"
        if arq.exists():
            continue
        vals = vitaldb.load_case(caso, VITALDB_TRILHAS, interval=intervalo_s)
        df = pd.DataFrame(vals, columns=[t.split("/")[1] for t in VITALDB_TRILHAS])
        df.insert(0, "t_s", np.arange(len(df)) * intervalo_s)
        df.to_csv(arq, index=False, compression="gzip")
        if i % 10 == 0:
            print(f"  vitaldb {i}/{len(casos)}")
    print(f"  vitaldb: {len(casos)} casos")


MIMIC_BASE = "https://physionet.org/files/mimic-iv-demo/2.2"
MIMIC_TABELAS = [
    "hosp/patients.csv.gz", "hosp/admissions.csv.gz", "hosp/prescriptions.csv.gz",
    "hosp/pharmacy.csv.gz", "hosp/diagnoses_icd.csv.gz", "hosp/d_icd_diagnoses.csv.gz",
    "icu/icustays.csv.gz", "icu/d_items.csv.gz", "icu/chartevents.csv.gz",
]


def baixar_mimic() -> None:
    for tabela in MIMIC_TABELAS:
        baixar(f"{MIMIC_BASE}/{tabela}", RAW / "mimic-iv-demo" / tabela)
        print(f"  mimic {tabela}")


# --------------------------------------------------------------------------------------
# Áudio
# --------------------------------------------------------------------------------------
PRIMOCK_RAW = "https://raw.githubusercontent.com/babylonhealth/primock57/main"
PRIMOCK_MEDIA = "https://media.githubusercontent.com/media/babylonhealth/primock57/main"
#: Queixas respiratórias, de fadiga, cardíacas e neurológicas + 3 controles (pele, urinária, ansiedade).
PRIMOCK_CONSULTAS = [
    "day1_consultation06",  # hard to breathe
    "day2_consultation04",  # I feel breathless
    "day2_consultation06",  # difficulty breathing
    "day2_consultation07",  # chest discomfort
    "day2_consultation09",  # can't move my left arm
    "day3_consultation03",  # no appetite or energy
    "day4_consultation06",  # chest hurts, hard to breathe
    "day5_consultation09",  # tired all the time
    "day1_consultation02",  # controle: pele
    "day1_consultation09",  # controle: dor ao urinar
    "day5_consultation03",  # controle: ansiedade
]


def baixar_primock(consultas=PRIMOCK_CONSULTAS) -> None:
    destino = RAW / "primock57"
    for c in consultas:
        for papel in ("patient", "doctor"):
            baixar(f"{PRIMOCK_MEDIA}/audio/{c}_{papel}.wav", destino / "audio" / f"{c}_{papel}.wav", 10_000)
            baixar(f"{PRIMOCK_RAW}/transcripts/{c}_{papel}.TextGrid",
                   destino / "transcripts" / f"{c}_{papel}.TextGrid")
        baixar(f"{PRIMOCK_RAW}/notes/{c}.json", destino / "notes" / f"{c}.json")
        print(f"  primock57 {c}")


TORGO_HF = "https://huggingface.co/datasets/birgermoell/dysarthria/resolve/main"


def baixar_torgo() -> None:
    """Baixa as 400 gravações do espelho (15 falantes: 8 com disartria, 7 controles).

    O rótulo vem da pasta (``dysarthria_*`` / ``non_dysarthria_*``) e o falante do prefixo do
    arquivo (F01, MC03…), o que permite validação cruzada *por falante*.
    """
    from huggingface_hub import list_repo_files

    destino = RAW / "torgo"
    arquivos = [f for f in list_repo_files("birgermoell/dysarthria", repo_type="dataset") if f.endswith(".wav")]
    for i, rel in enumerate(sorted(arquivos), 1):
        baixar(f"{TORGO_HF}/{rel}", destino / rel, 1000)
        if i % 100 == 0:
            print(f"  torgo {i}/{len(arquivos)}")
    falantes = {Path(f).name.split("_")[0] for f in arquivos}
    print(f"  torgo: {len(arquivos)} arquivos, {len(falantes)} falantes")


COSWARA_RAW = "https://raw.githubusercontent.com/iiscleap/Coswara-Data/master"
#: Lotes com mais participantes que relataram dificuldade respiratória (bd) ou fadiga (ftg),
#: + um lote de abril/2020 majoritariamente saudável (controles).
COSWARA_LOTES = {"20210830": 2, "20210630": 3, "20210914": 2, "20210816": 4, "20210930": 5, "20200413": 4}
COSWARA_ARQUIVOS = ("counting-normal.wav", "vowel-a.wav")


def baixar_coswara(lotes=COSWARA_LOTES) -> None:
    """Coswara (IISc Bangalore): gravações de contagem 1–20 e vogal sustentada com sintomas
    autorrelatados. Só extrai os dois arquivos usados por participante e apaga os .tar.gz."""
    import string
    import tarfile

    destino = RAW / "coswara"
    baixar(f"{COSWARA_RAW}/combined_data.csv", destino / "combined_data.csv")
    for lote, n_partes in lotes.items():
        marcador = destino / lote / ".ok"
        if marcador.exists():
            continue
        partes = []
        for i in range(n_partes):
            sufixo = "a" + string.ascii_lowercase[i]
            partes.append(baixar(f"{COSWARA_RAW}/{lote}/{lote}.tar.gz.{sufixo}", destino / "_tmp" / f"{lote}.{sufixo}"))
        juntos = destino / "_tmp" / f"{lote}.tar.gz"
        with open(juntos, "wb") as out:
            for parte in partes:
                out.write(parte.read_bytes())
        n = 0
        with tarfile.open(juntos, "r:gz") as tar:
            for m in tar:
                if m.isfile() and Path(m.name).name in COSWARA_ARQUIVOS:
                    pid = Path(m.name).parent.name
                    alvo = destino / lote / pid / Path(m.name).name
                    alvo.parent.mkdir(parents=True, exist_ok=True)
                    alvo.write_bytes(tar.extractfile(m).read())
                    n += 1
        for arq in [*partes, juntos]:
            arq.unlink()
        marcador.write_text(str(n))
        print(f"  coswara {lote}: {n} arquivos")


# --------------------------------------------------------------------------------------
# Vídeo
# --------------------------------------------------------------------------------------
REHAB_ZENODO = "https://zenodo.org/records/13305826/files"


def baixar_rehab(exercicios=(1, 6), camera: str = "Camera17", tentativas: int = 5) -> None:
    """Extrai do ``videos.zip`` (2,6 GB) só os vídeos dos exercícios pedidos, via HTTP Range."""
    import time

    from remotezip import RemoteZip

    destino = RAW / "rehab24-6"
    baixar(f"{REHAB_ZENODO}/Segmentation.csv?download=1", destino / "Segmentation.csv")
    for tentativa in range(1, tentativas + 1):
        try:
            with RemoteZip(f"{REHAB_ZENODO}/videos.zip?download=1") as z:
                for info in z.infolist():
                    nome = info.filename
                    if not nome.endswith(".mp4") or camera not in nome:
                        continue
                    if int(nome.split("/")[0].removeprefix("Ex")) not in exercicios:
                        continue
                    alvo = destino / "videos" / nome
                    if alvo.exists():
                        continue
                    alvo.parent.mkdir(parents=True, exist_ok=True)
                    tmp = alvo.with_suffix(".part")
                    tmp.write_bytes(z.read(info))
                    tmp.replace(alvo)
                    print(f"  rehab {nome} ({info.file_size / 1e6:.1f} MB)")
                # ground truth 2D (mocap projetado na câmera 17, 30 fps) — avalia a pose estimada
                for info in RemoteZip(f"{REHAB_ZENODO}/2d_joints.zip?download=1").infolist():
                    nome = info.filename
                    if nome.endswith("-c17-30fps.npy") and int(nome.split("/")[0].removeprefix("Ex")) in exercicios:
                        alvo = destino / "joints2d" / nome
                        if not alvo.exists():
                            alvo.parent.mkdir(parents=True, exist_ok=True)
                            with RemoteZip(f"{REHAB_ZENODO}/2d_joints.zip?download=1") as zj:
                                alvo.write_bytes(zj.read(info))
                baixar(f"{REHAB_ZENODO}/joints_names.txt?download=1", destino / "joints_names.txt")
            return
        except Exception as e:  # conexões longas com o Zenodo às vezes caem
            print(f"  rehab: tentativa {tentativa} falhou ({type(e).__name__}); retomando…")
            time.sleep(5 * tentativa)
    raise RuntimeError("não foi possível baixar os vídeos do REHAB24-6")


URFALL = "https://fenix.ur.edu.pl/~mkepski/ds/data"


def baixar_urfall(n_quedas: int = 30, n_adl: int = 40) -> None:
    destino = RAW / "urfall"
    for prefixo, n in (("fall", n_quedas), ("adl", n_adl)):
        for i in range(1, n + 1):
            nome = f"{prefixo}-{i:02d}-cam0.mp4"
            try:
                baixar(f"{URFALL}/{nome}", destino / nome, 10_000)
            except requests.HTTPError as e:
                print(f"  urfall: falhou {nome}: {e}")
    # rótulos por quadro (-1 em pé/atividade, 0 transição/queda, 1 deitado) publicados pelos autores
    for nome in ("urfall-cam0-falls.csv", "urfall-cam0-adls.csv"):
        baixar(f"{URFALL}/{nome}", destino / nome)
    print(f"  urfall: {n_quedas} quedas + {n_adl} atividades normais")


ETAPAS = {
    "vitaldb": baixar_vitaldb,
    "mimic": baixar_mimic,
    "primock": baixar_primock,
    "torgo": baixar_torgo,
    "coswara": baixar_coswara,
    "rehab": baixar_rehab,
    "urfall": baixar_urfall,
}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--only", nargs="*", choices=sorted(ETAPAS), help="baixa só estes datasets")
    p.add_argument("--rehab-exercicios", nargs="*", type=int, default=[1, 6])
    a = p.parse_args()
    for nome in a.only or list(ETAPAS):
        print(f"[{nome}]")
        if nome == "rehab":
            baixar_rehab(tuple(a.rehab_exercicios))
        else:
            ETAPAS[nome]()
    print("ok")


if __name__ == "__main__":
    main()

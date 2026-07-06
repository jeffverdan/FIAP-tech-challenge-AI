"""Teste rapido para validar que o ambiente esta configurado corretamente."""

def test_imports():
    import numpy as np
    import pandas as pd
    import matplotlib
    import seaborn
    import sklearn
    import shap
    print("Todas as libs importadas com sucesso!")

def test_versions():
    import numpy, pandas, sklearn, shap
    libs = {
        "numpy": numpy.__version__,
        "pandas": pandas.__version__,
        "scikit-learn": sklearn.__version__,
        "shap": shap.__version__,
    }
    for name, version in libs.items():
        print(f"  {name}: {version}")

if __name__ == "__main__":
    print("Validando setup...")
    test_imports()
    test_versions()
    print("Setup OK!")

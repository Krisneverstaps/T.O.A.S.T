import urllib.request
import urllib.error
import pandas as pd
from pathlib import Path
import numpy as np

GITHUB_REPO = "des-science/DES-SN5YR"
GITHUB_BRANCH = "main"

def download_file(file_path, local_dir="data"):
    """Downloads a file from GitHub if it doesn't exist."""
    local_path = Path(local_dir) / Path(file_path).name
    local_path.parent.mkdir(exist_ok=True, parents=True)

    if local_path.exists():
        return local_path.resolve()

    url = f"https://raw.githubusercontent.com/{GITHUB_REPO}/{GITHUB_BRANCH}/{file_path}"
    try:
        urllib.request.urlretrieve(url, local_path)
    except Exception:
        # Fallback to master branch
        url = f"https://raw.githubusercontent.com/{GITHUB_REPO}/master/{file_path}"
        urllib.request.urlretrieve(url, local_path)
    
    return local_path.resolve()

def load_snana_format(filepath):
    """Parses the special VARNAMES/SN: format into a Pandas DataFrame."""
    var_names = None
    data_rows = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'): continue
            if line.startswith('VARNAMES:'):
                var_names = [v for v in line.replace('VARNAMES:', '').split() if v]
                continue
            if line.startswith('SN:'):
                values = [v for v in line.replace('SN:', '').split() if v]
                if var_names and len(values) >= len(var_names):
                    data_rows.append(values[:len(var_names)])
    
    df = pd.DataFrame(data_rows, columns=var_names)

    for col in df.columns:
        if col != 'CID': # ID as a string
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
    # Fill any missing data with NaN 
    df = df.dropna(how='all') 
    return df

def load_cov_npz(filepath):
    d = np.load(filepath)
    n = int(d["nsn"][0])
    inv_cov = np.zeros((n, n))
    inv_cov[np.triu_indices(n)] = d["cov"]
    inv_cov[np.tril_indices(n, -1)] = inv_cov.T[np.tril_indices(n, -1)]
    return inv_cov
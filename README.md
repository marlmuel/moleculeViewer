# 🧬 Molecule Similarity Explorer

An interactive cheminformatics dashboard built with Streamlit, demonstrating core drug discovery concepts.

Link: https://moleculeviewer-axpjesj9hz8haujz4xsjwd.streamlit.app/

## Features

- **Tanimoto similarity heatmap** — pairwise structural comparison of compounds
- **Chemical space projection** — UMAP dimensionality reduction of fingerprint vectors
- **Lipinski Ro5 radar chart** — drug-likeness profiling per molecule
- **Custom molecule input** — add your own compounds via SMILES

## Cheminformatics concepts

| Concept | Description |
|---|---|
| ECFP4 fingerprints | Morgan fingerprints (radius=2) encoding local atomic neighbourhoods |
| Tanimoto coefficient | `bits_in_common / bits_in_either` — structural similarity metric |
| UMAP | Dimensionality reduction of fingerprint space for visual exploration |
| Lipinski Ro5 | Rule-of-5 oral bioavailability filter (MW, logP, HBD, HBA) |

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy to Streamlit Cloud

1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your repo and set `app.py` as the main file
4. Deploy

## Production upgrade

This demo uses deterministic mock fingerprints. To use real RDKit chemistry:

```python
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs

mol = Chem.MolFromSmiles(smiles)
fp  = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
```

Add `rdkit` to `requirements.txt` and replace the `mock_fingerprint()` function.

## Stack

- [Streamlit](https://streamlit.io) — UI framework
- [Plotly](https://plotly.com) — interactive visualizations
- [NumPy](https://numpy.org) — fingerprint math
- [Pandas](https://pandas.pydata.org) — data tables

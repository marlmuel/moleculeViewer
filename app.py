import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from collections import defaultdict

st.set_page_config(
    page_title="Molecule Similarity Explorer",
    page_icon="🧬",
    layout="wide"
)

# ── Data ──────────────────────────────────────────────────────────────────────

MOLECULES = [
    {"name": "Aspirin",      "smiles": "CC(=O)Oc1ccccc1C(=O)O",                                                              "mw": 180.2, "logp": 1.2,  "hbd": 1, "hba": 4,  "tpsa": 63.6,  "rotbonds": 3},
    {"name": "Ibuprofen",    "smiles": "CC(C)Cc1ccc(cc1)C(C)C(=O)O",                                                         "mw": 206.3, "logp": 3.5,  "hbd": 1, "hba": 2,  "tpsa": 37.3,  "rotbonds": 4},
    {"name": "Caffeine",     "smiles": "Cn1cnc2c1c(=O)n(c(=O)n2C)C",                                                         "mw": 194.2, "logp": -0.1, "hbd": 0, "hba": 6,  "tpsa": 58.4,  "rotbonds": 0},
    {"name": "Paracetamol",  "smiles": "CC(=O)Nc1ccc(O)cc1",                                                                  "mw": 151.2, "logp": 0.5,  "hbd": 2, "hba": 3,  "tpsa": 49.3,  "rotbonds": 2},
    {"name": "Naproxen",     "smiles": "COc1ccc2cc(ccc2c1)C(C)C(=O)O",                                                       "mw": 230.3, "logp": 3.2,  "hbd": 1, "hba": 3,  "tpsa": 46.5,  "rotbonds": 4},
    {"name": "Metformin",    "smiles": "CN(C)C(=N)NC(=N)N",                                                                   "mw": 129.2, "logp": -1.4, "hbd": 4, "hba": 5,  "tpsa": 88.6,  "rotbonds": 2},
    {"name": "Atorvastatin", "smiles": "CC(C)c1c(C(=O)Nc2ccccc2F)c(-c2ccccc2)c(-c2ccc(F)cc2)n1CCC(O)CC(O)CC(=O)O",         "mw": 558.6, "logp": 4.5,  "hbd": 4, "hba": 9,  "tpsa": 112.0, "rotbonds": 14},
    {"name": "Lidocaine",    "smiles": "CCN(CC)CC(=O)Nc1c(C)cccc1C",                                                         "mw": 234.3, "logp": 2.3,  "hbd": 1, "hba": 3,  "tpsa": 32.3,  "rotbonds": 6},
]

COLORS = [
    "#378ADD", "#1D9E75", "#D85A30", "#7F77DD",
    "#BA7517", "#D4537E", "#888780", "#5DCAA5",
]

RO5_LIMITS = {"mw": 500, "logp": 5, "hbd": 5, "hba": 10, "tpsa": 140}
RO5_LABELS = {"mw": "MW ≤500", "logp": "logP ≤5", "hbd": "HBD ≤5", "hba": "HBA ≤10", "tpsa": "TPSA ≤140"}

# ── Cheminformatics helpers (pure Python, no RDKit) ───────────────────────────

def mock_fingerprint(smiles: str) -> np.ndarray:
    """
    Deterministic pseudo-fingerprint from SMILES characters.
    In production replace with:
        from rdkit.Chem import AllChem
        mol = Chem.MolFromSmiles(smiles)
        fp  = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=512)
    """
    rng = np.random.default_rng(seed=abs(hash(smiles)) % (2**31))
    return rng.integers(0, 2, size=512).astype(float)

def tanimoto(a: np.ndarray, b: np.ndarray) -> float:
    """Tanimoto / Jaccard coefficient for binary fingerprints."""
    both  = np.sum(a * b)
    either = np.sum((a + b) > 0)
    return float(both / either) if either > 0 else 0.0

def similarity_matrix(mols: list) -> np.ndarray:
    fps = [mock_fingerprint(m["smiles"]) for m in mols]
    n   = len(fps)
    mat = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            mat[i, j] = tanimoto(fps[i], fps[j])
    return mat

def lipinski_pass(mol: dict) -> bool:
    return (mol["mw"] <= 500 and mol["logp"] <= 5
            and mol["hbd"] <= 5 and mol["hba"] <= 10)

def mock_umap(mols: list) -> list[dict]:
    """
    Deterministic 2-D projection for demo.
    In production replace with:
        from umap import UMAP
        X = np.array([fingerprint(m) for m in mols])
        coords = UMAP(n_components=2).fit_transform(X)
    """
    mat = similarity_matrix(mols)
    # simple MDS-like projection from similarity matrix
    dist = 1 - mat
    np.fill_diagonal(dist, 0)
    center = dist.mean(axis=1)
    grand  = dist.mean()
    x = center - grand
    y = np.array([dist[i, (i + 1) % len(mols)] for i in range(len(mols))]) - grand
    return [{"name": m["name"], "x": float(x[i]), "y": float(y[i])}
            for i, m in enumerate(mols)]

# ── Session state ─────────────────────────────────────────────────────────────

if "molecules" not in st.session_state:
    st.session_state.molecules = MOLECULES.copy()
if "selected" not in st.session_state:
    st.session_state.selected = 0

mols    = st.session_state.molecules
sel_idx = st.session_state.selected
sel_mol = mols[sel_idx]

# ── Header ────────────────────────────────────────────────────────────────────

st.title("🧬 Molecule Similarity Explorer")
st.caption("ECFP4 fingerprints · Tanimoto similarity · Lipinski Ro5 · Chemical space")

# ── Metrics row ───────────────────────────────────────────────────────────────

druglike_count = sum(1 for m in mols if lipinski_pass(m))
sim_mat        = similarity_matrix(mols)
n              = len(mols)
avg_sim        = (sim_mat.sum() - n) / (n * (n - 1)) if n > 1 else 0

c1, c2, c3, c4 = st.columns(4)
c1.metric("Compounds",    n)
c2.metric("Drug-like",    f"{druglike_count} / {n}")
c3.metric("Avg similarity", f"{avg_sim:.2f}")
c4.metric("Selected",     sel_mol["name"])

st.divider()

# ── Molecule selector ─────────────────────────────────────────────────────────

st.subheader("Compound library")
st.caption("Select a molecule to update the drug-likeness radar chart")

cols = st.columns(min(len(mols), 4))
for i, mol in enumerate(mols):
    col = cols[i % 4]
    with col:
        passes = lipinski_pass(mol)
        border_color = COLORS[i % len(COLORS)]
        badge  = "✅ Drug-like" if passes else "⚠️ Violates Ro5"
        if st.button(
            f"**{mol['name']}**\n\nMW {mol['mw']} · logP {mol['logp']}\n\n{badge}",
            key=f"mol_{i}",
            use_container_width=True,
            type="primary" if i == sel_idx else "secondary",
        ):
            st.session_state.selected = i
            st.rerun()

st.divider()

# ── Tabs ──────────────────────────────────────────────────────────────────────

tab1, tab2, tab3 = st.tabs(["🟩 Similarity heatmap", "🔵 Chemical space", "📊 Drug-likeness"])

# ── Tab 1 — Heatmap ───────────────────────────────────────────────────────────

with tab1:
    st.markdown("**Tanimoto similarity matrix** — computed from ECFP4-style fingerprints (radius=2)")

    labels = [m["name"] for m in mols]

    fig_heat = go.Figure(go.Heatmap(
        z=sim_mat,
        x=labels,
        y=labels,
        colorscale=[[0, "#E1F5EE"], [1, "#0F6E56"]],
        zmin=0, zmax=1,
        text=[[f"{sim_mat[i][j]:.2f}" for j in range(n)] for i in range(n)],
        texttemplate="%{text}",
        textfont={"size": 11},
        hovertemplate="%{y} vs %{x}<br>Tanimoto: %{z:.3f}<extra></extra>",
    ))

    fig_heat.update_layout(
        height=420,
        margin=dict(l=20, r=20, t=20, b=20),
        xaxis=dict(tickangle=-40),
    )

    st.plotly_chart(fig_heat, use_container_width=True)

    st.caption(
        "Tanimoto = (bits in common) / (bits in either). "
        "Score of 1.0 = identical structures. "
        "Scores > 0.4 generally indicate structural similarity in medicinal chemistry."
    )

# ── Tab 2 — Chemical space ────────────────────────────────────────────────────

with tab2:
    st.markdown("**Chemical space projection** — UMAP of fingerprint vectors (2D)")

    embed = mock_umap(mols)
    df_embed = pd.DataFrame(embed)

    fig_scatter = go.Figure()
    for i, row in df_embed.iterrows():
        mol = mols[i]
        fig_scatter.add_trace(go.Scatter(
            x=[row["x"]], y=[row["y"]],
            mode="markers+text",
            name=mol["name"],
            text=[mol["name"]],
            textposition="top center",
            marker=dict(size=16, color=COLORS[i % len(COLORS)]),
            hovertemplate=(
                f"<b>{mol['name']}</b><br>"
                f"MW: {mol['mw']} g/mol<br>"
                f"logP: {mol['logp']}<br>"
                f"Ro5: {'Pass' if lipinski_pass(mol) else 'Fail'}"
                "<extra></extra>"
            ),
        ))

    fig_scatter.update_layout(
        height=400,
        showlegend=False,
        margin=dict(l=20, r=20, t=20, b=20),
        xaxis=dict(title="UMAP 1", zeroline=False, showgrid=True, gridcolor="#eee"),
        yaxis=dict(title="UMAP 2", zeroline=False, showgrid=True, gridcolor="#eee"),
    )

    st.plotly_chart(fig_scatter, use_container_width=True)
    st.caption(
        "In production this uses real UMAP on ECFP4 fingerprint vectors. "
        "Molecules that cluster together share structural features."
    )

# ── Tab 3 — Drug-likeness radar ───────────────────────────────────────────────

with tab3:
    left, right = st.columns([1, 1])

    with left:
        st.markdown(f"**Lipinski Ro5 profile — {sel_mol['name']}**")

        categories  = ["MW", "logP", "HBD", "HBA", "TPSA"]
        keys        = ["mw", "logp", "hbd", "hba", "tpsa"]
        values      = [sel_mol[k] for k in keys]
        limits      = [RO5_LIMITS[k] for k in keys]
        normalized  = [min(v / l, 1.2) for v, l in zip(values, limits)]

        fig_radar = go.Figure()

        # Ro5 boundary
        fig_radar.add_trace(go.Scatterpolar(
            r=[1, 1, 1, 1, 1, 1],
            theta=categories + [categories[0]],
            fill="toself",
            fillcolor="rgba(0,0,0,0.04)",
            line=dict(color="#ccc", dash="dot"),
            name="Ro5 limit",
        ))

        # Molecule profile
        color = COLORS[sel_idx % len(COLORS)]
        fig_radar.add_trace(go.Scatterpolar(
            r=normalized + [normalized[0]],
            theta=categories + [categories[0]],
            fill="toself",
            fillcolor=color + "33",
            line=dict(color=color, width=2),
            name=sel_mol["name"],
        ))

        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1.2],
                                       tickvals=[0.25, 0.5, 0.75, 1.0],
                                       ticktext=["25%", "50%", "75%", "100%"])),
            showlegend=False,
            height=350,
            margin=dict(l=40, r=40, t=40, b=40),
        )

        st.plotly_chart(fig_radar, use_container_width=True)

    with right:
        st.markdown(f"**Descriptor details**")
        st.markdown("")

        ro5_rows = []
        for k in keys:
            val   = sel_mol[k]
            limit = RO5_LIMITS[k]
            label = RO5_LABELS[k]
            passes = val <= limit
            ro5_rows.append({
                "Descriptor": label,
                "Value": val,
                "Pass": "✅" if passes else "❌",
            })

        df_ro5 = pd.DataFrame(ro5_rows)
        st.dataframe(df_ro5, hide_index=True, use_container_width=True)

        passes_all = lipinski_pass(sel_mol)
        if passes_all:
            st.success(f"**{sel_mol['name']}** passes all Lipinski Ro5 rules — predicted good oral bioavailability.")
        else:
            violations = sum(1 for k in keys if sel_mol[k] > RO5_LIMITS[k])
            st.warning(f"**{sel_mol['name']}** violates {violations} Ro5 rule(s) — may have reduced oral bioavailability.")

        st.markdown("")
        st.markdown("**Rotatable bonds**")
        st.progress(min(sel_mol["rotbonds"] / 15, 1.0),
                    text=f"{sel_mol['rotbonds']} rotatable bonds (≤10 preferred)")

st.divider()

# ── Add custom molecule ───────────────────────────────────────────────────────

st.subheader("Add a molecule")

with st.form("add_molecule", clear_on_submit=True):
    col_name, col_smiles = st.columns([1, 2])
    with col_name:
        new_name = st.text_input("Name", placeholder="My molecule")
    with col_smiles:
        new_smiles = st.text_input("SMILES", placeholder="CC(=O)Oc1ccccc1C(=O)O")

    col_mw, col_logp, col_hbd, col_hba, col_tpsa = st.columns(5)
    new_mw   = col_mw.number_input("MW",   min_value=0.0, value=200.0, step=0.1)
    new_logp = col_logp.number_input("logP", min_value=-10.0, value=2.0, step=0.1)
    new_hbd  = col_hbd.number_input("HBD",  min_value=0, value=1, step=1)
    new_hba  = col_hba.number_input("HBA",  min_value=0, value=2, step=1)
    new_tpsa = col_tpsa.number_input("TPSA", min_value=0.0, value=50.0, step=0.1)
    new_rotb = st.number_input("Rotatable bonds", min_value=0, value=3, step=1)

    submitted = st.form_submit_button("➕ Add molecule", use_container_width=True)

    if submitted:
        if not new_name or not new_smiles:
            st.error("Please provide both a name and a SMILES string.")
        elif any(m["smiles"] == new_smiles for m in st.session_state.molecules):
            st.error("This SMILES is already in the library.")
        else:
            st.session_state.molecules.append({
                "name": new_name, "smiles": new_smiles,
                "mw": new_mw, "logp": new_logp,
                "hbd": new_hbd, "hba": new_hba,
                "tpsa": new_tpsa, "rotbonds": new_rotb,
            })
            st.success(f"Added **{new_name}** to the library!")
            st.rerun()

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("ℹ️ About")
    st.markdown("""
    **Molecule Similarity Explorer** demonstrates core cheminformatics concepts:

    **ECFP4 fingerprints**
    Extended connectivity fingerprints (radius=2) encode each atom's local chemical neighbourhood into a bit vector.

    **Tanimoto similarity**
    Measures structural overlap between two fingerprints:
    `score = bits_in_common / bits_in_either`

    **UMAP projection**
    Reduces high-dimensional fingerprint vectors to 2D for visual exploration of chemical space.

    **Lipinski Ro5**
    Rule-of-5 filter for oral drug-likeness:
    - MW ≤ 500 g/mol
    - logP ≤ 5
    - H-bond donors ≤ 5
    - H-bond acceptors ≤ 10
    """)

    st.divider()
    st.markdown("**Example SMILES**")
    st.code("CC(=O)Oc1ccccc1C(=O)O", language=None)
    st.caption("Aspirin")

    st.divider()
    st.markdown("**Production upgrade**")
    st.markdown("""
    Replace mock fingerprints with:
    ```python
    from rdkit.Chem import AllChem
    fp = AllChem.GetMorganFingerprintAsBitVect(
        mol, radius=2, nBits=2048
    )
    ```
    """)

    if st.button("🗑️ Reset to defaults", use_container_width=True):
        st.session_state.molecules = MOLECULES.copy()
        st.session_state.selected  = 0
        st.rerun()

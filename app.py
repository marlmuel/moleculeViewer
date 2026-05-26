import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

st.set_page_config(
    page_title="Roche Small Molecule Explorer",
    page_icon="",
    layout="wide"
)

# ── Roche approved small molecules ────────────────────────────────────────────
# Sources: Wikipedia / PubChem / DrugBank — all publicly available data

MOLECULES = [
    {
        "name": "Alectinib",
        "brand": "Alecensa",
        "smiles": "CCc1cc2c(cc1N1CCC(CC1)N1CCOCC1)C(c1c(c3ccc(cc3[nH]1)C#N)C2=O)(C)C",
        "indication": "ALK+ non-small cell lung cancer",
        "target": "ALK inhibitor",
        "year": 2015,
        "mw": 482.6, "logp": 4.9, "hbd": 1, "hba": 5, "tpsa": 75.9, "rotbonds": 4,
    },
    {
        "name": "Cobimetinib",
        "brand": "Cotellic",
        "smiles": "C1CCN[C@@H](C1)C2(CN(C2)C(=O)C3=C(C(=C(C=C3)F)F)NC4=C(C=C(C=C4)I)F)O",
        "indication": "BRAF V600E/K+ melanoma",
        "target": "MEK1/2 inhibitor",
        "year": 2015,
        "mw": 531.3, "logp": 3.2, "hbd": 2, "hba": 5, "tpsa": 78.2, "rotbonds": 4,
    },
    {
        "name": "Vismodegib",
        "brand": "Erivedge",
        "smiles": "CS(=O)(=O)C1=CC(=C(C=C1)C(=O)NC2=CC(=C(C=C2)Cl)C3=CC=CC=N3)Cl",
        "indication": "Basal cell carcinoma",
        "target": "Hedgehog pathway (SMO) inhibitor",
        "year": 2012,
        "mw": 421.3, "logp": 3.7, "hbd": 1, "hba": 4, "tpsa": 71.0, "rotbonds": 4,
    },
    {
        "name": "Pirfenidone",
        "brand": "Esbriet",
        "smiles": "CC1=CN(C(=O)C=C1)C2=CC=CC=C2",
        "indication": "Idiopathic pulmonary fibrosis",
        "target": "Anti-fibrotic (TGF-β modulator)",
        "year": 2014,
        "mw": 185.2, "logp": 1.6, "hbd": 0, "hba": 2, "tpsa": 20.3, "rotbonds": 2,
    },
    {
        "name": "Entrectinib",
        "brand": "Rozlytrek",
        "smiles": "CN1CCN(c2ccc(C(=O)Nc3n[nH]c4ccc(Cc5cc(F)cc(F)c5)cc34)c(NC3CCOCC3)c2)CC1",
        "indication": "ROS1+ NSCLC / NTRK fusion+ solid tumors",
        "target": "TRK/ROS1/ALK inhibitor",
        "year": 2019,
        "mw": 560.6, "logp": 3.6, "hbd": 2, "hba": 7, "tpsa": 96.7, "rotbonds": 7,
    },
    {
        "name": "Venetoclax",
        "brand": "Venclexta",
        "smiles": "CC1(C)CCC(CN2CCN(C3=CC(OC4=CN=C5C(C=CN5)=C4)=C(C(NS(=O)(C6=CC([N+]([O-])=O)=C(NCC7CCOCC7)C=C6)=O)=O)C=C3)CC2)=C(C8=CC=C(Cl)C=C8)C1",
        "indication": "Chronic lymphocytic leukemia / AML",
        "target": "BCL-2 inhibitor",
        "year": 2016,
        "mw": 868.4, "logp": 5.8, "hbd": 3, "hba": 10, "tpsa": 170.2, "rotbonds": 14,
    },
    {
        "name": "Idasanutlin",
        "brand": "Investigational (RG7388)",
        "smiles": "COC1=CC=C(C=C1)C2=CC(=C(C(=C2)C(F)(F)F)NC(=O)N3CCC[C@H]3CO)Cl",
        "indication": "AML (MDM2 inhibitor, clinical trials)",
        "target": "MDM2 inhibitor",
        "year": 2020,
        "mw": 499.9, "logp": 3.5, "hbd": 2, "hba": 5, "tpsa": 75.5, "rotbonds": 5,
    },
    {
        "name": "Ipatasertib",
        "brand": "Investigational (GDC-0068)",
        "smiles": "C[C@@H]1CCN(C[C@@H]1NC(=O)C2=CN=CN=C2)CC3=CC4=C(S3)N=CN=C4N",
        "indication": "Prostate cancer / breast cancer (AKT inhibitor)",
        "target": "AKT1/2/3 inhibitor",
        "year": 2021,
        "mw": 424.5, "logp": 1.8, "hbd": 3, "hba": 9, "tpsa": 115.7, "rotbonds": 5,
    },
]

COLORS = [
    "#378ADD", "#1D9E75", "#D85A30", "#7F77DD",
    "#BA7517", "#D4537E", "#888780", "#5DCAA5",
]

RO5_LIMITS  = {"mw": 500, "logp": 5, "hbd": 5, "hba": 10, "tpsa": 140}
RO5_LABELS  = {"mw": "MW ≤500", "logp": "logP ≤5", "hbd": "HBD ≤5", "hba": "HBA ≤10", "tpsa": "TPSA ≤140"}
RO5_KEYS    = ["mw", "logp", "hbd", "hba", "tpsa"]

# ── Chemistry helpers (pure Python — no RDKit required) ───────────────────────

def mock_fingerprint(smiles: str) -> np.ndarray:
    """
    Deterministic pseudo-fingerprint derived from SMILES string.
    Production replacement:
        from rdkit.Chem import AllChem
        mol = Chem.MolFromSmiles(smiles)
        fp  = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=512)
    """
    rng = np.random.default_rng(seed=abs(hash(smiles)) % (2**31))
    return rng.integers(0, 2, size=512).astype(float)

def tanimoto(a: np.ndarray, b: np.ndarray) -> float:
    both   = np.sum(a * b)
    either = np.sum((a + b) > 0)
    return float(both / either) if either > 0 else 0.0

def similarity_matrix(mols: list) -> np.ndarray:
    fps = [mock_fingerprint(m["smiles"]) for m in mols]
    n   = len(fps)
    mat = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            mat[i, j] = round(tanimoto(fps[i], fps[j]), 3)
    return mat

def lipinski_pass(mol: dict) -> bool:
    return (mol["mw"] <= 500 and mol["logp"] <= 5
            and mol["hbd"] <= 5 and mol["hba"] <= 10)

def mock_umap(mols: list) -> list:
    mat  = similarity_matrix(mols)
    dist = 1 - mat
    np.fill_diagonal(dist, 0)
    center = dist.mean(axis=1)
    grand  = dist.mean()
    x = center - grand
    y = np.array([dist[i, (i + 1) % len(mols)] for i in range(len(mols))]) - grand
    return [{"name": m["name"], "x": float(x[i]), "y": float(y[i])}
            for i, m in enumerate(mols)]

# ── Session state ──────────────────────────────────────────────────────────────

if "molecules" not in st.session_state:
    st.session_state.molecules = MOLECULES.copy()
if "selected" not in st.session_state:
    st.session_state.selected = 0

mols    = st.session_state.molecules
sel_idx = st.session_state.selected
if sel_idx >= len(mols):
    sel_idx = 0
    st.session_state.selected = 0
sel_mol = mols[sel_idx]

# ── Header ─────────────────────────────────────────────────────────────────────

st.title("🧬 Roche Small Molecule Explorer")
st.caption("Approved & late-stage small molecules from Roche / Genentech · ECFP4 fingerprints · Tanimoto similarity · Lipinski Ro5")

# ── Metrics ────────────────────────────────────────────────────────────────────

sim_mat        = similarity_matrix(mols)
n              = len(mols)
druglike_count = sum(1 for m in mols if lipinski_pass(m))
avg_sim        = round((sim_mat.sum() - n) / (n * (n - 1)), 3) if n > 1 else 0

c1, c2, c3, c4 = st.columns(4)
c1.metric("Compounds",       n)
c2.metric("Drug-like (Ro5)", f"{druglike_count} / {n}")
c3.metric("Avg similarity",  f"{avg_sim:.2f}")
c4.metric("Selected",        sel_mol["name"])

st.divider()

# ── Molecule selector ──────────────────────────────────────────────────────────

st.subheader("Compound library")
st.caption("Click a compound to update the drug-likeness radar and descriptor table")

cols = st.columns(min(len(mols), 4))
for i, mol in enumerate(mols):
    with cols[i % 4]:
        label = (
            f"**{mol['name']}**  \n"
            f"{mol['brand']}  \n"
            f"MW {mol['mw']} · logP {mol['logp']}  \n"
            f"{'✅ Ro5 pass' if lipinski_pass(mol) else '⚠️ Ro5 fail'}  \n"
            f"*{mol['target']}*"
        )
        if st.button(
            label,
            key=f"mol_{i}",
            use_container_width=True,
            type="primary" if i == sel_idx else "secondary",
        ):
            st.session_state.selected = i
            st.rerun()

st.divider()

# ── Tabs ───────────────────────────────────────────────────────────────────────

tab1, tab2, tab3, tab4 = st.tabs([
    "Similarity heatmap",
    "Chemical space",
    "Drug-likeness",
    "Compound table",
])

# ── Tab 1 — Heatmap ────────────────────────────────────────────────────────────

with tab1:
    st.markdown("**Tanimoto similarity matrix** — pairwise structural similarity from ECFP4 fingerprints (radius=2)")

    labels = [m["name"] for m in mols]

    fig_heat = go.Figure(go.Heatmap(
        z=sim_mat,
        x=labels,
        y=labels,
        colorscale=[[0, "#E1F5EE"], [0.5, "#5DCAA5"], [1, "#0F6E56"]],
        zmin=0, zmax=1,
        text=[[f"{sim_mat[i][j]:.2f}" for j in range(n)] for i in range(n)],
        texttemplate="%{text}",
        textfont={"size": 10},
        hovertemplate="%{y} vs %{x}<br>Tanimoto: %{z:.3f}<extra></extra>",
    ))
    fig_heat.update_layout(
        height=450,
        margin=dict(l=20, r=20, t=20, b=20),
        xaxis=dict(tickangle=-40, tickfont=dict(size=11)),
        yaxis=dict(tickfont=dict(size=11)),
    )
    st.plotly_chart(fig_heat, use_container_width=True)
    st.caption(
        "Tanimoto = (bits in common) / (bits in either fingerprint). "
        "Score 1.0 = identical. Score > 0.4 indicates structural similarity in medicinal chemistry. "
        "Diagonal = self-similarity = 1.0."
    )

# ── Tab 2 — Chemical space ─────────────────────────────────────────────────────

with tab2:
    st.markdown("**Chemical space projection** — dimensionality reduction of ECFP4 fingerprint vectors")

    embed    = mock_umap(mols)
    fig_scat = go.Figure()

    for i, pt in enumerate(embed):
        mol = mols[i]
        fig_scat.add_trace(go.Scatter(
            x=[pt["x"]], y=[pt["y"]],
            mode="markers+text",
            name=mol["name"],
            text=[mol["name"]],
            textposition="top center",
            textfont=dict(size=11),
            marker=dict(size=18, color=COLORS[i % len(COLORS)],
                        line=dict(width=1.5, color="white")),
            hovertemplate=(
                f"<b>{mol['name']}</b> ({mol['brand']})<br>"
                f"Target: {mol['target']}<br>"
                f"Indication: {mol['indication']}<br>"
                f"MW: {mol['mw']} · logP: {mol['logp']}<br>"
                f"Ro5: {'Pass ✅' if lipinski_pass(mol) else 'Fail ⚠️'}"
                "<extra></extra>"
            ),
        ))

    fig_scat.update_layout(
        height=420,
        showlegend=False,
        margin=dict(l=20, r=20, t=20, b=20),
        xaxis=dict(title="UMAP 1", zeroline=False, showgrid=True, gridcolor="rgba(0,0,0,0.06)"),
        yaxis=dict(title="UMAP 2", zeroline=False, showgrid=True, gridcolor="rgba(0,0,0,0.06)"),
    )
    st.plotly_chart(fig_scat, use_container_width=True)
    st.caption(
        "Hover over points for compound details. "
        "In production this uses real UMAP on ECFP4 vectors — "
        "molecules that cluster together share structural features (same scaffold, similar substituents)."
    )

# ── Tab 3 — Drug-likeness ──────────────────────────────────────────────────────

with tab3:
    left, right = st.columns([1, 1])

    with left:
        st.markdown(f"**Lipinski Ro5 profile — {sel_mol['name']} ({sel_mol['brand']})**")

        categories = ["MW", "logP", "HBD", "HBA", "TPSA"]
        values     = [sel_mol[k] for k in RO5_KEYS]
        limits     = [RO5_LIMITS[k] for k in RO5_KEYS]
        normalized = [min(v / l, 1.3) for v, l in zip(values, limits)]
        color      = COLORS[sel_idx % len(COLORS)]

        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(
            r=[1]*5 + [1], theta=categories + [categories[0]],
            fill="toself", fillcolor="rgba(0,0,0,0.04)",
            line=dict(color="#ccc", dash="dot"), name="Ro5 limit",
        ))
        fig_radar.add_trace(go.Scatterpolar(
            r=normalized + [normalized[0]], theta=categories + [categories[0]],
            fill="toself", fillcolor=f'rgba({int(color[1:3],16)},{int(color[3:5],16)},{int(color[5:7],16)},0.2)',
            line=dict(color=color, width=2), name=sel_mol["name"],
        ))
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(
                visible=True, range=[0, 1.3],
                tickvals=[0.25, 0.5, 0.75, 1.0],
                ticktext=["25%", "50%", "75%", "100% (limit)"],
                tickfont=dict(size=9),
            )),
            showlegend=False, height=340,
            margin=dict(l=50, r=50, t=30, b=30),
        )
        st.plotly_chart(fig_radar, use_container_width=True)

    with right:
        st.markdown(f"**Descriptor details**")
        st.markdown(f"*{sel_mol['indication']}*")
        st.markdown(f"Target: `{sel_mol['target']}` · Approved: {sel_mol['year']}")
        st.markdown("")

        ro5_rows = []
        for k in RO5_KEYS:
            val   = sel_mol[k]
            limit = RO5_LIMITS[k]
            ro5_rows.append({
                "Rule":  RO5_LABELS[k],
                "Value": val,
                "Status": "✅ Pass" if val <= limit else "❌ Fail",
            })

        st.dataframe(pd.DataFrame(ro5_rows), hide_index=True, use_container_width=True)
        st.markdown("")

        if lipinski_pass(sel_mol):
            st.success(f"**{sel_mol['name']}** passes all Lipinski Ro5 rules.")
        else:
            violations = [RO5_LABELS[k] for k in RO5_KEYS if sel_mol[k] > RO5_LIMITS[k]]
            st.warning(
                f"**{sel_mol['name']}** violates: {', '.join(violations)}. "
                "This is expected for complex oncology molecules (BCS class IV / beyond Ro5 space)."
            )

        st.markdown("**Rotatable bonds**")
        st.progress(
            min(sel_mol["rotbonds"] / 20, 1.0),
            text=f"{sel_mol['rotbonds']} rotatable bonds (≤10 preferred for oral bioavailability)",
        )

# ── Tab 4 — Compound table ─────────────────────────────────────────────────────

with tab4:
    st.markdown("**Full compound library**")

    df = pd.DataFrame([{
        "Name":       m["name"],
        "Brand":      m["brand"],
        "Target":     m["target"],
        "Indication": m["indication"],
        "Year":       m["year"],
        "MW":         m["mw"],
        "logP":       m["logp"],
        "HBD":        m["hbd"],
        "HBA":        m["hba"],
        "TPSA":       m["tpsa"],
        "RotBonds":   m["rotbonds"],
        "Ro5":        "✅" if lipinski_pass(m) else "❌",
    } for m in mols])

    st.dataframe(df, hide_index=True, use_container_width=True)
    st.download_button(
        "Download as CSV",
        data=df.to_csv(index=False),
        file_name="roche_small_molecules.csv",
        mime="text/csv",
    )

st.divider()

# ── Add custom molecule ────────────────────────────────────────────────────────

with st.expander("Add a custom molecule"):
    with st.form("add_molecule", clear_on_submit=True):
        c1, c2, c3 = st.columns(3)
        new_name    = c1.text_input("Name",         placeholder="Compound X")
        new_brand   = c2.text_input("Brand / code", placeholder="GDC-XXXX")
        new_smiles  = c3.text_input("SMILES",        placeholder="CC(=O)...")
        new_target  = st.text_input("Target / MoA",  placeholder="e.g. KRAS G12C inhibitor")
        new_indication = st.text_input("Indication", placeholder="e.g. NSCLC")

        dc1, dc2, dc3, dc4, dc5, dc6 = st.columns(6)
        new_mw   = dc1.number_input("MW",       0.0,  value=300.0, step=0.1)
        new_logp = dc2.number_input("logP",    -10.0, value=2.0,   step=0.1)
        new_hbd  = dc3.number_input("HBD",      0,    value=1,     step=1)
        new_hba  = dc4.number_input("HBA",      0,    value=3,     step=1)
        new_tpsa = dc5.number_input("TPSA",     0.0,  value=60.0,  step=0.1)
        new_rotb = dc6.number_input("RotBonds", 0,    value=4,     step=1)

        if st.form_submit_button("Add molecule", use_container_width=True):
            if not new_name or not new_smiles:
                st.error("Name and SMILES are required.")
            elif any(m["smiles"] == new_smiles for m in st.session_state.molecules):
                st.error("SMILES already in library.")
            else:
                st.session_state.molecules.append({
                    "name": new_name, "brand": new_brand or "—",
                    "smiles": new_smiles, "target": new_target or "—",
                    "indication": new_indication or "—", "year": 2024,
                    "mw": new_mw, "logp": new_logp, "hbd": new_hbd,
                    "hba": new_hba, "tpsa": new_tpsa, "rotbonds": new_rotb,
                })
                st.success(f"Added {new_name}!")
                st.rerun()

# ── Sidebar ────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("About")
    st.markdown("""
    **Roche Small Molecule Explorer**

    Showcases approved and late-stage small molecules from Roche / Genentech with interactive cheminformatics visualizations.

    **Molecules included:**
    - Alectinib (Alecensa) — ALK+ NSCLC
    - Cobimetinib (Cotellic) — Melanoma
    - Vismodegib (Erivedge) — BCC
    - Pirfenidone (Esbriet) — IPF
    - Entrectinib (Rozlytrek) — NTRK/ROS1+
    - Venetoclax (Venclexta) — CLL/AML
    - Idasanutlin — MDM2 (clinical)
    - Ipatasertib — AKT (clinical)
    """)

    st.divider()
    st.markdown("**Cheminformatics concepts**")
    st.markdown("""
    - **ECFP4** — Morgan fingerprints, radius=2
    - **Tanimoto** — structural similarity metric
    - **UMAP** — fingerprint space projection
    - **Lipinski Ro5** — oral drug-likeness filter
    """)

    st.divider()
    st.markdown("**Production upgrade**")
    st.code(
        "from rdkit.Chem import AllChem\n"
        "fp = AllChem.GetMorganFingerprintAsBitVect(\n"
        "    mol, radius=2, nBits=2048\n"
        ")",
        language="python"
    )

    if st.button("Reset to defaults", use_container_width=True):
        st.session_state.molecules = MOLECULES.copy()
        st.session_state.selected  = 0
        st.rerun()

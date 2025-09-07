# graph_tool.py
# -*- coding: utf-8 -*-


from __future__ import annotations
import os
import re
import pandas as pd
from typing import List, Tuple, Optional, Dict
from rapidfuzz import process, fuzz


# ---------------- helpers ----------------

def _pick_col(df: pd.DataFrame, candidates: List[str], required: bool = True) -> Optional[str]:
    """Return first existing column from candidates; fallback to the sole column if any."""
    for c in candidates:
        if c in df.columns:
            return c
    if len(df.columns) == 1:
        return df.columns[0]
    if required:
        raise ValueError(f"Cannot find any of {candidates} in columns: {list(df.columns)}")
    return None


def _infer_plasmid_type_from_name(name: str) -> Optional[str]:
    """Infer plasmid 'type' from the plasmid name string when absent."""
    if not isinstance(name, str):
        name = str(name)
    s = name.strip()
    m = re.search(r'(Inc[A-Za-z0-9\-\(\)_]+)', s)
    if m:
        return m.group(1)
    m = re.search(r'(Col[0-9A-Za-z\-\(\)_]+)', s)
    if m:
        return m.group(1)
    return None


def _soft_norm(s: str) -> str:
    """Light normalization to improve contig matching."""
    if not isinstance(s, str):
        s = str(s)
    s = s.strip().lower()
    s = re.sub(r'\s+', '', s)
    s = re.sub(r'[^a-z0-9_]+', '', s)
    return s


# --------- ARG normalization (family-level) ---------
def _normalize_gene(g: str) -> str:
    if not isinstance(g, str):
        g = str(g)
    s = g.strip()
    s = re.sub(r"\s+", "", s)  # collapse whitespace

    # common case fixes
    s = s.replace("Bla", "bla").replace("BLANDM", "blaNDM")

    # unify separators to hyphen for parsing
    s_std = re.sub(r"[ _]+", "-", s)

    families = {
        "NDM": "blaNDM",
        "KPC": "blaKPC",
        "IMP": "blaIMP",
        "VIM": "blaVIM",
        "OXA": "blaOXA",
        "TEM": "blaTEM",
        "SHV": "blaSHV",
        "CTX-M": "CTX-M",   # many tables keep CTX-M without 'bla' prefix
        "CMY": "blaCMY",
        "DHA": "blaDHA",
        "GES": "blaGES",
    }

    # patterns like: blaNDM-1, NDM-5, CTX-M-15, blaOXA-48-like
    m = re.match(r"^(?:bla)?(NDM|KPC|IMP|VIM|OXA|TEM|SHV|CTX-M|CMY|DHA|GES)(?:-[0-9A-Za-z]+)?$",
                 s_std, flags=re.IGNORECASE)
    if m:
        fam = m.group(1).upper()
        return families.get(fam, fam)

    return s


# --------- contig reconciliation via nodes_contig ---------
def _reconcile_contig_keys(
    e_df: pd.DataFrame,
    e_col: str,
    nodes_contig_raw: pd.DataFrame,
    candidate_node_cols: List[str],
    canonical_name: str = "contig"
) -> pd.DataFrame:
    """Map edge table contig IDs to a unified canonical key via nodes_contig.csv."""
    df = e_df.copy()
    df[e_col] = df[e_col].astype(str).str.strip()

    nodes = nodes_contig_raw.copy()
    for c in nodes.columns:
        nodes[c] = nodes[c].astype(str).str.strip()

    canonical_node_col = None
    for c in ["contig", "name", "label", "id", "contig_id", "node"]:
        if c in nodes.columns:
            canonical_node_col = c
            break
    if canonical_node_col is None:
        canonical_node_col = nodes.columns[0]

    def _build_mapping(nc: str):
        if nc not in nodes.columns or canonical_node_col not in nodes.columns:
            return None
        try:
            subset = nodes.drop_duplicates(subset=[nc])[[nc, canonical_node_col]]
            return subset.set_index(nc)[canonical_node_col].to_dict()
        except Exception:
            return None

    # direct mapping
    for nc in candidate_node_cols:
        mp = _build_mapping(nc)
        if not mp:
            continue
        hit = df[e_col].isin(mp).mean()
        if hit > 0.01:
            df[canonical_name] = df[e_col].map(mp)
            return df

    # upper-case mapping as fallback
    for nc in candidate_node_cols:
        if nc not in nodes.columns or canonical_node_col not in nodes.columns:
            continue
        try:
            mapping = nodes.drop_duplicates(subset=[nc])[[nc, canonical_node_col]].copy()
            mapping["__key"] = mapping[nc].str.upper()
            mp = mapping.set_index("__key")[canonical_node_col].to_dict()
            df["__key"] = df[e_col].str.upper()
            hit = df["__key"].isin(mp).mean()
            if hit > 0.01:
                df[canonical_name] = df["__key"].map(mp)
                df.drop(columns=["__key"], inplace=True)
                return df
        except Exception:
            continue

    # fallback: keep original
    df[canonical_name] = df[e_col]
    return df


def _auto_fuzzy_bridge(E_CA: pd.DataFrame, E_CP: pd.DataFrame, cutoff: int = 95) -> Tuple[pd.DataFrame, Dict[str, str]]:
    ca_list = E_CA["contig"].astype(str).tolist()
    cp_list = E_CP["contig"].astype(str).unique().tolist()

    cp_norm_map = {}
    for c in cp_list:
        cp_norm_map.setdefault(_soft_norm(c), []).append(c)

    mapping: Dict[str, str] = {}
    choices = list(cp_norm_map.keys())

    for contig in ca_list:
        key = _soft_norm(contig)
        hit = process.extractOne(key, choices, scorer=fuzz.WRatio, score_cutoff=cutoff)
        if hit:
            cand_norm = hit[0]
            original_cp = cp_norm_map[cand_norm][0]
            mapping[contig] = original_cp

    if not mapping:
        return E_CA, mapping

    E_CA_new = E_CA.copy()
    E_CA_new["contig"] = E_CA_new["contig"].map(lambda x: mapping.get(str(x), x))
    return E_CA_new, mapping


# --------- Step 0 hook: prefer bridged/aligned edge files ---------
def _override_edges_from_files(csv_dir: str,
                               E_CA: pd.DataFrame,
    def _pick(df: pd.DataFrame, cands: list[str]) -> Optional[str]:
        for c in cands:
            if c in df.columns: return c
        return None

    def _load_edge(path: Optional[str], kind: str) -> Optional[pd.DataFrame]:
        if not path or not os.path.exists(path):
            return None
        df = pd.read_csv(path)
        if kind == "CA":
            col_c = _pick(df, ["contig","contig_id","name","label","id","node","from","u"])
            col_a = _pick(df, ["arg","gene","ARG","to","v"])
            if col_c and col_a:
                return (df.rename(columns={col_c:"contig", col_a:"arg"})
                          [["contig","arg"]].dropna().drop_duplicates())
        else:
            col_c = _pick(df, ["contig","contig_id","name","label","id","node","from","u"])
            col_p = _pick(df, ["plasmid","plasmid_id","replicon","to","v"])
            if col_c and col_p:
                # FIXED: correct closing brace here
                return (df.rename(columns={col_c:"contig", col_p:"plasmid"})
                          [["contig","plasmid"]].dropna().drop_duplicates())
        return None

    ca_candidates = [
        os.environ.get("GT_E_CA"),
        os.path.join(csv_dir, "edges_contig_arg_bridged.csv"),
        os.path.join(csv_dir, "edges_contig_arg_aligned.csv"),
    ]
    cp_candidates = [
        os.environ.get("GT_E_CP"),
        os.path.join(csv_dir, "edges_contig_plasmid_aligned.csv"),
    ]

    for p in ca_candidates:
        df = _load_edge(p, "CA")
        if df is not None and not df.empty:
            E_CA = df
            break

    for p in cp_candidates:
        df = _load_edge(p, "CP")
        if df is not None and not df.empty:
            E_CP = df
            break

    return E_CA, E_CP


# ---------------- entity lexicon ----------------
class EntityLexicon:
    """Keeps vocabularies and performs fuzzy matching with RapidFuzz."""
    def __init__(self, genes, plasmids, samples) -> None:
        self.genes = sorted(set([str(x) for x in genes if pd.notna(x)]))
        self.plasmids = sorted(set([str(x) for x in plasmids if pd.notna(x)]))
        self.samples = sorted(set([str(x) for x in samples if pd.notna(x)]))

    def best(self, text: str, kind: str, cutoff: int = 82):
        text = str(text)
        pool = getattr(self, kind)
        hit = process.extractOne(text, pool, scorer=fuzz.WRatio, score_cutoff=cutoff)
        return hit[0] if hit else None


# ---------------- core tool ----------------
class GraphTool:
    def __init__(self, csv_dir: str = "/nobackup/hkkq91/ai_genomes/chatbot_project/csv") -> None:
        self.csv_dir = os.path.abspath(csv_dir)
        req = [
            "nodes_arg.csv","nodes_contig.csv","nodes_plasmid.csv","nodes_sample.csv",
            "edges_contig_arg.csv","edges_contig_plasmid.csv","edges_sample_contig.csv",
        ]
        miss = [f for f in req if not os.path.exists(os.path.join(self.csv_dir, f))]
        if miss:
            raise FileNotFoundError(f"Missing CSV files in {self.csv_dir}: {miss}")

        # nodes
        na = pd.read_csv(os.path.join(self.csv_dir, "nodes_arg.csv"))
        nc_raw = pd.read_csv(os.path.join(self.csv_dir, "nodes_contig.csv"))
        npz = pd.read_csv(os.path.join(self.csv_dir, "nodes_plasmid.csv"))
        ns = pd.read_csv(os.path.join(self.csv_dir, "nodes_sample.csv"))

        node_contig_pref = ["contig","contig_id","name","label","id","node"]

        self.col_arg = _pick_col(na, ["arg","gene","ARG","arg_name"])
        self.col_contig = _pick_col(nc_raw, node_contig_pref)
        self.col_plasmid_id = _pick_col(npz, ["plasmid","plasmid_id","name","id","replicon"])
        self.col_plasmid_type = _pick_col(npz, ["type","inc","group","inc_group"], required=False)
        self.col_sample = _pick_col(ns, ["sample","sample_id","name","id"])

        self.nodes_arg = na.rename(columns={self.col_arg:"arg"})[["arg"]].drop_duplicates()

        tmp_nc = nc_raw[[self.col_contig]].copy()
        self.nodes_contig = tmp_nc.rename(columns={self.col_contig:"contig"}).drop_duplicates()

        self.nodes_plasmid = npz.rename(columns={self.col_plasmid_id:"plasmid"})
        if self.col_plasmid_type and self.col_plasmid_type != "type":
            self.nodes_plasmid = self.nodes_plasmid.rename(columns={self.col_plasmid_type:"type"})
        if "type" not in self.nodes_plasmid.columns:
            self.nodes_plasmid["type"] = None
        if self.nodes_plasmid["type"].isna().all():
            inferred = self.nodes_plasmid["plasmid"].astype(str).map(_infer_plasmid_type_from_name)
            self.nodes_plasmid["type"] = self.nodes_plasmid["type"].fillna(inferred)
        self.nodes_plasmid = self.nodes_plasmid[["plasmid","type"]].drop_duplicates()

        self.nodes_sample = ns.rename(columns={self.col_sample:"sample"})[["sample"]].drop_duplicates()

        # edges (raw)
        e_ca_raw = pd.read_csv(os.path.join(self.csv_dir, "edges_contig_arg.csv"))
        e_cp_raw = pd.read_csv(os.path.join(self.csv_dir, "edges_contig_plasmid.csv"))
        e_sc_raw = pd.read_csv(os.path.join(self.csv_dir, "edges_sample_contig.csv"))

        col_e_ca_contig = _pick_col(e_ca_raw, node_contig_pref + ["from","u"])
        col_e_ca_arg = _pick_col(e_ca_raw, ["arg","gene","to","v"])
        col_e_cp_contig = _pick_col(e_cp_raw, node_contig_pref + ["from","u"])
        col_e_cp_plasmid = _pick_col(e_cp_raw, ["plasmid","plasmid_id","to","v","replicon"])
        col_e_sc_sample = _pick_col(e_sc_raw, ["sample","sample_id","from","u"])
        col_e_sc_contig = _pick_col(e_sc_raw, node_contig_pref + ["to","v"])

        # reconcile via nodes_contig
        e_ca = _reconcile_contig_keys(
            e_df=e_ca_raw.rename(columns={col_e_ca_arg:"arg"}),
            e_col=col_e_ca_contig, nodes_contig_raw=nc_raw,
            candidate_node_cols=node_contig_pref, canonical_name="contig",
        )
        e_cp = _reconcile_contig_keys(
            e_df=e_cp_raw.rename(columns={col_e_cp_plasmid:"plasmid"}),
            e_col=col_e_cp_contig, nodes_contig_raw=nc_raw,
            candidate_node_cols=node_contig_pref, canonical_name="contig",
        )
        e_sc = _reconcile_contig_keys(
            e_df=e_sc_raw.rename(columns={col_e_sc_sample:"sample"}),
            e_col=col_e_sc_contig, nodes_contig_raw=nc_raw,
            candidate_node_cols=node_contig_pref, canonical_name="contig",
        )

        self.E_CA = e_ca[["contig","arg"]].dropna().drop_duplicates()
        self.E_CP = e_cp[["contig","plasmid"]].dropna().drop_duplicates()
        self.E_SC = e_sc[["sample","contig"]].dropna().drop_duplicates()

        # --- Step 0: prefer bridged/aligned edges if available (or via env) ---
        self.E_CA, self.E_CP = _override_edges_from_files(self.csv_dir, self.E_CA, self.E_CP)

        # auto fuzzy bridge if still no overlap
        overlap = len(set(self.E_CA["contig"]) & set(self.E_CP["contig"]))
        if overlap == 0 and len(self.E_CA) and len(self.E_CP):
            E_CA_new, mapping = _auto_fuzzy_bridge(self.E_CA, self.E_CP, cutoff=95)
            if mapping:
                self.E_CA = E_CA_new

        # lexicon
        genes = self.nodes_arg["arg"].map(_normalize_gene).tolist()
        plasmids = self.nodes_plasmid["plasmid"].astype(str).tolist()
        samples = self.nodes_sample["sample"].astype(str).tolist()
        self.lex = EntityLexicon(genes, plasmids, samples)

    # ------------- query APIs -------------
    def plasmids_by_gene(self, gene_or_q: str):
        """Return plasmids (and contigs) that carry the given ARG."""
        gene = self.lex.best(_normalize_gene(gene_or_q), "genes", cutoff=70) or _normalize_gene(gene_or_q)
        contigs = set(self.E_CA.loc[self.E_CA["arg"] == gene, "contig"])
        df = self.E_CP[self.E_CP["contig"].isin(contigs)][["plasmid","contig"]].drop_duplicates()
        prov = [{"file":"edges_contig_arg.csv","note":f"arg={gene}"},
                {"file":"edges_contig_plasmid.csv","note":"join by contig"}]
        return df, prov

    def genes_on_plasmid(self, plasmid_or_q: str):
        """Return ARGs found on a given plasmid (via contigs)."""
        plm = self.lex.best(plasmid_or_q, "plasmids", cutoff=70) or plasmid_or_q
        contigs = set(self.E_CP.loc[self.E_CP["plasmid"] == plm, "contig"])
        df = (self.E_CA[self.E_CA["contig"].isin(contigs)][["arg","contig"]]
              .drop_duplicates().sort_values("arg"))
        prov = [{"file":"edges_contig_plasmid.csv","note":f"plasmid={plm}"},
                {"file":"edges_contig_arg.csv","note":"join by contig"}]
        return df, prov

    def samples_by_gene(self, gene_or_q: str):
        """Return samples that contain the given ARG (via contigs)."""
        gene = self.lex.best(_normalize_gene(gene_or_q), "genes", cutoff=70) or _normalize_gene(gene_or_q)
        contigs = set(self.E_CA.loc[self.E_CA["arg"] == gene, "contig"])
        df = (self.E_SC[self.E_SC["contig"].isin(contigs)][["sample","contig"]]
              .drop_duplicates().sort_values("sample"))
        prov = [{"file":"edges_contig_arg.csv","note":f"arg={gene}"},
                {"file":"edges_sample_contig.csv","note":"join by contig"}]
        return df, prov

    def cooccur_genes(self, gene_or_q: str):
        """Return ARGs that co-occur with the given ARG on the same contigs, with counts."""
        gene = self.lex.best(_normalize_gene(gene_or_q), "genes", cutoff=70) or _normalize_gene(gene_or_q)
        contigs = set(self.E_CA.loc[self.E_CA["arg"] == gene, "contig"])
        df = self.E_CA[self.E_CA["contig"].isin(contigs)]
        counts = (df.loc[df["arg"] != gene, "arg"].value_counts()
                  .reset_index().rename(columns={"index":"arg","arg":"count"}))
        prov = [{"file":"edges_contig_arg.csv","note":f"co-occur on contigs carrying {gene}"}]
        return counts, prov

    def plasmid_types_by_gene(self, gene_or_q: str):
        """Return plasmid type(s) (INC/group) carrying the given ARG, if types are available."""
        gene = self.lex.best(_normalize_gene(gene_or_q), "genes", cutoff=70) or _normalize_gene(gene_or_q)
        contigs = set(self.E_CA.loc[self.E_CA["arg"] == gene, "contig"])
        plm = self.E_CP[self.E_CP["contig"].isin(contigs)]["plasmid"]
        df = self.nodes_plasmid[self.nodes_plasmid["plasmid"].isin(plm)][["plasmid","type"]].drop_duplicates()
        df = df.sort_values(["type","plasmid"], na_position="last")
        prov = [{"file":"edges_contig_arg.csv","note":f"arg={gene}"},
                {"file":"edges_contig_plasmid.csv","note":"get plasmids"},
                {"file":"nodes_plasmid.csv","note":"type from column or inferred"}]
        return df, prov

    def genes_in_sample(self, sample_or_q: str):
        """Return all ARGs found in a given sample (via contigs)."""
        smp = self.lex.best(sample_or_q, "samples", cutoff=75) or sample_or_q
        contigs = set(self.E_SC.loc[self.E_SC["sample"] == smp, "contig"])
        df = (self.E_CA[self.E_CA["contig"].isin(contigs)][["arg","contig"]]
              .drop_duplicates().sort_values("arg"))
        prov = [{"file":"edges_sample_contig.csv","note":f"sample={smp}"},
                {"file":"edges_contig_arg.csv","note":"join by contig"}]
        return df, prov

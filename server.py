# -*- coding: utf-8 -*-


from __future__ import annotations
import os
import re
import urllib.request
from typing import Any, Dict, List, Optional, Tuple
import logging
import numpy as np

import torch
import pandas as pd
from fastapi import Body, FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response, PlainTextResponse
from starlette.staticfiles import StaticFiles

# BERT and transformers imports
try:
    import torch
    from transformers import AutoTokenizer, AutoModel, pipeline
    from sentence_transformers import SentenceTransformer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logging.warning("Transformers not available. Falling back to rule-based NLP.")

# ---------------------------------------------------------------------
# Paths and Configuration
# ---------------------------------------------------------------------
BASE_DIR = os.path.dirname(__file__)
WEB_DIR  = os.path.join(BASE_DIR, "web")
CSV_DIR  = os.path.join(BASE_DIR, "csv")

HF_CACHE = "/nobackup/hkkq91/ai_genomes/hf_cache"
if os.path.isdir(os.path.dirname(HF_CACHE)):
    os.environ.setdefault("HF_HOME", HF_CACHE)
    os.environ.setdefault("TRANSFORMERS_CACHE", HF_CACHE)

app = FastAPI(title="AMR Graph Explorer API with BERT")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

assets_dir = os.path.join(WEB_DIR, "assets")
if os.path.isdir(assets_dir):
    app.mount("/assets", StaticFiles(directory=assets_dir), name="assets")

if os.path.isdir(WEB_DIR):
    app.mount("/web", StaticFiles(directory=WEB_DIR), name="web")

# ---------------------------------------------------------------------
# BERT-based NLP Components
# ---------------------------------------------------------------------
class BERTQueryProcessor:
    """Advanced NLP processor using BERT for query understanding"""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.initialized = False
        self.sentence_model = None
        self.intent_classifier = None
        self.gene_embeddings = None
        self.gene_names = None
        
        if TRANSFORMERS_AVAILABLE:
            self._initialize_models()
    
    def _initialize_models(self):
        """Initialize BERT models for NLP processing"""
        try:
            # Sentence transformer for semantic similarity
            self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
            
            # Intent classification pipeline
            self.intent_classifier = pipeline(
                "zero-shot-classification",
                model="facebook/bart-large-mnli",
                device= -1
            )
            
            # Pre-compute gene embeddings for similarity matching
            self._build_gene_embeddings()
            
            self.initialized = True
            logging.info("BERT models initialized successfully")
            
        except Exception as e:
            logging.error(f"Failed to initialize BERT models: {e}")
            self.initialized = False
    
    def _build_gene_embeddings(self):
        """Build embeddings for known genes for similarity matching"""
        try:
            # Load gene data from CSV
            gene_df = _safe_read_csv(os.path.join(CSV_DIR, "nodes_arg.csv"))
            if not gene_df.empty and 'id' in gene_df.columns:
                self.gene_names = gene_df['id'].astype(str).tolist()
                
                # Create embeddings for all gene names
                if self.sentence_model:
                    self.gene_embeddings = self.sentence_model.encode(
                        self.gene_names, 
                        convert_to_tensor=True
                    )
                    logging.info(f"Built embeddings for {len(self.gene_names)} genes")
        except Exception as e:
            logging.error(f"Failed to build gene embeddings: {e}")
            self.gene_names = []
            self.gene_embeddings = None
    
    def classify_intent(self, query: str) -> Dict[str, Any]:
        """Classify query intent using BERT"""
        if not self.initialized or not self.intent_classifier:
            return self._fallback_intent_classification(query)
        
        try:
            # Define possible intents
            candidate_labels = [
                "find genes in plasmids",
                "find genes in samples", 
                "explain gene function",
                "predict relationships",
                "search database",
                "general question"
            ]
            
            result = self.intent_classifier(query, candidate_labels)
            
            # Map intents to action types
            intent_mapping = {
                "find genes in plasmids": {"type": "graph", "target": "plasmid"},
                "find genes in samples": {"type": "graph", "target": "sample"},
                "explain gene function": {"type": "explain", "target": None},
                "predict relationships": {"type": "predict", "target": None},
                "search database": {"type": "graph", "target": None},
                "general question": {"type": "explain", "target": None}
            }
            
            best_intent = result['labels'][0]
            confidence = result['scores'][0]
            
            classification = intent_mapping.get(best_intent, {"type": "graph", "target": None})
            classification['confidence'] = confidence
            classification['intent'] = best_intent
            
            return classification
            
        except Exception as e:
            logging.error(f"Intent classification failed: {e}")
            return self._fallback_intent_classification(query)
    
    def extract_gene_entity(self, query: str) -> Optional[str]:
        """Extract gene entities using BERT similarity matching"""
        if not self.initialized or not self.sentence_model or not self.gene_embeddings:
            return self._fallback_gene_extraction(query)
        
        try:
            # First try direct pattern matching for common gene formats
            patterns = [
                r'(bla[A-Za-z0-9-]+)',
                r'(NDM[-0-9]*)',
                r'(CTX[-A-Za-z0-9]*)', 
                r'(mcr[-0-9]*)',
                r'(tet[A-Za-z0-9-]*)',
                r'(amp[A-Za-z0-9-]*)',
                r'(oxa[-0-9]*)',
                r'(tem[-0-9]*)',
                r'(shv[-0-9]*)'
            ]
            
            for pattern in patterns:
                match = re.search(pattern, query, re.IGNORECASE)
                if match:
                    candidate = match.group(1)
                    # Verify against known genes using similarity
                    verified = self._verify_gene_similarity(candidate)
                    if verified:
                        return verified
                    return candidate
            
            # If no pattern match, try semantic similarity with all query tokens
            query_embedding = self.sentence_model.encode([query], convert_to_tensor=True)
            similarities = torch.cosine_similarity(query_embedding, self.gene_embeddings)
            
            # Get top matches above threshold
            threshold = 0.6
            top_indices = torch.where(similarities > threshold)[0]
            
            if len(top_indices) > 0:
                best_idx = torch.argmax(similarities).item()
                return self.gene_names[best_idx]
            
            return None
            
        except Exception as e:
            logging.error(f"Gene extraction failed: {e}")
            return self._fallback_gene_extraction(query)
    
    def _verify_gene_similarity(self, candidate: str) -> Optional[str]:
        """Verify gene candidate using similarity matching"""
        if not self.sentence_model or not self.gene_embeddings:
            return candidate
        
        try:
            candidate_embedding = self.sentence_model.encode([candidate], convert_to_tensor=True)
            similarities = torch.cosine_similarity(candidate_embedding, self.gene_embeddings)
            
            threshold = 0.7
            best_idx = torch.argmax(similarities).item()
            best_score = similarities[best_idx].item()
            
            if best_score > threshold:
                return self.gene_names[best_idx]
            return candidate
            
        except Exception:
            return candidate
    
    def _fallback_intent_classification(self, query: str) -> Dict[str, Any]:
        """Fallback rule-based intent classification"""
        low = query.lower()
        
        if re.search(r'\bwhat\s+is\b|\bexplain\b|\bmechanism\b|\bfunction\b', low):
            return {"type": "explain", "target": None, "confidence": 0.8}
        
        target = None
        if "plasmid" in low or "carry" in low:
            target = "plasmid"
        elif "sample" in low or "contain" in low:
            target = "sample"
        
        return {"type": "graph", "target": target, "confidence": 0.6}
    
    def _fallback_gene_extraction(self, query: str) -> Optional[str]:
        """Fallback rule-based gene extraction"""
        patterns = [
            r'(bla[A-Za-z0-9-]+)',
            r'(NDM[-0-9]*)',
            r'(CTX[-A-Za-z0-9]*)', 
            r'(mcr[-0-9]*)',
            r'(tet[A-Za-z0-9-]*)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                return match.group(1)
        
        return None

# Initialize global NLP processor
nlp_processor = BERTQueryProcessor()

# ---------------------------------------------------------------------
# Vendor proxy & cache (unchanged)
# ---------------------------------------------------------------------
VENDOR_CACHE_DIR = os.path.join(WEB_DIR, "vendor_cache")
os.makedirs(VENDOR_CACHE_DIR, exist_ok=True)

_VENDOR_SOURCES = {
    "marked.min.js": [
        "https://cdn.jsdelivr.net/npm/marked/marked.min.js",
        "https://unpkg.com/marked@latest/marked.min.js",
        "https://cdnjs.cloudflare.com/ajax/libs/marked/12.0.2/marked.min.js",
    ],
    "cytoscape.min.js": [
        "https://unpkg.com/cytoscape@3.26.0/dist/cytoscape.min.js",
        "https://cdn.jsdelivr.net/npm/cytoscape@3.26.0/dist/cytoscape.min.js",
        "https://cdnjs.cloudflare.com/ajax/libs/cytoscape/3.26.0/cytoscape.min.js",
    ],
}

_SHIMS = {
    "marked.min.js": "window.marked={parse:(s)=>String(s||'')};",
    "cytoscape.min.js": (
        "window.cytoscape=function(o){if(o&&o.container){o.container.textContent="
        "'Graph visualization area';}return{destroy(){},add(){},elements(){return[]}}};"
    ),
}

def _fetch_and_cache(name: str) -> Optional[bytes]:
    for url in _VENDOR_SOURCES.get(name, []):
        try:
            with urllib.request.urlopen(url, timeout=12) as resp:
                data = resp.read()
                if data:
                    with open(os.path.join(VENDOR_CACHE_DIR, name), "wb") as f:
                        f.write(data)
                    return data
        except Exception:
            continue
    return None

@app.get("/vendor/{name}")
def vendor(name: str):
    if name not in _VENDOR_SOURCES:
        return Response(status_code=404)
    cache_path = os.path.join(VENDOR_CACHE_DIR, name)
    if os.path.exists(cache_path):
        return FileResponse(cache_path, media_type="application/javascript")
    data = _fetch_and_cache(name)
    if data:
        return Response(content=data, media_type="application/javascript")
    return PlainTextResponse(_SHIMS[name], media_type="application/javascript")

@app.get("/favicon.ico")
def favicon():
    ico = os.path.join(WEB_DIR, "favicon.ico")
    return FileResponse(ico) if os.path.exists(ico) else Response(status_code=204)

@app.get("/.well-known/appspecific/{path:path}")
def _silence_devtools_probe(path: str):
    return Response(status_code=204)

# ---------------------------------------------------------------------
# CSV cache helpers (unchanged)
# ---------------------------------------------------------------------
_DF_CACHE: Dict[str, pd.DataFrame] = {}

def _safe_read_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame()
    try:
        return pd.read_csv(path, dtype=str, keep_default_na=False)
    except Exception:
        return pd.read_csv(path)

def _df(name: str) -> pd.DataFrame:
    if name not in _DF_CACHE:
        _DF_CACHE[name] = _safe_read_csv(os.path.join(CSV_DIR, name))
    return _DF_CACHE[name]

# ---------------------------------------------------------------------
# Utilities (unchanged)
# ---------------------------------------------------------------------
def base_contig(x: str) -> str:
    s = str(x or "").strip()
    s = re.sub(r"^.*[|:]", "", s)
    m = re.match(r"(contig_\d+)", s)
    return m.group(1) if m else s

def gene_variants(q: str) -> List[str]:
    q = (q or "").strip()
    if not q:
        return []
    token = q
    m = re.search(r"(bla[A-Za-z0-9-]+|NDM[-0-9]*|CTX[-A-Za-z0-9]*|mcr[-0-9]*|tet[A-Za-z0-9-]*)", q, flags=re.I)
    if m:
        token = m.group(0)
    cands, seen = [], set()
    def add(s: str):
        s = (s or "").strip()
        if s and s not in seen:
            seen.add(s); cands.append(s)
    add(token)
    if "-" in token:
        add(token.split("-", 1)[0])
    if token.lower().startswith("bla"):
        core = token[3:]
        add(core)
        if "-" in core:
            add(core.split("-", 1)[0])
    return cands or [q]

# ---------------------------------------------------------------------
# Enhanced NLQ parser using BERT
# ---------------------------------------------------------------------
def parse_query_with_bert(q: str) -> Dict[str, Any]:
    """Enhanced query parsing using BERT models"""
    text = (q or "").strip()
    text = text.replace("â€œ", '"').replace("â€", '"').replace("â€˜", "'").replace("â€™", "'")
    text = re.sub(r'^\s*["\']+|["\']+\s*$', "", text)
    
    if not text:
        return {"type": "explain", "gene": None, "target": None}
    
    # Use BERT for intent classification
    intent_result = nlp_processor.classify_intent(text)
    
    # Extract gene entity using BERT
    gene = nlp_processor.extract_gene_entity(text)
    
    # If no gene found with BERT, try fallback CSV matching
    if not gene:
        try:
            ids = _df("nodes_arg.csv")["id"].astype(str)
            for p in gene_variants(text) + [text]:
                hit = ids.str.contains(re.escape(p), case=False, na=False)
                if hit.any():
                    gene = ids[hit].iloc[0]
                    break
        except Exception:
            pass
    
    return {
        "type": intent_result["type"],
        "gene": gene,
        "target": intent_result["target"],
        "confidence": intent_result.get("confidence", 0.5),
        "intent": intent_result.get("intent", "unknown")
    }

# Fallback to original parser if BERT fails
def parse_query(q: str) -> Dict[str, Any]:
    """Parse query with BERT enhancement and fallback"""
    try:
        if TRANSFORMERS_AVAILABLE and nlp_processor.initialized:
            return parse_query_with_bert(q)
    except Exception as e:
        logging.error(f"BERT parsing failed: {e}")
    
    # Fallback to original rule-based parsing
    text = (q or "").strip()
    text = text.replace("â€œ", '"').replace("â€", '"').replace("â€˜", "'").replace("â€™", "'")
    text = re.sub(r'^\s*["\']+|["\']+\s*$', "", text)
    low = text.lower()
    
    if re.search(r"\bwhat\s+is\b|\bexplain\b|\bmechanism\b", low):
        return {"type": "explain", "gene": None, "target": None}
    
    target = None
    if "plasmid" in low or "carry" in low:
        target = "plasmid"
    elif "sample" in low or "contain" in low:
        target = "sample"
    
    gene = None
    m = re.search(r"(bla[A-Za-z0-9-]+|NDM[-0-9]*|CTX[-A-Za-z0-9]*|mcr[-0-9]*|tet[A-Za-z0-9-]*)", text, flags=re.I)
    if m:
        gene = m.group(0)
    else:
        try:
            ids = _df("nodes_arg.csv")["id"].astype(str)
            for p in gene_variants(text) + [text]:
                hit = ids.str.contains(re.escape(p), case=False, na=False)
                if hit.any():
                    gene = ids[hit].iloc[0]
                    break
        except Exception:
            pass
    
    return {"type": "graph", "gene": gene, "target": target}

# ---------------------------------------------------------------------
# Rest of the code remains the same...
# (CSV fallback, GraphTool, HTML rewrite, wiretap injection, web routes, etc.)
# ---------------------------------------------------------------------

# ---------------------------------------------------------------------
# CSV fallback for /ask (unchanged from original)
# ---------------------------------------------------------------------
def _detect_direction(df: pd.DataFrame, nodes_a: set, nodes_b: set) -> Tuple[str, str]:
    if df.empty:
        return "from", "to"
    fa = df["from"].isin(nodes_a).sum()
    ta = df["to"].isin(nodes_a).sum()
    fb = df["from"].isin(nodes_b).sum()
    tb = df["to"].isin(nodes_b).sum()
    return ("from", "to") if (fa + tb) >= (ta + fb) else ("to", "from")

def _detect_sc_columns(df: pd.DataFrame) -> Tuple[str, str]:
    if df.empty:
        return "from", "to"
    scores = []
    for col in ("from", "to"):
        s = df[col].astype(str)
        hits = s.str.contains(r"(?:^|[|:])contig[_0-9]", case=False, na=False).sum()
        scores.append((hits, col))
    scores.sort(reverse=True)
    contig_col = scores[0][1]
    sample_col = "to" if contig_col == "from" else "from"
    return contig_col, sample_col

def csv_fallback_query(gene_text: str, limit: int = 300):
    N_ARG = _df("nodes_arg.csv")
    N_CONTIG = _df("nodes_contig.csv")
    N_PLASMID = _df("nodes_plasmid.csv")
    N_SAMPLE = _df("nodes_sample.csv")
    E_CA = _df("edges_contig_arg.csv")
    E_CP = _df("edges_contig_plasmid.csv")
    E_SC = _df("edges_sample_contig.csv")

    arg_ids = []
    if not N_ARG.empty:
        ids = N_ARG["id"].astype(str)
        exact = ids[ids.str.lower().eq(gene_text.strip().lower())]
        if len(exact):
            arg_ids = exact.unique().tolist()
        else:
            for v in gene_variants(gene_text) + [gene_text]:
                hit = ids[ids.str.contains(re.escape(v), case=False, na=False)]
                if len(hit):
                    arg_ids = hit.unique().tolist(); break

    if not arg_ids:
        return [], {"nodes": [], "edges": []}, ""

    ca_contig_col, ca_arg_col = _detect_direction(
        E_CA,
        set(N_CONTIG.get("id", pd.Series([], dtype=str)).astype(str)),
        set(N_ARG.get("id", pd.Series([], dtype=str)).astype(str)),
    )
    ECA_hit = E_CA[E_CA[ca_arg_col].astype(str).isin(arg_ids)].copy()
    if ECA_hit.empty:
        return [], {"nodes": [], "edges": []}, ""

    ECA_hit["contig_orig"] = ECA_hit[ca_contig_col].astype(str)
    ECA_hit["contig_base"] = ECA_hit["contig_orig"].map(base_contig)

    base2orig = (
        ECA_hit.groupby("contig_base")["contig_orig"]
        .apply(lambda s: sorted(s.unique()))
        .to_dict()
    )

    # contig -> plasmid
    E_CP2 = pd.DataFrame(columns=["contig_orig", "plasmid"])
    E_CP_tmp = _df("edges_contig_plasmid.csv")
    if not E_CP_tmp.empty:
        cp_c, cp_p = _detect_direction(
            E_CP_tmp,
            set(N_CONTIG["id"].astype(str)),
            set(N_PLASMID["id"].astype(str)) if not N_PLASMID.empty else set()
        )
        tmp = E_CP_tmp.copy()
        tmp["contig_base"] = tmp[cp_c].astype(str).map(base_contig)
        tmp = tmp[tmp["contig_base"].isin(base2orig)]
        rows = []
        for _, r in tmp.iterrows():
            for co in base2orig[r["contig_base"]]:
                rows.append({"contig_orig": co, "plasmid": r[cp_p]})
        if rows:
            E_CP2 = pd.DataFrame(rows)

    # contig -> sample
    E_SC2 = pd.DataFrame(columns=["contig_orig", "sample"])
    E_SC_tmp = _df("edges_sample_contig.csv")
    if not E_SC_tmp.empty:
        sc_c, sc_s = _detect_sc_columns(E_SC_tmp)
        tmp = E_SC_tmp.copy()
        tmp["contig_base"] = tmp[sc_c].astype(str).map(base_contig)
        tmp = tmp[tmp["contig_base"].isin(base2orig)]
        rows = []
        for _, r in tmp.iterrows():
            for co in base2orig[r["contig_base"]]:
                rows.append({"contig_orig": co, "sample": r[sc_s]})
        if rows:
            E_SC2 = pd.DataFrame(rows)

    shell = pd.DataFrame({"Gene": [arg_ids[0]]})
    df = ECA_hit[["contig_orig", "contig_base"]].drop_duplicates().merge(shell, how="cross")
    df = df.merge(E_CP2, on="contig_orig", how="left") if not E_CP2.empty else df.assign(plasmid="")
    df = df.merge(E_SC2, on="contig_orig", how="left")
    df = (
        df.rename(columns={"contig_orig": "contig"})[["Gene", "plasmid", "contig", "sample"]]
        .drop_duplicates()
        .head(limit)
    )

    nodes, edges, seen = [], [], set()
    def add_node(x, t):
        if x is None or str(x) == "": return
        s = str(x)
        if s in seen: return
        seen.add(s)
        nodes.append({"data": {"id": s, "label": s, "type": t}})

    for gid in arg_ids: add_node(gid, "gene")
    for c in df["contig"].dropna().unique(): add_node(c, "contig")
    for p in df["plasmid"].dropna().unique():
        if p: add_node(p, "plasmid")
    for s in df["sample"].dropna().unique(): add_node(s, "sample")

    for _, r in ECA_hit.iterrows():
        edges.append({"data": {"id": f"ca:{r['contig_orig']}->{r[ca_arg_col]}",
                               "source": r["contig_orig"], "target": str(r[ca_arg_col]), "type": "contig-arg"}})
    if not E_CP2.empty:
        for _, r in E_CP2.dropna().iterrows():
            edges.append({"data": {"id": f"cp:{r['contig_orig']}->{r['plasmid']}",
                                   "source": r["contig_orig"], "target": str(r["plasmid"]), "type": "contig-plasmid"}})
    for _, r in E_SC2.dropna().iterrows():
        edges.append({"data": {"id": f"cs:{r['contig_orig']}->{r['sample']}",
                               "source": r["contig_orig"], "target": str(r["sample"]), "type": "contig-sample"}})

    csv_text = "Gene,Plasmid,Contig,Sample\n" + "\n".join(
        f"{a},{b},{c},{d}" for a, b, c, d in df.itertuples(index=False)
    )
    return df.to_dict(orient="records"), {"nodes": nodes, "edges": edges}, csv_text

# ---------------------------------------------------------------------
# Optional GraphTool (unchanged from original)
# ---------------------------------------------------------------------
GT = None
try:
    from graph_tool import GraphTool as _GraphTool  # type: ignore
    GT = _GraphTool(CSV_DIR)
except Exception:
    GT = None

def _rows_to_graph(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not rows:
        return {"nodes": [], "edges": []}
    norm = []
    for r in rows:
        rr = {str(k).lower(): v for k, v in r.items()}
        if "arg" in rr and "gene" not in rr: rr["gene"] = rr["arg"]
        norm.append(rr)
    cols = set().union(*[set(r.keys()) for r in norm])
    has = {c: (c in cols) for c in ["gene", "plasmid", "contig", "sample"]}
    nodes, edges, seen = [], [], set()

    def add_node(x, t):
        if x is None or str(x) == "": return
        s = str(x)
        if s in seen: return
        seen.add(s)
        nodes.append({"data": {"id": s, "label": s, "type": t}})

    def add_edge(s, t, tp):
        if not s or not t: return
        edges.append({"data": {"id": f"{tp}:{s}->{t}", "source": str(s), "target": str(t), "type": tp}})

    for r in norm:
        if has["gene"]:    add_node(r.get("gene"), "gene")
        if has["contig"]:  add_node(r.get("contig"), "contig")
        if has["plasmid"]: add_node(r.get("plasmid"), "plasmid")
        if has["sample"]:  add_node(r.get("sample"), "sample")
        if has["contig"] and has["gene"]:    add_edge(r.get("contig"), r.get("gene"), "contig-arg")
        if has["contig"] and has["plasmid"]: add_edge(r.get("contig"), r.get("plasmid"), "contig-plasmid")
        if has["contig"] and has["sample"]:  add_edge(r.get("contig"), r.get("sample"), "contig-sample")
    return {"nodes": nodes, "edges": edges}

def _to_csv_text(rows: List[Dict[str, Any]]) -> str:
    if not rows: return ""
    headers = []
    for c in ["Gene", "Plasmid", "Contig", "Sample"]:
        if any(c.lower() == k.lower() for k in rows[0].keys()):
            headers.append(c)
    if not headers:
        headers = list(rows[0].keys())
    lines = [",".join(headers)]
    for r in rows:
        vals = []
        for h in headers:
            v = next((r[k] for k in r.keys() if k.lower() == h.lower()), "")
            vals.append("" if v is None else str(v).replace('"', '""'))
        lines.append(",".join(vals))
    return "\n".join(lines)

def try_gt(gene: str, target: Optional[str]):
    if GT is None or not gene:
        return [], {}, ""
    rows = None
    try:
        if target == "plasmid" and hasattr(GT, "plasmids_by_gene"):
            rows = GT.plasmids_by_gene(gene)
        elif target == "sample" and hasattr(GT, "samples_by_gene"):
            rows = GT.samples_by_gene(gene)
        else:
            if hasattr(GT, "samples_by_gene"):
                rows = GT.samples_by_gene(gene)
            elif hasattr(GT, "plasmids_by_gene"):
                rows = GT.plasmids_by_gene(gene)
    except Exception:
        rows = None

    if rows is None:
        return [], {}, ""
    if isinstance(rows, pd.DataFrame):
        rows = rows.to_dict(orient="records")
    elif isinstance(rows, dict):
        rows = [rows]
    elif isinstance(rows, tuple) and rows:
        r0 = rows[0]
        if isinstance(r0, pd.DataFrame):
            rows = r0.to_dict(orient="records")
        elif isinstance(r0, dict):
            rows = [r0]
        else:
            rows = list(r0)
    elif not isinstance(rows, List):
        return [], {}, ""

    if rows:
        graph = _rows_to_graph(rows)
        csv_text = _to_csv_text(rows)
        if not graph["edges"]:
            rows2, graph2, csv2 = csv_fallback_query(gene)
            if rows2:
                return rows2, graph2, csv2
        return rows, graph, csv_text
    return [], {}, ""

# ---------------------------------------------------------------------
# HTML rewrite + wiretap injection (unchanged from original)
# ---------------------------------------------------------------------
def _rewrite_vendor_scripts(html_text: str) -> str:
    if not html_text:
        return html_text
    html_text = re.sub(
        r'<script\s+src=["\']https?://(?:cdn\.jsdelivr\.net/npm/marked/marked\.min\.js|unpkg\.com/marked@[^/]+/marked\.min\.js|cdnjs\.cloudflare\.com/.+?/marked\.min\.js)["\']\s*>\s*</script>',
        '<script src="/vendor/marked.min.js"></script>',
        html_text, flags=re.IGNORECASE,
    )
    html_text = re.sub(
        r'<script\s+src=["\']https?://(?:unpkg\.com/cytoscape@[^/]+/dist/cytoscape\.min\.js|cdn\.jsdelivr\.net/npm/cytoscape@[^/]+/dist/cytoscape\.min\.js|cdnjs\.cloudflare\.com/.+?/cytoscape\.min\.js)["\']\s*>\s*</script>',
        '<script src="/vendor/cytoscape.min.js"></script>',
        html_text, flags=re.IGNORECASE,
    )
    return html_text

def _inject_wiretap(html_text: str) -> str:
    wiretap = r"""
<script>
(function(){
  // log front-end errors so我们能看到是哪里报错
  window.addEventListener('error', function(e){
    try{ console.error('[frontend error]', e.message || e.error || e); }catch(_){}
  });
  window.addEventListener('unhandledrejection', function(e){
    try{ console.error('[frontend promise]', e.reason || e); }catch(_){}
  });

  function bind(){
    try{
      // Run
      var rq = document.getElementById('runQueryBtn');
      if (rq && !rq.__wiredByServer) {
        rq.addEventListener('click', function(){
          if (window.runQuery) { window.runQuery(); }
        });
        rq.__wiredByServer = true;
      }
      // Predict
      var pr = document.getElementById('predictBtn');
      if (pr && !pr.__wiredByServer) {
        pr.addEventListener('click', function(){
          if (window.runPredict) { window.runPredict(); }
        });
        pr.__wiredByServer = true;
      }
      
      var tt = document.getElementById('themeToggle');
      if (tt && !tt.__wiredByServer) {
        tt.addEventListener('click', function(){
          var root = document.documentElement;
          var cur = root.getAttribute('data-theme') || 'auto';
          var next = (cur === 'dark') ? 'light' : 'dark';
          root.setAttribute('data-theme', next);
        });
        tt.__wiredByServer = true;
      }
      // Split resizer 兜底：提供最简左右拖拽
      var split = document.getElementById('splitGrid');
      var res = document.getElementById('splitResizer');
      if (split && res && !res.__wiredByServer) {
        res.addEventListener('mousedown', function(e){
          var startX = e.clientX;
          var left = parseFloat(getComputedStyle(split).getPropertyValue('--left')) || 55;
          function move(ev){
            var rect = split.getBoundingClientRect();
            var pct = left + (ev.clientX - startX) / rect.width * 100;
            pct = Math.max(30, Math.min(70, pct));
            split.style.setProperty('--left', pct + '%');
            split.style.setProperty('--right', (100-pct) + '%');
          }
          function up(){
            window.removeEventListener('mousemove', move);
            window.removeEventListener('mouseup', up);
          }
          window.addEventListener('mousemove', move);
          window.addEventListener('mouseup', up);
        });
        res.__wiredByServer = true;
      }
      console.log('[wiretap] front-end is alive.');
    }catch(err){
      try{ console.error('[wiretap bind error]', err); }catch(_){}
    }
  }
  if (document.readyState === 'complete' || document.readyState === 'interactive') bind();
  else document.addEventListener('DOMContentLoaded', bind);
})();
</script>
"""
    if "</body>" in html_text:
        return html_text.replace("</body>", wiretap + "\n</body>")
    return html_text + wiretap

@app.post("/ask")
def ask(payload: Dict[str, Any] = Body(None)):
    q = ""
    if payload and isinstance(payload, dict):
        q = payload.get("q") or payload.get("query") or payload.get("question") or ""
    
    # Use enhanced BERT-based parsing with error handling
    try:
        parsed = parse_query(q)
    except Exception as e:
        logging.error(f"Query parsing failed: {e}")
        # Emergency fallback to simple parsing
        parsed = {"type": "graph", "gene": None, "target": None}

    if parsed["type"] == "explain":
        term = (q or "").strip()
        if re.search(r"(bla[A-Za-z0-9-]+|NDM[-0-9]*|CTX[-A-Za-z0-9]*)", term, flags=re.I):
            family = re.search(r"(bla[A-Za-z0-9-]+|NDM|CTX)", term, flags=re.I).group(0)
            
            # Enhanced explanation using BERT context
            confidence_info = ""
            if parsed.get("confidence", 0) > 0.8:
                confidence_info = f" (High confidence: {parsed.get('confidence', 0):.2f})"
            
            md = (
                f"**{family}**{confidence_info}\n\n"
                f"{family} is a β-lactamase family associated with antimicrobial resistance. "
                f"Variants (e.g., *{family}-1*, *{family}-5*) are often carried on plasmids and can spread via horizontal gene transfer. "
                f"Ask a graph question such as *Which plasmids carry {family}?* to see relationships in the dataset."
            )
            return JSONResponse({"mode": "text", "answer_md": md})
        
        return JSONResponse({"mode": "text", "answer_md": "**Explanation placeholder.**"})

    gene, target = parsed.get("gene"), parsed.get("target")
    if not gene:
        suggestion = "Try **Which plasmids carry blaCTX-M-15?** or **Which samples contain tetA?**"
        if parsed.get("confidence", 0) < 0.5:
            suggestion += " (Low confidence in query understanding - try being more specific)"
        
        return JSONResponse({
            "mode": "text",
            "answer_md": f"I couldn't find a gene name. {suggestion}"
        })

    # Rest of the function remains the same as original
    rows, graph, csv_text = try_gt(gene, target)
    if rows:
        return JSONResponse({"mode":"graph","graph":graph,"csv_text":csv_text})

    rows, graph, csv_text = csv_fallback_query(gene)
    if rows:
        return JSONResponse({"mode":"graph","graph":graph,"csv_text":csv_text})

# ---------------------------------------------------------------------
# Web routes
# ---------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
def index():
    html_path = os.path.join(WEB_DIR, "index.html")
    if not os.path.exists(html_path):
        return HTMLResponse("<h1>AMR Graph Explorer</h1><p>index.html not found.</p>")
    try:
        with open(html_path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
        text = _rewrite_vendor_scripts(text)
        text = _inject_wiretap(text)  
        return HTMLResponse(text)
    except Exception:
        return FileResponse(html_path, media_type="text/html; charset=utf-8")

@app.get("/healthz")
def healthz():
    return {"ok": True, "bert_available": nlp_processor.initialized if nlp_processor else False}

    return JSONResponse({"mode":"text","answer_md":f"No results for **{gene}** in this dataset."})

# ---------------------------------------------------------------------
# /predict routes (unchanged from original)
# ---------------------------------------------------------------------
def _overlay_pair_nodes_edges(pairs: List[Tuple[str, str]], left_type: str, right_type: str) -> Dict[str, Any]:
    nodes, edges, seen = [], [], set()
    def add_node(x, t):
        s = str(x)
        if not s or s in seen: return
        seen.add(s)
        nodes.append({"data": {"id": s, "label": s, "type": t}})
    for a, b in pairs:
        add_node(a, left_type)
        add_node(b, right_type)
        edges.append({"data": {"id": f"pred:{a}->{b}", "source": str(a), "target": str(b), "type": "predicted"}})
    return {"nodes": nodes, "edges": edges}

def _predict_fallback(task: str, gene: Optional[str], contig: Optional[str], topk: int) -> Tuple[Dict[str, Any], List[Dict[str, Any]], str]:
    msg = "fallback from CSV"
    rows: List[Dict[str, Any]] = []

    if task == "link":
        recs, _graph, _csv = csv_fallback_query(gene or "")
        pairs = []
        for r in recs:
            c = r.get("contig"); p = r.get("plasmid")
            if c and p: pairs.append((c, p))
        pairs = pairs[:topk] if pairs else []
        overlay = _overlay_pair_nodes_edges(pairs, "contig", "plasmid")
        rows = [{"Gene": gene, "Contig": a, "Plasmid": b, "Score": 0.0, "Prob": 0.5} for a, b in pairs]
        return overlay, rows, msg

    cont = base_contig(contig or "")

    if task == "arg":
        E_CA = _df("edges_contig_arg.csv")
        if not E_CA.empty:
            ca_c = "from" if E_CA["from"].astype(str).str.contains(r"(?:^|[|:])contig_", case=False, na=False).sum() >= \
                               E_CA["to"].astype(str).str.contains(r"(?:^|[|:])contig_", case=False, na=False).sum() else "to"
            ca_a = "to" if ca_c == "from" else "from"
            E_CA["contig_base"] = E_CA[ca_c].astype(str).map(base_contig)
            sub = E_CA[E_CA["contig_base"].eq(cont)]
            pairs = list({(str(cont), str(a)) for a in sub[ca_a].astype(str).tolist()})[:topk]
            overlay = _overlay_pair_nodes_edges(pairs, "contig", "gene")
            rows = [{"Contig": a, "ARG": b, "Score": 0.0, "Prob": 0.5} for a, b in pairs]
            return overlay, rows, msg

    if task == "plasmid":
        E_CP = _df("edges_contig_plasmid.csv")
        if not E_CP.empty:
            cp_c = "from" if E_CP["from"].astype(str).str.contains(r"(?:^|[|:])contig_", case=False, na=False).sum() >= \
                               E_CP["to"].astype(str).str.contains(r"(?:^|[|:])contig_", case=False, na=False).sum() else "to"
            cp_p = "to" if cp_c == "from" else "from"
            E_CP["contig_base"] = E_CP[cp_c].astype(str).map(base_contig)
            sub = E_CP[E_CP["contig_base"].eq(cont)]
            pairs = list({(str(cont), str(p)) for p in sub[cp_p].astype(str).tolist()})[:topk]
            overlay = _overlay_pair_nodes_edges(pairs, "contig", "plasmid")
            rows = [{"Contig": a, "Plasmid": b, "Score": 0.0, "Prob": 0.5} for a, b in pairs]
            return overlay, rows, msg

    if task in ("mge", "host"):
        edges_name = f"edges_contig_{task}.csv"
        df = _df(edges_name)
        if not df.empty:
            df["contig_base"] = df["from"].astype(str).map(base_contig)
            sub = df[df["contig_base"].eq(cont)]
            right_type = "mge" if task == "mge" else "host"
            pairs = list({(str(cont), str(t)) for t in sub["to"].astype(str).tolist()})[:topk]
            overlay = _overlay_pair_nodes_edges(pairs, "contig", right_type)
            rows = [{"Contig": a, right_type.capitalize(): b, "Score": 0.0, "Prob": 0.5} for a, b in pairs]
            return overlay, rows, msg

    return {"nodes": [], "edges": []}, [], "no candidates from CSV"

def _predict_core(task: str, gene: Optional[str], contig: Optional[str], topk: int = 10) -> JSONResponse:
    if task == "link" and gene:
        try:
            from cnn.predict_cnn_seq import predict as cnn_predict  # type: ignore
            rows = cnn_predict(gene, topk=topk) or []
            pairs = []
            for r in rows:
                c = r.get("Contig"); p = r.get("Plasmid")
                if c and p: pairs.append((str(c), str(p)))
            overlay = _overlay_pair_nodes_edges(pairs, "contig", "plasmid")
            return JSONResponse({"ok": True, "overlay": overlay, "rows": rows, "msg": "cnn"})
        except Exception as e:
            overlay, rows, msg = _predict_fallback(task, gene, contig, topk)
            return JSONResponse({"ok": True, "overlay": overlay, "rows": rows, "msg": f"fallback ({e})"})

    overlay, rows, msg = _predict_fallback(task, gene, contig, topk)
    return JSONResponse({"ok": True, "overlay": overlay, "rows": rows, "msg": msg})

@app.post("/predict")
def predict_post(payload: Dict[str, Any] = Body(None)):
    task = "link"
    gene = None
    contig = None
    topk = 10
    if payload and isinstance(payload, dict):
        task = str(payload.get("task") or "link").lower()
        gene = payload.get("gene")
        contig = payload.get("contig")
        try: topk = int(payload.get("topk", 10))
        except Exception: topk = 10
    if task == "link" and not gene:
        return JSONResponse({"ok": False, "msg": "gene required for task=link", "overlay": {"nodes": [], "edges": []}})
    if task != "link" and not contig:
        return JSONResponse({"ok": False, "msg": "contig required for this task", "overlay": {"nodes": [], "edges": []}})
    return _predict_core(task, gene, contig, topk)

@app.get("/predict")
def predict_get(
    task: str = Query("link", description="predict task: link/arg/plasmid/mge/host"),
    gene: Optional[str] = Query(None, description="gene for task=link"),
    contig: Optional[str] = Query(None, description="contig for arg/plasmid/mge/host"),
    topk: int = Query(10, ge=1, le=100, description="Top-K predictions"),
):
    if task == "link" and not gene:
        return JSONResponse({"ok": False, "msg": "gene required for task=link", "overlay": {"nodes": [], "edges": []}})
    if task != "link" and not contig:
        return JSONResponse({"ok": False, "msg": "contig required for this task", "overlay": {"nodes": [], "edges": []}})
    return _predict_core(task.lower(), gene, contig, topk)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=7860, reload=False)
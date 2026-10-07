# ==============================================================================
# Copyright (c) 2026 INFIDOS LLP. All Rights Reserved.
# Developer: BALAVIGNESH M
# Company: INFIDOS LLP
# System: Mayon-Cortex Cognitive Architecture
#
# DUAL LICENSED: GNU Affero General Public License v3.0 (AGPL-3.0)
# or Commercial License.
#
# Open-source use is governed by the AGPL-3.0 license. For proprietary
# or commercial applications seeking exemption from copyleft requirements,
# a commercial license must be obtained from INFIDOS LLP.
# Contact: balavignesh@infidos.com | https://infidos.com
# ==============================================================================

"""
Mayon Knowledge Loader
======================
Auto-populates the Mayon from REAL knowledge sources across multiple sectors:
- Python official docs & Codebases (Code sector)
- Medical / Clinical guidelines & diagnostic relationships (Medical sector)
- Legal regulations & compliance standards (Legal sector)
- Finance & economic ontology (Finance sector)
- Multimodal Image / Vision inputs (Visual Cortex via CLIP embeddings)

Uses real embedding models (sentence-transformers / CLIP) for concept vectors.
"""

import ast
import importlib
import inspect
import json
import os
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from mayon_cortex.core.graph import MayonGraph, GraphLevel


class EmbeddingModel:
    """
    Multimodal Semantic Embedding Model.
    - Text: sentence-transformers / TF-IDF
    - Vision / Image: CLIP (OpenAI CLIP ViT-B/32) projected to concept_dim
    """

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", dim: int = 384):
        self.target_dim = dim
        self.model = None
        self.tfidf = None
        self._method = None

        # Vision module (CLIP)
        self.clip_model = None
        self.clip_processor = None
        self._vision_initialized = False

        # Try sentence-transformers first
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(model_name)
            self._native_dim = self.model.get_sentence_embedding_dimension()
            self._method = "sentence-transformers"
            print(f"  ✓ Using sentence-transformers ({model_name}, dim={self._native_dim}→{dim})")
        except Exception:
            pass

        # Try transformers (AutoTokenizer + AutoModel)
        if self._method is None:
            try:
                from transformers import AutoTokenizer, AutoModel
                hf_name = model_name if "/" in model_name else f"sentence-transformers/{model_name}"
                self.hf_tokenizer = AutoTokenizer.from_pretrained(hf_name)
                self.hf_model = AutoModel.from_pretrained(hf_name)
                self.hf_model.eval()
                self._method = "transformers"
                print(f"  ✓ Using HuggingFace transformers ({hf_name}, dim={dim})")
            except Exception:
                pass

        # Fallback: TF-IDF + SVD
        if self._method is None:
            try:
                from sklearn.feature_extraction.text import TfidfVectorizer
                from sklearn.decomposition import TruncatedSVD
                self._tfidf_vectorizer = TfidfVectorizer(max_features=5000)
                self._svd = TruncatedSVD(n_components=dim)
                self._method = "tfidf"
                self._fitted = False
                print(f"  ✓ Using TF-IDF + SVD (dim={dim})")
            except Exception:
                pass

        # Fallback: High-precision semantic hash
        if self._method is None:
            self._method = "hash"
            print(f"  ✓ Using semantic hash embeddings (dim={dim})")

    def _init_vision(self):
        """Lazy load CLIP for vision understanding."""
        if self._vision_initialized:
            return
        try:
            from transformers import CLIPProcessor, CLIPModel
            import torch
            clip_name = "openai/clip-vit-base-patch32"
            self.clip_processor = CLIPProcessor.from_pretrained(clip_name)
            self.clip_model = CLIPModel.from_pretrained(clip_name)
            self.clip_model.eval()
            self._vision_initialized = True
            print(f"  ✓ Loaded CLIP Vision Model ({clip_name})")
        except Exception as e:
            self._vision_initialized = False

    def encode(self, texts: List[str]) -> np.ndarray:
        """
        Encode text into concept vectors (normalized, dim=target_dim).
        """
        if self._method == "sentence-transformers":
            return self._encode_st(texts)
        elif self._method == "transformers":
            return self._encode_hf(texts)
        elif self._method == "tfidf":
            return self._encode_tfidf(texts)
        else:
            return self._encode_hash(texts)

    def encode_single(self, text: str) -> np.ndarray:
        """Encode a single text with instant dictionary cache."""
        if not hasattr(self, "_cache"):
            self._cache = {}
        if text in self._cache:
            return self._cache[text]
        vec = self.encode([text])[0]
        if len(self._cache) < 20000:
            self._cache[text] = vec
        return vec

    def _encode_hf(self, texts: List[str]) -> np.ndarray:
        """Encode using HuggingFace Transformers AutoModel with Mean Pooling."""
        import torch
        encoded = self.hf_tokenizer(
            texts, padding=True, truncation=True, max_length=128, return_tensors="pt"
        )
        with torch.no_grad():
            out = self.hf_model(**encoded)
            token_embeddings = out[0]
            input_mask_expanded = encoded['attention_mask'].unsqueeze(-1).expand(token_embeddings.size()).float()
            sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
            sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            pooled = sum_embeddings / sum_mask
            pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
            return pooled.cpu().numpy().astype(np.float32)

    def _encode_st(self, texts: List[str]) -> np.ndarray:
        """Encode with sentence-transformers and project to target dim."""
        embeddings = self.model.encode(texts, show_progress_bar=False)
        if embeddings.shape[1] != self.target_dim:
            if not hasattr(self, '_proj_matrix'):
                rng = np.random.RandomState(42)
                self._proj_matrix = rng.randn(
                    embeddings.shape[1], self.target_dim
                ).astype(np.float32)
                norms = np.linalg.norm(self._proj_matrix, axis=0, keepdims=True)
                self._proj_matrix /= norms
            embeddings = embeddings @ self._proj_matrix
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        return (embeddings / norms).astype(np.float32)

    def _encode_tfidf(self, texts: List[str]) -> np.ndarray:
        """Encode with TF-IDF + SVD."""
        if not self._fitted:
            tfidf_matrix = self._tfidf_vectorizer.fit_transform(texts)
            if tfidf_matrix.shape[0] >= self.target_dim:
                self._svd.fit(tfidf_matrix)
            self._fitted = True

        tfidf_matrix = self._tfidf_vectorizer.transform(texts)
        if hasattr(self._svd, 'components_'):
            embeddings = self._svd.transform(tfidf_matrix)
        else:
            embeddings = tfidf_matrix.toarray()[:, :self.target_dim]
            if embeddings.shape[1] < self.target_dim:
                pad = np.zeros((embeddings.shape[0], self.target_dim - embeddings.shape[1]))
                embeddings = np.hstack([embeddings, pad])

        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        return (embeddings / norms).astype(np.float32)

    def _encode_hash(self, texts: List[str]) -> np.ndarray:
        """Deterministic subword n-gram semantic hash with stopword filtering."""
        stopwords = {
            "what", "is", "a", "an", "the", "did", "does", "do", "how", "why", "who", "whom",
            "which", "where", "when", "in", "on", "at", "to", "for", "of", "with", "by", "from",
            "about", "into", "through", "during", "before", "after", "above", "below", "up", "down",
            "it", "its", "they", "them", "their", "this", "that", "these", "those", "be", "been",
            "being", "have", "has", "had", "can", "could", "shall", "should", "will", "would",
            "may", "might", "must", "and", "or", "but", "if", "because", "as", "until", "while",
            "tell", "me", "show", "give", "explain", "describe", "find",
        }
        embeddings = np.zeros((len(texts), self.target_dim), dtype=np.float32)
        import zlib
        for i, text in enumerate(texts):
            raw_words = re.findall(r'[a-zA-Z0-9]+', text.lower())
            meaningful_words = [w for w in raw_words if w not in stopwords and len(w) > 1]
            if not meaningful_words:
                meaningful_words = raw_words or ["concept"]

            for word in meaningful_words:
                # Whole word hash
                h_w = zlib.crc32(word.encode('utf-8'))
                idx_w = h_w % self.target_dim
                sign_w = 1.0 if (h_w & 1) else -1.0
                embeddings[i, idx_w] += 2.0 * sign_w

                # Subword character 3-grams
                for k in range(len(word) - 2):
                    ngram = word[k:k+3]
                    h_ng = zlib.crc32(ngram.encode('utf-8'))
                    idx_ng = h_ng % self.target_dim
                    sign_ng = 1.0 if (h_ng & 1) else -1.0
                    embeddings[i, idx_ng] += 0.5 * sign_ng

            norm = np.linalg.norm(embeddings[i])
            if norm > 0:
                embeddings[i] /= norm
        return embeddings


class PythonDocsLoader:
    """
    Loads Python knowledge from actual Python modules.
    Extracts docstrings, function signatures, class hierarchies,
    and module-level documentation.
    """

    # Modules to extract knowledge from
    CORE_MODULES = [
        "builtins", "str", "list", "dict", "set", "tuple",
        "os", "os.path", "sys", "json", "re", "math",
        "collections", "itertools", "functools", "pathlib",
        "datetime", "typing", "io", "copy", "operator",
        "string", "textwrap", "struct", "hashlib",
        "urllib", "http", "logging", "argparse",
        "unittest", "dataclasses", "enum", "abc",
    ]

    def __init__(self, embedding_model: EmbeddingModel):
        self.embedder = embedding_model

    def extract_all(self) -> List[Dict]:
        """Extract knowledge from all core Python modules."""
        all_facts = []

        # 1. Built-in functions and types
        all_facts.extend(self._extract_builtins())

        # 2. Standard library modules
        for module_name in self.CORE_MODULES:
            try:
                facts = self._extract_module(module_name)
                all_facts.extend(facts)
            except Exception:
                continue  # Skip modules that can't be imported

        # 3. Common patterns and idioms
        all_facts.extend(self._python_idioms())

        # 4. Error handling patterns
        all_facts.extend(self._error_patterns())

        for f in all_facts:
            f["sector"] = "code"

        return all_facts

    def _extract_builtins(self) -> List[Dict]:
        """Extract all built-in functions and their docs."""
        facts = []
        for name in dir(__builtins__) if isinstance(__builtins__, dict) else dir(__builtins__):
            try:
                obj = getattr(__builtins__, name) if hasattr(__builtins__, name) else eval(name)
                if callable(obj) and not name.startswith('_'):
                    doc = inspect.getdoc(obj) or ""
                    sig = ""
                    try:
                        sig = str(inspect.signature(obj))
                    except (ValueError, TypeError):
                        pass

                    if doc:
                        # Take first 2 sentences
                        short_doc = '. '.join(doc.split('. ')[:2]).strip()
                        if len(short_doc) > 200:
                            short_doc = short_doc[:200] + "..."

                        facts.append({
                            "name": name,
                            "text": f"{name}{sig}: {short_doc}",
                            "level": GraphLevel.FACT,
                            "category": "builtin",
                            "provenance": "python_builtins",
                        })
            except Exception:
                continue
        return facts

    def _extract_module(self, module_name: str) -> List[Dict]:
        """Extract functions, classes, and docs from a module."""
        facts = []
        try:
            module = importlib.import_module(module_name)
        except ImportError:
            return facts

        # Module-level doc
        if module.__doc__:
            short_doc = '. '.join(module.__doc__.split('. ')[:2]).strip()
            if short_doc and len(short_doc) > 10:
                facts.append({
                    "name": module_name,
                    "text": f"{module_name}: {short_doc[:200]}",
                    "level": GraphLevel.CONCEPT,
                    "category": "module",
                    "provenance": f"python_stdlib:{module_name}",
                })

        # Functions and classes in the module
        for name, obj in inspect.getmembers(module):
            if name.startswith('_'):
                continue

            try:
                if inspect.isfunction(obj) or inspect.isbuiltin(obj):
                    doc = inspect.getdoc(obj) or ""
                    if doc:
                        short_doc = '. '.join(doc.split('. ')[:2]).strip()[:200]
                        sig = ""
                        try:
                            sig = str(inspect.signature(obj))
                        except (ValueError, TypeError):
                            pass
                        facts.append({
                            "name": f"{module_name}.{name}",
                            "text": f"{module_name}.{name}{sig}: {short_doc}",
                            "level": GraphLevel.FACT,
                            "category": "function",
                            "provenance": f"python_stdlib:{module_name}",
                        })

                elif inspect.isclass(obj):
                    doc = inspect.getdoc(obj) or ""
                    if doc:
                        short_doc = '. '.join(doc.split('. ')[:2]).strip()[:200]
                        facts.append({
                            "name": f"{module_name}.{name}",
                            "text": f"{module_name}.{name}: {short_doc}",
                            "level": GraphLevel.CONCEPT,
                            "category": "class",
                            "provenance": f"python_stdlib:{module_name}",
                        })

                        # Extract class methods
                        for method_name, method in inspect.getmembers(obj):
                            if method_name.startswith('__') and method_name.endswith('__'):
                                continue
                            if method_name.startswith('_'):
                                continue
                            method_doc = inspect.getdoc(method) or ""
                            if method_doc:
                                short_method_doc = method_doc.split('\n')[0][:150]
                                facts.append({
                                    "name": f"{module_name}.{name}.{method_name}",
                                    "text": f"{module_name}.{name}.{method_name}: {short_method_doc}",
                                    "level": GraphLevel.FACT,
                                    "category": "method",
                                    "provenance": f"python_stdlib:{module_name}.{name}",
                                })
            except Exception:
                continue

        return facts

    def _python_idioms(self) -> List[Dict]:
        """Common Python patterns and idioms."""
        idioms = [
            ("list comprehension", "Create list with [expr for x in iterable if cond]. Faster than for+append. Example: squares = [x**2 for x in range(10)]"),
            ("dict comprehension", "Create dict with {k: v for k, v in iterable}. Example: {x: x**2 for x in range(5)}"),
            ("set comprehension", "Create set with {expr for x in iterable}. Example: {x % 3 for x in range(10)}"),
            ("generator expression", "Lazy version of list comp: (expr for x in iterable). Memory efficient for large data."),
            ("unpacking", "a, b, c = [1, 2, 3]. Also: a, *rest = [1,2,3,4] gives rest=[2,3,4]."),
            ("walrus operator", "(Python 3.8+) := assigns in expression. while chunk := f.read(8192): process(chunk)"),
            ("f-string", "f'Hello {name}' for string formatting. Can include expressions: f'{2+2=}' → '2+2=4'"),
            ("context manager", "with open(f) as handle: ... Auto-closes file. Custom: __enter__/__exit__ or @contextmanager"),
            ("decorator pattern", "@functools.wraps(func) to preserve metadata. @lru_cache for memoization."),
            ("ternary expression", "value = x if condition else y. Concise conditional assignment."),
            ("defaultdict usage", "from collections import defaultdict; d = defaultdict(list); d['key'].append(val)"),
            ("Counter usage", "from collections import Counter; c = Counter(iterable); c.most_common(5)"),
            ("enumerate usage", "for i, item in enumerate(lst, start=0): Use instead of range(len(lst))"),
            ("zip usage", "for a, b in zip(list1, list2): Parallel iteration. zip_longest for unequal lengths."),
            ("try/except/else/finally", "try: risky. except Error: handle. else: no error. finally: always runs."),
            ("EAFP vs LBYL", "EAFP: try/except (Pythonic). LBYL: if check then act. Prefer EAFP in Python."),
            ("slots", "__slots__ = ['x', 'y'] prevents __dict__, saves memory. Use in classes with many instances."),
            ("property decorator", "@property for computed attributes. @x.setter for validation on assignment."),
            ("classmethod vs staticmethod", "@classmethod gets cls, used for alt constructors. @staticmethod gets nothing, utility."),
            ("abstract base class", "from abc import ABC, abstractmethod. Forces subclasses to implement methods."),
            ("dataclass", "@dataclass auto-generates __init__, __repr__, __eq__. field(default_factory=list) for mutables."),
            ("type hints", "def f(x: int) -> str: Use Optional[X] for X|None. Use Union[X,Y] for either."),
            ("match statement", "(3.10+) match val: case Pattern: ... Structural pattern matching with guards."),
            ("async/await", "async def f(): await coroutine. Use asyncio.run(main()) to start event loop."),
            ("pathlib", "from pathlib import Path. p = Path('dir') / 'file'. p.read_text(), p.exists(), p.glob('*.py')"),
        ]
        return [{
            "name": name,
            "text": f"{name}: {desc}",
            "level": GraphLevel.FACT,
            "category": "idiom",
            "provenance": "python_idioms",
        } for name, desc in idioms]

    def _error_patterns(self) -> List[Dict]:
        """Common error patterns and how to fix them."""
        patterns = [
            ("fix IndexError", "IndexError: list index out of range. Fix: check len(lst) > idx, or use try/except, or lst[idx] if idx < len(lst) else default"),
            ("fix KeyError", "KeyError: key not in dict. Fix: use dict.get(key, default), or 'if key in d:', or try/except KeyError"),
            ("fix TypeError", "TypeError: unsupported operand type. Fix: check types with isinstance(), convert types, or add type hints"),
            ("fix ValueError", "ValueError: invalid value for type. Fix: validate input before conversion. E.g., if s.isdigit(): int(s)"),
            ("fix AttributeError", "AttributeError: object has no attribute. Fix: check with hasattr(obj, 'attr'), or use getattr(obj, 'attr', default)"),
            ("fix ImportError", "ImportError: No module named X. Fix: pip install X, check spelling, check PYTHONPATH, check virtual env"),
            ("fix FileNotFoundError", "FileNotFoundError: file not found. Fix: check path with os.path.exists(), use absolute paths, check working directory"),
            ("fix RecursionError", "RecursionError: maximum recursion depth exceeded. Fix: add base case, use iteration instead, or sys.setrecursionlimit()"),
            ("fix NameError", "NameError: name not defined. Fix: check spelling, check scope (local vs global), check if imported"),
            ("fix IndentationError", "IndentationError: unexpected indent. Fix: use consistent spaces (4), don't mix tabs and spaces"),
            ("fix SyntaxError", "SyntaxError: invalid syntax. Fix: check missing colons, parentheses, quotes. Check Python version compatibility."),
            ("fix UnicodeDecodeError", "UnicodeDecodeError: can't decode byte. Fix: open(f, encoding='utf-8'), or try encoding='latin-1'"),
            ("fix MemoryError", "MemoryError: not enough memory. Fix: use generators instead of lists, process in chunks, reduce data size"),
            ("fix PermissionError", "PermissionError: access denied. Fix: check file permissions, run as admin, or use a different path"),
        ]
        return [{
            "name": name,
            "text": desc,
            "level": GraphLevel.FACT,
            "category": "error_fix",
            "provenance": "python_error_patterns",
        } for name, desc in patterns]


class CodebaseLoader:
    """
    Load knowledge from an actual codebase (your code or any Python project).
    Parses .py files and extracts:
    - Function/class definitions with docstrings
    - Import relationships
    - Call graphs
    """

    def __init__(self, embedding_model: EmbeddingModel):
        self.embedder = embedding_model

    def extract_from_directory(
        self, directory: str, max_files: int = 500
    ) -> List[Dict]:
        """Parse all .py files in a directory and extract knowledge."""
        facts = []
        py_files = list(Path(directory).rglob("*.py"))[:max_files]

        for filepath in py_files:
            try:
                source = filepath.read_text(encoding="utf-8", errors="ignore")
                file_facts = self._parse_python_file(str(filepath), source)
                facts.extend(file_facts)
            except Exception:
                continue

        return facts

    def _parse_python_file(self, filepath: str, source: str) -> List[Dict]:
        """Parse a single Python file using AST."""
        facts = []
        try:
            tree = ast.parse(source)
        except SyntaxError:
            return facts

        filename = Path(filepath).stem

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) or isinstance(node, ast.AsyncFunctionDef):
                doc = ast.get_docstring(node) or ""
                args = [a.arg for a in node.args.args if a.arg != 'self']
                sig = f"({', '.join(args)})"
                if doc:
                    short_doc = doc.split('\n')[0][:200]
                else:
                    short_doc = f"function in {filename}"

                facts.append({
                    "name": f"{filename}.{node.name}",
                    "text": f"{filename}.{node.name}{sig}: {short_doc}",
                    "level": GraphLevel.FACT,
                    "category": "user_function",
                    "provenance": f"codebase:{filepath}",
                })

            elif isinstance(node, ast.ClassDef):
                doc = ast.get_docstring(node) or ""
                bases = [
                    getattr(b, 'id', getattr(b, 'attr', '?'))
                    for b in node.bases
                ]
                if doc:
                    short_doc = doc.split('\n')[0][:200]
                else:
                    short_doc = f"class in {filename}"

                facts.append({
                    "name": f"{filename}.{node.name}",
                    "text": f"{filename}.{node.name}({', '.join(bases)}): {short_doc}",
                    "level": GraphLevel.CONCEPT,
                    "category": "user_class",
                    "provenance": f"codebase:{filepath}",
                })

        return facts


class MultiSectorKnowledgeLoader:
    """
    Loads baseline foundational knowledge for diverse sectors:
    - Medical & Healthcare (conditions, diagnoses, treatments)
    - Legal & Compliance (regulations, GDPR, contracts)
    - Finance & Economics (monetary policy, banking, instruments)
    - Science & Logic (laws, principles, theorems)
    """

    def __init__(self, embedding_model: Optional[EmbeddingModel] = None):
        self.embedder = embedding_model

    def get_all_sector_knowledge(self) -> List[Dict]:
        """Load comprehensive knowledge across all 6 sectors from SECTOR_KNOWLEDGE."""
        facts = []
        try:
            # Try standalone data module, fallback to mayon-net bundle
            try:
                from mayon_net.data.dataset_generator import SECTOR_KNOWLEDGE
            except ImportError:
                raise ImportError("SECTOR_KNOWLEDGE not available in standalone mayon-graph")
            for sector, rel_dict in SECTOR_KNOWLEDGE.items():
                for relation, pairs in rel_dict.items():
                    for src, tgt in pairs:
                        facts.append({
                            "name": f"{src}",
                            "text": f"{src}: {relation} {tgt}",
                            "level": GraphLevel.FACT if relation in ("treats", "diagnoses", "calls", "returns") else GraphLevel.CONCEPT,
                            "sector": sector,
                            "relation": relation,
                            "target": tgt,
                            "provenance": f"sector_kb:{sector}",
                        })
        except Exception:
            pass

        # Add explicit medical and legal/finance foundational facts
        facts.extend(self.get_medical_knowledge())
        facts.extend(self.get_legal_finance_knowledge())
        return facts

    def get_medical_knowledge(self) -> List[Dict]:
        facts = [
            ("type 2 diabetes", "Type 2 diabetes is a chronic condition affecting blood glucose regulation and insulin sensitivity.", GraphLevel.CONCEPT, "medical", "clinical_guidelines"),
            ("metformin", "Metformin is a first-line biguanide medication that treats type 2 diabetes by lowering hepatic glucose production.", GraphLevel.FACT, "medical", "pharmacopeia"),
            ("glaucoma", "Glaucoma is an eye disease caused by elevated intraocular pressure damaging the optic nerve.", GraphLevel.CONCEPT, "medical", "ophthalmology"),
            ("hypertension", "Hypertension is persistent high arterial blood pressure exceeding 130/80 mmHg.", GraphLevel.CONCEPT, "medical", "cardiology"),
            ("lisinopril", "Lisinopril is an ACE inhibitor medication that treats hypertension and congestive heart failure.", GraphLevel.FACT, "medical", "pharmacopeia"),
            ("myocardial infarction", "Myocardial infarction (heart attack) occurs when blood flow decreases or stops to a part of the heart muscle.", GraphLevel.FACT, "medical", "cardiology"),
        ]
        return [{
            "name": name,
            "text": f"{name}: {desc}",
            "level": level,
            "sector": sector,
            "provenance": prov,
        } for name, desc, level, sector, prov in facts]

    def get_legal_finance_knowledge(self) -> List[Dict]:
        facts = [
            ("GDPR Article 6", "GDPR Article 6 regulates lawfulness of personal data processing under EU privacy law.", GraphLevel.FACT, "legal", "gdpr_statute"),
            ("breach of contract", "Breach of contract is a legal cause of action where a binding agreement or bargained-for exchange is not honored.", GraphLevel.CONCEPT, "legal", "contract_law"),
            ("indemnity clause", "An indemnity clause holds one party harmless for loss or damage occurring from specified liabilities.", GraphLevel.FACT, "legal", "commercial_law"),
            ("central bank interest rate", "Central bank interest rate decisions regulate commercial lending rates and monetary money supply.", GraphLevel.CONCEPT, "finance", "monetary_economics"),
            ("inflation rate", "Inflation is the rate at which the general level of prices for goods and services rises over time.", GraphLevel.CONCEPT, "finance", "macroeconomics"),
            ("liquidity ratio", "Liquidity ratio measures a debtor's ability to pay off current debt obligations without raising external capital.", GraphLevel.FACT, "finance", "banking_standards"),
        ]
        return [{
            "name": name,
            "text": f"{name}: {desc}",
            "level": level,
            "sector": sector,
            "provenance": prov,
        } for name, desc, level, sector, prov in facts]


def populate_mayon_from_knowledge(
    mayon: MayonGraph,
    facts: List[Dict],
    embedding_model: EmbeddingModel,
    batch_size: int = 64,
    verbose: bool = True,
) -> Dict:
    """
    Populate a Mayon graph with real knowledge across sectors using real embeddings.
    """
    if verbose:
        print(f"\n  Embedding {len(facts)} knowledge entries...")

    texts = [f["text"] for f in facts]
    all_vectors = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        vectors = embedding_model.encode(batch)
        all_vectors.append(vectors)

    all_vectors = np.vstack(all_vectors)

    levels = [f.get("level", GraphLevel.FACT) for f in facts]
    sectors = [f.get("sector", "general") for f in facts]
    node_types = [f.get("node_type", "text") for f in facts]
    provenances = [f.get("provenance", "unknown") for f in facts]
    metadata_list = [{"name": f["name"]} for f in facts]

    nodes = mayon.add_nodes_bulk(
        vectors=all_vectors,
        source_texts=texts,
        levels=levels,
        sectors=sectors,
        node_types=node_types,
        provenances=provenances,
        metadata_list=metadata_list,
    )
    node_ids = {node.metadata["name"]: node.id for node in nodes}

    # Auto-link explicit and hierarchical relations
    edges_to_add = []
    for fact in facts:
        rel = fact.get("relation")
        tgt = fact.get("target")
        src_name = fact["name"]
        if rel and tgt and src_name in node_ids and tgt in node_ids:
            edges_to_add.append({
                "source_id": node_ids[src_name],
                "target_id": node_ids[tgt],
                "relation_type": rel,
                "confidence": 0.95,
                "sector": fact.get("sector", "general"),
                "provenance": fact.get("provenance", "explicit_kb"),
            })

    for name, node_id in node_ids.items():
        # Hierarchy link (e.g. A.B -> part-of -> A)
        parts = name.split('.')
        if len(parts) >= 2:
            parent_name = '.'.join(parts[:-1])
            if parent_name in node_ids:
                edges_to_add.append({
                    "source_id": node_id,
                    "target_id": node_ids[parent_name],
                    "relation_type": "part-of",
                    "confidence": 0.90,
                    "sector": mayon.nodes[node_id].sector,
                    "provenance": "auto_hierarchy",
                })

    created_edges = mayon.add_edges_bulk(edges_to_add)
    relations_added = len(created_edges)

    stats = {
        "nodes_added": len(facts),
        "relations_added": relations_added,
        "total_nodes": mayon.num_nodes,
        "total_edges": mayon.num_edges,
    }

    if verbose:
        print(f"  ✓ Added {stats['nodes_added']} nodes, {stats['relations_added']} auto-relations")
        print(f"  ✓ Mayon total: {stats['total_nodes']} nodes, {stats['total_edges']} edges")

    return stats


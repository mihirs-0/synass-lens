"""
Controlled entity-disambiguation dataset for Experiment 6.

Generates sentences of the form:

    The [ROLE] who [CLAUSE] [ACTION].

mapped to the standard ``<BOS> B <SEP> z <SEP> A <EOS>`` format where

* B = ROLE  (shared category, analogous to the base string)
* z = CLAUSE (distinguishing descriptor, the selector)
* A = ACTION (unique fact to predict)

The dataset is "linguistically structured" but character-tokenised using
the existing CharTokenizer, making it fully compatible with the training
and probing pipeline.  Each role, clause, and action is encoded as a
short fixed-length string drawn from a curated vocabulary of natural-
language-like fragments (e.g. "quanta", "orbits", "wrote2") to keep the
synthetic flavour while adding semantic structure.
"""

import random
from typing import Dict, List, Tuple

from .dataset import MappingData

# -----------------------------------------------------------------------
# Curated word banks — each entry is max 6 chars so it fits b_length/
# a_length/z_length constraints of the existing pipeline.
# -----------------------------------------------------------------------

ROLES = [
    "physic", "chemst", "author", "paint_", "musici", "biolog",
    "mathem", "astron", "philos", "econom", "engine", "archit",
    "sculpt", "poetr_", "histor", "geogra", "psycho", "sociol",
    "pharma", "lingui", "neuros", "geneti", "ecolog", "paleob",
    "forens", "roboti", "crypto", "optic_", "acoust", "thermo",
    "plasma", "geolog", "meteor", "oceanr", "volcan", "seismo",
    "glacie", "mycolo", "entomo", "herpto", "ornith", "ichthy",
    "mammol", "dendro", "bryolo", "lichem", "mycorh", "algolo",
    "virolo", "parasi",
]  # 50 roles

# Clauses: distinguishing descriptors (z). We generate a pool and
# draw K unique clauses per role.
CLAUSE_POOL = [
    "studie", "explor", "invent", "discov", "analyz", "measur",
    "predic", "mappin", "modele", "proves", "derive", "reform",
    "simula", "decryp", "photon", "nuclei", "waves_", "fields",
    "signal", "quanta", "orbits", "genome", "neuron", "fossil",
    "clmate", "crysta", "volcan", "magnet", "plasma", "syntax",
    "morpho", "phoner", "semant", "pragmt", "etymol", "dialct",
    "corpus", "prosod", "lexicn", "cogntv",
    "galxyz", "starsq", "comets", "planit", "nebula", "pulsar",
    "quasar", "redsft", "dakmtr", "expnsn",
    "enzymz", "protns", "lipids", "rna_st", "dna_rp", "mitosi",
    "meioss", "osmsis", "diffus", "cataly",
    "toplgy", "algebr", "calcls", "probab", "combnt", "grapht",
    "numbrt", "logicx", "geomtr", "statis",
    "surger", "dosage", "trials", "vaccne", "immnty", "therpy",
    "diagns", "progns", "syndro", "pathol",
]

# Actions: unique facts (A).  Same pool idea — draw K unique per role.
ACTION_POOL = [
    "wrote2", "publis", "proved", "founds", "coinds", "formul",
    "patent", "discvr", "lectrd", "toured", "awardx", "exhbtd",
    "compsd", "soloed", "debutd", "perfmd", "editds", "debatd",
    "filmds", "teachs", "curads", "archvd", "transl", "revwed",
    "refutd", "synthd", "clonzd", "seqund", "imagds", "calibr",
    "mapppd", "chartd", "survyd", "drilld", "sampleq", "tagggd",
    "ringds", "bandds", "nestds", "trackg",
    "wrote3", "publs2", "provd2", "fond2_", "coind2", "forml2",
    "patnt2", "dscv2_", "lectr2", "tour2_", "awrd2_", "exhbt2",
    "compd2", "solo2_", "debut2", "perf2_", "edit2_", "debat2",
    "film2_", "teach2", "cura2_", "arch2_", "tran2_", "revw2_",
    "refut2", "synth2", "clon2_", "seqn2_", "imag2_", "calb2_",
    "mapd2_", "chrt2_", "srvy2_", "drll2_", "smpl2_", "tagd2_",
    "ring2_", "band2_", "nest2_", "trck2_",
    "wrote4", "publs3", "provd3", "fond3_", "coind3", "forml3",
    "patnt3", "dscv3_", "lectr3", "tour3_", "awrd3_", "exhbt3",
    "compd3", "solo3_", "debut3", "perf3_", "edit3_", "debat3",
    "film3_", "teach3",
]


def generate_natural_lang_mappings(
    n_groups: int = 50,
    k: int = 20,
    b_length: int = 6,
    z_length: int = 6,
    a_length: int = 6,
    seed: int = 42,
) -> MappingData:
    """Generate the controlled entity-disambiguation dataset.

    Each of *n_groups* role strings is paired with *k* unique
    (clause, action) pairs.  The resulting ``MappingData`` is directly
    usable with ``DisambiguationDataset``.

    String lengths are truncated/padded to the requested fixed lengths
    so the existing tokenizer works unchanged.
    """
    rng = random.Random(seed)

    if n_groups > len(ROLES):
        raise ValueError(f"n_groups={n_groups} exceeds available roles ({len(ROLES)})")
    if k > len(CLAUSE_POOL):
        raise ValueError(f"k={k} exceeds clause pool size ({len(CLAUSE_POOL)})")
    if n_groups * k > len(ACTION_POOL):
        raise ValueError(f"n_groups*k={n_groups * k} exceeds action pool ({len(ACTION_POOL)})")

    roles = ROLES[:n_groups]
    clause_pool = list(CLAUSE_POOL)
    action_pool = list(ACTION_POOL)
    rng.shuffle(clause_pool)
    rng.shuffle(action_pool)

    def _pad(s: str, length: int) -> str:
        s = s[:length]
        return s.ljust(length, "_")

    used_actions: set = set()
    mappings: Dict[str, List[Tuple[str, str]]] = {}
    examples: List[Dict[str, str]] = []
    action_idx = 0

    for gi, role in enumerate(roles):
        b_str = _pad(role, b_length)
        clauses = rng.sample(clause_pool, k)
        entries: List[Tuple[str, str]] = []

        for ci, clause in enumerate(clauses):
            z_str = _pad(clause, z_length)
            # Pick a unique action
            while action_idx < len(action_pool) and action_pool[action_idx] in used_actions:
                action_idx += 1
            if action_idx >= len(action_pool):
                raise ValueError("Ran out of unique actions")
            a_str = _pad(action_pool[action_idx], a_length)
            used_actions.add(action_pool[action_idx])
            action_idx += 1

            entries.append((z_str, a_str))
            examples.append({"b": b_str, "z": z_str, "a": a_str})

        mappings[b_str] = entries

    return MappingData(
        mappings=mappings,
        examples=examples,
        n_unique_b=n_groups,
        n_unique_a=len(used_actions),
        k=k,
        task="bz_to_a",
    )

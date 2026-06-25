import os
import sys

# Programmatic fix for Intel MKL/OpenMP library duplication crash on macOS
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import argparse
import json
import numpy as np

# Include local package path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.config import RankerConfig
from src.logger import get_logger
from src.parser import JDParser, parse_candidate_record, load_candidates
from src.preprocessor import detect_honeypot, clean_text
from src.embedding import CandidateEmbedder
from src.indexer import FAISSIndex
from src.scorer import HybridScorer
from src.reranker import CandidateReranker
from src.reasoning import ReasoningGenerator
from validate_submission import validate_submission

logger = get_logger("ranker")

def count_lines(filepath):
    """Fast line count for text files."""
    count = 0
    with open(filepath, "rb") as f:
        for line in f:
            count += 1
    return count

def build_skills_vocab(profiles):
    """Dynamically builds candidate skill vocabulary."""
    vocab = set()
    for cand in profiles:
        for sk in cand.get("skills", []):
            name = sk.get("name")
            if name:
                vocab.add(name.strip())
    return vocab

def estimate_current_date(profiles, default_date=None):
    """Dynamically estimates reference date based on latest candidate activity."""
    from datetime import datetime
    if default_date is None:
        default_date = datetime(2026, 6, 25)
    max_date = None
    for cand in profiles:
        signals = cand.get("redrob_signals", {})
        for key in ["last_active_date", "signup_date"]:
            d_str = signals.get(key)
            if d_str:
                try:
                    d = datetime.strptime(d_str, "%Y-%m-%d")
                    if max_date is None or d > max_date:
                        max_date = d
                except:
                    pass
    return max_date if max_date is not None else default_date

def load_metadata_cache_summary(summary_path):
    """Loads the precomputed metadata cache summary containing vocab and reference date."""
    with open(summary_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    from datetime import datetime
    ref_date = datetime.strptime(data["reference_date"], "%Y-%m-%d")
    return set(data["skills_vocab"]), ref_date, data["total_count"]

def run_on_the_fly_indexing(candidates, config, embedder, current_date):
    """Runs preprocessing, embedding generation, and indexing on-the-fly for custom pools."""
    logger.info("Custom or small candidate pool detected. Processing on-the-fly...")
    valid_profiles = []
    texts_to_embed = []
    
    for cand in candidates:
        if detect_honeypot(cand, current_date=current_date):
            continue
            
        headline = clean_text(cand["profile"].get("headline", ""))
        summary = clean_text(cand["profile"].get("summary", ""))
        current_title = clean_text(cand["profile"].get("current_title", ""))
        
        profile_text = f"{headline} {summary} {current_title}"
        texts_to_embed.append(profile_text)
        valid_profiles.append(cand)
            
    logger.info(f"Loaded {len(valid_profiles)} non-honeypot candidates on-the-fly.")
    
    if not valid_profiles:
        return None, []
        
    # Generate embeddings
    embeddings = embedder.embed_texts(texts_to_embed, batch_size=128, show_progress=False)
    
    # Build index
    faiss_idx = FAISSIndex(dimension=embeddings.shape[1])
    faiss_idx.add(embeddings)
    
    return faiss_idx, valid_profiles

def main():
    parser = argparse.ArgumentParser(description="Redrob Intelligent Candidate Discovery & Ranker")
    parser.add_argument("--candidates", required=True, help="Path to candidates.jsonl file")
    parser.add_argument("--out", required=True, help="Path to write the output CSV file")
    args = parser.parse_args()
    
    logger.info("Initializing Ranker Engine...")
    config = RankerConfig()
    
    jd_path = config.paths["job_description"]
    index_cache_path = config.paths["index"]
    metadata_cache_path = config.paths["metadata_cache"]
    
    # Ingest embedding model to check dimension
    logger.info("Loading embedding model BAAI/bge-small-en-v1.5...")
    embedder = CandidateEmbedder(model_name=config.model_name, cache_dir=config.model_local_path)
    
    # Determine cache use and line count
    line_count = count_lines(args.candidates)
    logger.info(f"Candidate file contains {line_count} records.")
    
    faiss_idx = None
    profiles = {}
    raw_candidates = None
    
    use_cache = line_count > 50000 and os.path.exists(index_cache_path) and os.path.exists(metadata_cache_path)
    
    skills_vocab = set()
    current_date = None
    total_profiles_count = 0
    
    if use_cache:
        logger.info("Loading precomputed FAISS index...")
        try:
            # Dimension of bge-small-en-v1.5 is 384
            faiss_idx = FAISSIndex(dimension=384)
            faiss_idx.load(index_cache_path)
            
            summary_path = metadata_cache_path.replace(".json", "_summary.json")
            logger.info(f"Loading cache summary from {summary_path}...")
            skills_vocab, current_date, total_profiles_count = load_metadata_cache_summary(summary_path)
            logger.info(f"Summary loaded. Profiles in cache: {total_profiles_count}")
        except Exception as e:
            logger.warning(f"Failed to load cache: {e}. Falling back to on-the-fly indexing.")
            faiss_idx = None
            
    # Build skills vocab and estimate current date dynamically if cache is not used
    if faiss_idx is None:
        # Load raw candidate pool for on-the-fly indexing or fallback
        raw_candidates = load_candidates(args.candidates)
        skills_vocab = build_skills_vocab(raw_candidates)
        current_date = estimate_current_date(raw_candidates)
        
    from datetime import datetime
    logger.info(f"Estimated reference date: {current_date.strftime('%Y-%m-%d')}")
    logger.info(f"Dynamic skill vocabulary size: {len(skills_vocab)}")
    
    # 1. Parse Job Description using dynamically built vocabulary
    logger.info(f"Parsing Job Description from {jd_path}...")
    if not os.path.exists(jd_path):
        logger.error(f"Job Description file {jd_path} not found!")
        sys.exit(1)
        
    jd_parser = JDParser()
    parsed_jd = jd_parser.parse(jd_path, candidate_skills_vocab=skills_vocab)
    
    # 2. Ingest Query Text & Generate Query Embedding
    jd_raw_text = ""
    with open(jd_path, "r", encoding="utf-8") as jf:
        jd_raw_text = jf.read()
        
    jd_clean_text = clean_text(jd_raw_text)
    logger.info("Generating query vector...")
    jd_embedding = embedder.embed_texts([jd_clean_text], show_progress=False)[0]
    
    # On-the-fly indexing if needed
    if faiss_idx is None:
        faiss_idx, profiles_list = run_on_the_fly_indexing(raw_candidates, config, embedder, current_date)
        profiles = {i: cand for i, cand in enumerate(profiles_list)}
        total_profiles_count = len(profiles_list)
        
    if faiss_idx is None or faiss_idx.size() == 0:
        logger.error("No valid candidate profiles to rank! Exiting.")
        sys.exit(1)
        
    # 4. Semantic search retrieval
    k = min(5000, total_profiles_count)
    logger.info(f"Querying FAISS index for top {k} semantic matches...")
    distances, indices = faiss_idx.search(jd_embedding, k=k)
    
    # Load only the retrieved candidate profiles from metadata cache if using cache
    if use_cache and faiss_idx is not None and not raw_candidates:
        profiles = {}
        target_indices = set(indices)
        logger.info(f"Loading top {len(target_indices)} candidate profiles from cache...")
        with open(metadata_cache_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i in target_indices:
                    line_str = line.strip()
                    if line_str:
                        try:
                            cand = json.loads(line_str)
                            profiles[i] = cand
                        except:
                            pass
        logger.info(f"Loaded {len(profiles)} matching profiles.")
        
    # 5. Score and re-rank candidate profiles
    logger.info("Computing hybrid scoring and applying behavioral modifiers...")
    scorer = HybridScorer(config)
    scorer.set_jd_info(parsed_jd)
    
    reranker = CandidateReranker(config)
    reranker.set_jd_info(parsed_jd, current_date=current_date)
    
    candidates_scored = []
    
    for rank_idx, (dist, idx) in enumerate(zip(distances, indices)):
        if idx < 0 or idx not in profiles:
            continue
            
        cand = profiles[idx]
        cid = cand["candidate_id"]
        
        # Calculate scores
        base_score = scorer.compute_score(cand, dist)
        final_score = reranker.rerank_score(cand, base_score)
        
        candidates_scored.append({
            "candidate_id": cid,
            "score": final_score,
            "profile": cand
        })
        
    # 6. Sorting & Deterministic Tie-Breaking
    logger.info("Sorting and tie-breaking candidates...")
    candidates_scored.sort(key=lambda x: (-x["score"], x["candidate_id"]))
    
    # 7. Select Top 100 & Generate Reasoning Justifications
    logger.info("Generating fact-based reasonings for top 100 matches...")
    reasoning_gen = ReasoningGenerator(config)
    reasoning_gen.set_jd_info(parsed_jd)
    
    ranked_results = []
    for rank_pos in range(1, 101):
        if rank_pos - 1 < len(candidates_scored):
            item = candidates_scored[rank_pos - 1]
            cid = item["candidate_id"]
            score = item["score"]
            cand = item["profile"]
            
            # Generate justification
            reasoning = reasoning_gen.generate(cand, rank_pos, scorer=scorer)
            
            ranked_results.append({
                "candidate_id": cid,
                "rank": rank_pos,
                "score": round(score, 4),
                "reasoning": reasoning
            })
        else:
            # Fallback if there are fewer than 100 valid candidates
            ranked_results.append({
                "candidate_id": f"CAND_{rank_pos:07d}",
                "rank": rank_pos,
                "score": 0.0,
                "reasoning": "Fallback profile included as filler."
            })
            
    # 8. Write to submission CSV
    logger.info(f"Writing final submission CSV to {args.out}...")
    import csv
    with open(args.out, "w", encoding="utf-8", newline="") as cf:
        writer = csv.writer(cf)
        writer.writerow(["candidate_id", "rank", "score", "reasoning"])
        for res in ranked_results:
            writer.writerow([res["candidate_id"], res["rank"], f"{res['score']:.4f}", res["reasoning"]])
            
    logger.info("CSV generation completed.")
    
    # 9. Local validation check
    logger.info("Running validation checks on output CSV...")
    errors = validate_submission(args.out)
    if errors:
        logger.error(f"Validation failed with {len(errors)} errors:")
        for err in errors:
            logger.error(f"  - {err}")
        sys.exit(1)
    else:
        logger.info("Output CSV successfully validated. Ready for submission.")

if __name__ == "__main__":
    main()

import json
import os
import sys
import numpy as np

# Include local package path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.config import RankerConfig
from src.logger import get_logger
from src.parser import parse_candidate_record
from src.preprocessor import detect_honeypot, clean_text
from src.embedding import CandidateEmbedder
from src.indexer import FAISSIndex

logger = get_logger("train-indexer")

def main():
    logger.info("Initializing offline index preparation...")
    config = RankerConfig()
    
    candidates_path = config.paths["candidates"]
    index_path = config.paths["index"]
    metadata_cache_path = config.paths["metadata_cache"]
    
    logger.info(f"Loading candidates from {candidates_path}...")
    if not os.path.exists(candidates_path):
        logger.error(f"Candidates file {candidates_path} not found! Exiting.")
        sys.exit(1)
        
    valid_profiles = []
    texts_to_embed = []
    
    # 1. Read and preprocess candidates, drop honeypots
    logger.info("Filtering honeypots and cleaning profiles...")
    honeypot_count = 0
    total_count = 0
    
    with open(candidates_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            total_count += 1
            cand = parse_candidate_record(line)
            
            if detect_honeypot(cand):
                honeypot_count += 1
                continue
                
            # Valid profile: clean and append
            headline = clean_text(cand["profile"].get("headline", ""))
            summary = clean_text(cand["profile"].get("summary", ""))
            current_title = clean_text(cand["profile"].get("current_title", ""))
            
            # Combine fields to build text representation
            profile_text = f"{headline} {summary} {current_title}"
            
            texts_to_embed.append(profile_text)
            valid_profiles.append(cand)
            
    logger.info(f"Scanned {total_count} records. Filtered {honeypot_count} honeypots.")
    logger.info(f"Remaining active candidate count: {len(valid_profiles)}")
    
    # 2. Embedding Generation
    logger.info("Loading embedding model BAAI/bge-small-en-v1.5 on CPU...")
    embedder = CandidateEmbedder(model_name=config.model_name, cache_dir=config.model_local_path)
    
    logger.info("Generating candidate embeddings (this will take 2-4 minutes on CPU)...")
    embeddings = embedder.embed_texts(texts_to_embed, batch_size=256, show_progress=True)
    logger.info(f"Generated {embeddings.shape[0]} embeddings. Dimension: {embeddings.shape[1]}")
    
    # 3. Build FAISS Cosine Index
    logger.info("Building FAISS index...")
    faiss_idx = FAISSIndex(dimension=embeddings.shape[1])
    faiss_idx.add(embeddings)
    
    # 4. Save Index and Metadata Cache
    logger.info(f"Saving FAISS index to {index_path}...")
    faiss_idx.save(index_path)
    
    logger.info(f"Saving metadata cache to {metadata_cache_path}...")
    with open(metadata_cache_path, "w", encoding="utf-8") as mf:
        for p in valid_profiles:
            mf.write(json.dumps(p) + "\n")
            
    # Save cache summary for memory-efficient online loading
    summary_path = metadata_cache_path.replace(".json", "_summary.json")
    logger.info(f"Generating and saving cache summary to {summary_path}...")
    
    vocab = set()
    for cand in valid_profiles:
        for sk in cand.get("skills", []):
            name = sk.get("name")
            if name:
                vocab.add(name.strip())
                
    from datetime import datetime
    max_date = None
    for cand in valid_profiles:
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
                    
    ref_date_str = max_date.strftime("%Y-%m-%d") if max_date is not None else "2026-06-25"
    summary_data = {
        "reference_date": ref_date_str,
        "skills_vocab": list(vocab),
        "total_count": len(valid_profiles)
    }
    with open(summary_path, "w", encoding="utf-8") as sf:
        json.dump(summary_data, sf)
        
    logger.info("Offline index build completed successfully!")

if __name__ == "__main__":
    main()

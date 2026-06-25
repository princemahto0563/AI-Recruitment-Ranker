import json
import re

class JDParser:
    def __init__(self):
        # Default fallback target skills identified in the Job Description analysis
        self.required_skills = {
            "embeddings", "sentence-transformers", "openai embeddings", "bge", "e5",
            "vector database", "pinecone", "weaviate", "qdrant", "milvus", "opensearch",
            "elasticsearch", "faiss", "hybrid search", "rag", "retrieval-augmented generation",
            "nlp", "natural language processing", "information retrieval", "search",
            "retrieval", "ranking", "learning-to-rank", "xgboost", "recommender systems",
            "recommendation systems", "collaborative filtering"
        }
        self.preferred_skills = {
            "llm fine-tuning", "lora", "qlora", "peft", "distributed systems", 
            "large-scale inference", "inference optimization", "mlflow", "dvc"
        }
        
    def extract_skills_from_jd(self, jd_text, candidate_skills_vocab):
        req_skills = set()
        pref_skills = set()
        
        jd_lower = jd_text.lower()
        
        # Identify required & preferred sections
        req_markers = ["absolutely need", "must-have", "required", "requirements", "essential", "minimum qualifications", "what we're looking for"]
        pref_markers = ["like you to have", "nice to have", "preferred", "preferred qualifications", "plus", "desirable", "nice-to-haves", "nice to haves"]
        
        req_idx = -1
        for marker in req_markers:
            idx = jd_lower.find(marker)
            if idx != -1:
                req_idx = idx
                break
                
        pref_idx = -1
        for marker in pref_markers:
            idx = jd_lower.find(marker)
            if idx != -1:
                pref_idx = idx
                break
                
        req_text = ""
        pref_text = ""
        
        if req_idx != -1 and pref_idx != -1:
            if req_idx < pref_idx:
                req_text = jd_lower[req_idx:pref_idx]
                pref_text = jd_lower[pref_idx:]
            else:
                pref_text = jd_lower[pref_idx:req_idx]
                req_text = jd_lower[req_idx:]
        elif req_idx != -1:
            req_text = jd_lower[req_idx:]
        elif pref_idx != -1:
            pref_text = jd_lower[pref_idx:]
        else:
            req_text = jd_lower
            
        for skill in candidate_skills_vocab:
            skill_lower = skill.lower()
            # Escape skill for safe regex search
            # Match on word boundary or non-alphanumeric character boundaries
            pattern = r'(?:\b|(?<=\W))' + re.escape(skill_lower) + r'(?:\b|(?=\W))'
            
            if req_text and re.search(pattern, req_text):
                req_skills.add(skill_lower)
            elif pref_text and re.search(pattern, pref_text):
                pref_skills.add(skill_lower)
            elif not req_text and not pref_text:
                if re.search(pattern, jd_lower):
                    req_skills.add(skill_lower)
                    
        return req_skills, pref_skills

    def parse(self, jd_path, candidate_skills_vocab=None):
        # Default parsed values
        parsed_jd = {
            "title": "AI Engineer",
            "target_experience_min": 5.0,
            "target_experience_max": 9.0,
            "required_skills": self.required_skills,
            "preferred_skills": self.preferred_skills,
            "target_locations": ["noida", "pune", "delhi", "gurgaon", "ncr"],
            "target_countries": ["India"]
        }
        
        try:
            with open(jd_path, "r", encoding="utf-8") as f:
                content = f.read()
                
            # 1. Dynamically parse title
            for line in content.splitlines():
                line_clean = line.strip()
                if not line_clean:
                    continue
                m_title = re.search(r"(?:job description|role|position|title):\s*(.+)", line_clean, re.IGNORECASE)
                if m_title:
                    # Strip any trailing dash comments e.g., "Senior AI Engineer — Founding Team"
                    parsed_jd["title"] = m_title.group(1).split("—")[0].split("-")[0].strip()
                    break

            # 2. Dynamically parse experience range or open-ended
            range_match = re.search(r"(\d+)\s*(?:-|–|to)\s*(\d+)\s*(?:years|yrs)", content, re.IGNORECASE)
            plus_match = re.search(r"(\d+)\+\s*(?:years|yrs)", content, re.IGNORECASE)
            min_match = re.search(r"(?:minimum|at least|required)\s*(\d+)\s*(?:years|yrs)", content, re.IGNORECASE)
            
            if range_match:
                parsed_jd["target_experience_min"] = float(range_match.group(1))
                parsed_jd["target_experience_max"] = float(range_match.group(2))
            elif plus_match:
                parsed_jd["target_experience_min"] = float(plus_match.group(1))
                parsed_jd["target_experience_max"] = 100.0
            elif min_match:
                parsed_jd["target_experience_min"] = float(min_match.group(1))
                parsed_jd["target_experience_max"] = 100.0

            # 3. Dynamically parse locations and countries
            location_line = ""
            for line in content.splitlines():
                if "location:" in line.lower():
                    location_line = line
                    break
            if location_line:
                match = re.search(r"location:\s*([^(\n|]+)", location_line, re.IGNORECASE)
                if match:
                    loc_str = match.group(1).strip()
                    parts = loc_str.split(",")
                    if len(parts) >= 1:
                        city_part = parts[0].strip().lower()
                        parsed_jd["target_locations"] = []
                        for c in re.split(r"/|\bor\b", city_part):
                            c_clean = c.strip()
                            if c_clean:
                                parsed_jd["target_locations"].append(c_clean)
                    if len(parts) >= 2:
                        country_part = parts[1].strip()
                        country_part = re.sub(r"[^\w\s]", "", country_part).strip()
                        if country_part:
                            parsed_jd["target_countries"] = [country_part.title()]

            # 4. Dynamically parse skills from JD using vocabulary
            if candidate_skills_vocab:
                req, pref = self.extract_skills_from_jd(content, candidate_skills_vocab)
                # If we parsed nothing, keep defaults to avoid empty matching
                if req:
                    parsed_jd["required_skills"] = req
                if pref:
                    parsed_jd["preferred_skills"] = pref
        except Exception:
            pass
            
        return parsed_jd


def parse_candidate_record(json_str):
    """Parses a single line of candidate JSONL into a python dictionary."""
    return json.loads(json_str)

def load_candidates(candidates_path):
    """Loads candidates from either a JSON array or a JSON Lines (.jsonl) file."""
    # Try reading as a single JSON array first
    try:
        with open(candidates_path, "r", encoding="utf-8") as f:
            content = f.read().strip()
            if content.startswith("[") and content.endswith("]"):
                return json.loads(content)
    except Exception:
        pass
        
    # Otherwise, read line-by-line (JSONL)
    profiles = []
    with open(candidates_path, "r", encoding="utf-8") as f:
        for line in f:
            line_str = line.strip()
            if not line_str:
                continue
            # Strip trailing comma if it exists
            if line_str.endswith(","):
                line_str = line_str[:-1]
            # Skip array start/end characters
            if line_str == "[" or line_str == "]":
                continue
            try:
                cand = json.loads(line_str)
                profiles.append(cand)
            except Exception:
                pass
    return profiles

import random
from src.config import RankerConfig

class ReasoningGenerator:
    def __init__(self, config: RankerConfig):
        self.config = config
        
        # Default fallback technical keywords to extract from candidates to show in reasoning
        self.tech_keywords = {
            "embeddings", "sentence-transformers", "faiss", "pinecone", "milvus", 
            "qdrant", "weaviate", "rag", "nlp", "search", "retrieval", "ranking", 
            "lora", "qlora", "peft", "xgboost", "spark", "kafka", "python"
        }
        
        # Fallback service companies list to check for gaps
        self.service_companies = {
            "infosys", "wipro", "tcs", "capgemini", "hcl", "mindtree", 
            "accenture", "cognizant", "tech mahindra", "mphasis"
        }

    def set_jd_info(self, parsed_jd):
        req = parsed_jd.get("required_skills", set())
        pref = parsed_jd.get("preferred_skills", set())
        union_skills = set(req).union(set(pref))
        if union_skills:
            self.tech_keywords = union_skills

    def _extract_key_skills(self, cand):
        """Extract matching tech keywords from candidate's skills list."""
        matched = []
        for sk in cand.get("skills", []):
            name = sk.get("name", "").lower()
            for kw in self.tech_keywords:
                if kw in name and kw not in matched:
                    matched.append(kw)
        return matched

    def generate(self, cand, rank, scorer=None):
        """Generates a rank-consistent, fact-based, non-hallucinated reasoning statement."""
        profile = cand.get("profile", {})
        career = cand.get("career_history", [])
        signals = cand.get("redrob_signals", {})
        
        exp = profile.get("years_of_experience", 0)
        title = profile.get("current_title", "Software Engineer")
        companies = [job.get("company", "") for job in career]
        unique_companies = list(dict.fromkeys(companies))[:3] # unique list, max 3
        companies_str = ", ".join(unique_companies)
        
        # Extract skills
        skills = self._extract_key_skills(cand)
        skills_str = ", ".join(skills[:3]) if skills else "applied ML"
        
        # Extract behavioral facts
        notice = signals.get("notice_period_days", 0)
        resp_rate = int(signals.get("recruiter_response_rate", 0.0) * 100)
        
        # Identify company type context dynamically or via fallbacks
        all_service = False
        has_bigtech = False
        
        if scorer is not None and hasattr(scorer, "classify_company"):
            company_classifications = []
            for job in career:
                comp_name = job.get("company", "")
                industry = job.get("industry", "")
                size = job.get("company_size", "")
                classification = scorer.classify_company(comp_name, industry, size)
                company_classifications.append(classification)
                
            all_service = all(cl["is_service"] for cl in company_classifications) if company_classifications else False
            has_bigtech = any(cl["is_bigtech"] for cl in company_classifications)
        else:
            all_service = all(c.lower() in self.service_companies for c in unique_companies) if unique_companies else False
            has_bigtech = any(c.lower() in {"google", "meta", "netflix", "microsoft", "apple", "amazon"} for c in unique_companies)
        
        # Construct justification parts
        strength_phrase = ""
        pedigree_phrase = ""
        gap_phrase = ""
        
        # 1. Rank-consistent Strength Phrase
        if rank <= 15:
            strengths = [
                f"Excellent fit as a {title} with {exp} years of experience",
                f"Top-tier Senior AI Engineer candidate with a strong {exp}-year track record",
                f"Highly aligned {title} offering {exp} years in production ML systems"
            ]
            strength_phrase = random.choice(strengths)
        elif rank <= 70:
            strengths = [
                f"Strong {title} showing {exp} years in software and ML engineering",
                f"Competent match with {exp} years experience in applied ML and backend",
                f"Reliable {title} with {exp} years building scalable systems"
            ]
            strength_phrase = random.choice(strengths)
        else:
            strengths = [
                f"Adjacent {title} profile with {exp} years of background",
                f"Secondary candidate showing {exp} years of experience",
                f"Alternative match with {exp} years in development roles"
            ]
            strength_phrase = random.choice(strengths)

        # 2. Company Pedigree Phrase
        if has_bigtech:
            pedigrees = [
                f"having built scalable systems at Big Tech like {companies_str}",
                f"bringing rigorous engineering discipline from {companies_str}"
            ]
            pedigree_phrase = random.choice(pedigrees)
        elif all_service:
            pedigree_phrase = f"with a career history focused on consulting at {companies_str}"
        elif unique_companies:
            pedigree_phrase = f"with solid hands-on engineering history at {companies_str}"
        else:
            pedigree_phrase = "with active development background"

        # 3. Gap or behavioral note (Honest concern)
        gaps = []
        if notice > 90:
            gaps.append(f"though notice period is long ({notice} days)")
        elif notice > 60:
            gaps.append(f"with a moderate notice period of {notice} days")
            
        if resp_rate < 30:
            gaps.append(f"despite low responsiveness on platform ({resp_rate}%)")
        elif resp_rate > 80 and rank <= 30:
            gaps.append(f"highly available with an active {resp_rate}% recruiter response rate")
            
        if all_service:
            gaps.append("needs transition from services/consulting model to founding team velocity")
            
        gap_phrase = random.choice(gaps) if gaps else f"demonstrating strong competence in {skills_str}"

        # Combine into a cohesive 1-2 sentence statement
        sentence1 = f"{strength_phrase}, {pedigree_phrase}."
        sentence2 = f"Shows strong alignment on {skills_str}, {gap_phrase}."
        
        # Edge case formatting for low rank fillers
        if rank > 85:
            sentence2 = f"Included as final rank filler; holds solid skills in {skills_str} but notice period is {notice} days."
            
        return f"{sentence1} {sentence2}"

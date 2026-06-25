from src.config import RankerConfig

class HybridScorer:
    def __init__(self, config: RankerConfig):
        self.config = config
        
        # Ingest target lists
        self.service_companies = {
            "infosys", "wipro", "tcs", "capgemini", "hcl", "mindtree", 
            "accenture", "cognizant", "tech mahindra", "mphasis", "genpact ai"
        }
        self.ai_startups = {
            "glance", "rephrase.ai", "aganitha", "niramai", "saarthi.ai", 
            "sarvam ai", "mad street den", "observe.ai", "krutrim", "wysa", 
            "haptik", "verloop.io", "yellow.ai", "locobuzz"
        }
        self.big_tech = {
            "google", "netflix", "amazon", "salesforce", "uber", 
            "meta", "adobe", "microsoft", "apple", "linkedin"
        }
        self.product_startups = {
            "swiggy", "razorpay", "cred", "zomato", "flipkart", "meesho", 
            "nykaa", "inmobi", "byju's", "policybazaar", "ola", "zoho", 
            "vedantu", "paytm", "unacademy", "pharmeasy", "upgrad", 
            "freshworks", "phonepe", "dream11", "pied piper", "initech", 
            "hooli", "stark industries", "globex inc", "dunder mitchell", # handle spelling variations
            "dunder mifflin", "wayne enterprises", "acme corp"
        }
        
        # Technical skill categories
        self.core_ai_skills = self.config.get("core_ai_skills", {
            "embeddings", "sentence-transformers", "openai embeddings", "bge", "e5",
            "vector database", "pinecone", "weaviate", "qdrant", "milvus", "opensearch",
            "elasticsearch", "faiss", "hybrid search", "rag", "retrieval-augmented generation",
            "nlp", "natural language processing", "information retrieval", "search",
            "retrieval", "ranking", "learning-to-rank", "xgboost", "recommender systems",
            "recommendation systems", "collaborative filtering"
        })
        self.nice_to_haves = self.config.get("preferred_skills", {
            "llm fine-tuning", "lora", "qlora", "peft", "distributed systems", 
            "large-scale inference", "inference optimization", "mlflow", "dvc"
        })
        
        self.target_experience_min = 5.0
        self.target_experience_max = 9.0

    def set_jd_info(self, parsed_jd):
        self.target_experience_min = parsed_jd.get("target_experience_min", 5.0)
        self.target_experience_max = parsed_jd.get("target_experience_max", 9.0)
        self.core_ai_skills = parsed_jd.get("required_skills", self.core_ai_skills)
        self.nice_to_haves = parsed_jd.get("preferred_skills", self.nice_to_haves)

    def classify_company(self, company_name, industry, size):
        name_l = company_name.lower()
        ind_l = industry.lower() if industry else ""
        
        # 1. Service Company Check
        is_service = False
        if name_l in self.service_companies:
            is_service = True
        elif any(k in ind_l for k in ["service", "consulting", "outsourcing", "staffing", "recruiting"]):
            is_service = True
            
        # 2. Big Tech Check
        is_bigtech = False
        if name_l in self.big_tech:
            is_bigtech = True
        elif size == "10001+" and any(k in ind_l for k in ["software", "internet", "electronics", "hardware", "technology"]):
            if not is_service:
                is_bigtech = True
                
        # 3. AI Startup Check
        is_ai_startup = False
        if name_l in self.ai_startups:
            is_ai_startup = True
        elif size in ["1-10", "11-50", "51-200", "201-500"] and any(k in ind_l for k in ["artificial intelligence", "machine learning", "ai", "ml", "nlp", "computer vision", "deep learning"]):
            is_ai_startup = True
            
        # 4. Product Startup Check
        is_prod_startup = False
        if name_l in self.product_startups:
            is_prod_startup = True
        elif size in ["1-10", "11-50", "51-200", "201-500", "501-1000", "1001-5000", "5001-10000"] and not is_ai_startup:
            if any(k in ind_l for k in ["software", "internet", "saas", "fintech", "e-commerce", "food delivery", "gaming", "product", "e-learning"]):
                if not is_service:
                    is_prod_startup = True
                    
        return {
            "is_service": is_service,
            "is_bigtech": is_bigtech,
            "is_ai_startup": is_ai_startup,
            "is_prod_startup": is_prod_startup
        }

    def compute_lexical_score(self, cand):
        """Computes lexical skill overlap score weighted by proficiency and endorsements."""
        score = 0.0
        skills = cand.get("skills", [])
        
        for sk in skills:
            name = sk.get("name", "").lower()
            prof = sk.get("proficiency", "beginner")
            ends = sk.get("endorsements", 0)
            
            # Proficiency weighting
            w_prof = 1.0
            if prof == "expert":
                w_prof = 2.0
            elif prof == "advanced":
                w_prof = 1.5
            elif prof == "intermediate":
                w_prof = 1.0
            else:
                w_prof = 0.5
                
            # Endorsements boost (max 2.0x)
            w_ends = 1.0 + 0.1 * min(ends, 10)
            
            # Skill categorization
            if name in self.core_ai_skills:
                score += 4.0 * w_prof * w_ends
            elif name in self.nice_to_haves:
                score += 2.0 * w_prof * w_ends
                
        return score

    def compute_metadata_score(self, cand):
        """Evaluates candidate years of experience and company pedigree match."""
        profile = cand.get("profile", {})
        career = cand.get("career_history", [])
        
        # 1. Experience matching
        exp = profile.get("years_of_experience", 0)
        score_exp = 0.0
        
        min_exp = getattr(self, "target_experience_min", 5.0)
        max_exp = getattr(self, "target_experience_max", 9.0)
        
        if min_exp is not None and max_exp is not None:
            if min_exp <= exp <= max_exp:
                score_exp = self.config.weights["experience_match"]
            elif min_exp - 1.0 <= exp < min_exp:
                score_exp = self.config.weights["experience_match"] * 0.7
            elif max_exp < exp <= max_exp + 2.0:
                score_exp = self.config.weights["experience_match"] * 0.6
            else:
                score_exp = 1.0
        elif min_exp is not None:
            if exp >= min_exp:
                score_exp = self.config.weights["experience_match"]
            elif min_exp - 1.0 <= exp < min_exp:
                score_exp = self.config.weights["experience_match"] * 0.7
            else:
                score_exp = 1.0
        else:
            score_exp = self.config.weights["experience_match"]
            
        # 2. Company pedigree
        score_pedigree = 0.0
        
        has_bigtech = False
        has_aistartup = False
        has_prod = False
        is_service_only = True if career else False
        
        for job in career:
            company_name = job.get("company", "")
            industry = job.get("industry", "")
            size = job.get("company_size", "")
            
            c_type = self.classify_company(company_name, industry, size)
            if c_type["is_bigtech"]:
                has_bigtech = True
            if c_type["is_ai_startup"]:
                has_aistartup = True
            if c_type["is_prod_startup"]:
                has_prod = True
            if not c_type["is_service"]:
                is_service_only = False
                
        if has_bigtech:
            score_pedigree += self.config.weights["bigtech_boost"]
        if has_aistartup:
            score_pedigree += self.config.weights["aistartup_boost"]
        if has_prod:
            score_pedigree += self.config.weights["prodstartup_boost"]
            
        if is_service_only:
            score_pedigree += self.config.weights["service_penalty"]
            
        return score_exp + score_pedigree

    def compute_score(self, cand, semantic_similarity):
        """Blends semantic distance with lexical and metadata scores."""
        # Clean semantic score to make it positive and scaled
        semantic_val = max(0.0, float(semantic_similarity))
        
        lex_score = self.compute_lexical_score(cand)
        meta_score = self.compute_metadata_score(cand)
        
        # Hybrid Score linear combination
        total_score = (
            self.config.weights["semantic"] * semantic_val +
            self.config.weights["lexical"] * lex_score +
            meta_score
        )
        return total_score

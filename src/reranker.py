from datetime import datetime
from src.config import RankerConfig

CURRENT_DATE = datetime(2026, 6, 25)

class CandidateReranker:
    def __init__(self, config: RankerConfig):
        self.config = config
        self.target_cities = set(self.config.location["target_cities"])
        self.target_countries = {"India"}
        self.current_date = datetime(2026, 6, 25)

    def set_jd_info(self, parsed_jd, current_date=None):
        self.target_cities = set(parsed_jd.get("target_locations", self.config.location["target_cities"]))
        self.target_countries = set(parsed_jd.get("target_countries", ["India"]))
        if current_date is not None:
            self.current_date = current_date

    def get_location_multiplier(self, cand):
        """Calculates candidate location modifier based on alignment with JD target locations."""
        profile = cand.get("profile", {})
        signals = cand.get("redrob_signals", {})
        
        loc = profile.get("location", "").lower()
        country = profile.get("country", "")
        willing_relocate = signals.get("willing_to_relocate", False)
        
        # Check if they are in target cities
        in_target_city = False
        for city in self.target_cities:
            if city in loc:
                in_target_city = True
                break
                
        # Check country match dynamically (case-insensitive substring match)
        matches_country = False
        if not self.target_countries:
            matches_country = True
        else:
            for tc in self.target_countries:
                if tc.lower() in country.lower() or country.lower() in tc.lower():
                    matches_country = True
                    break

        if matches_country:
            if in_target_city:
                return self.config.location["local_boost"] / 10.0 # 1.0
            elif willing_relocate:
                return self.config.location["relocation_boost"] / 10.0 # 0.85
            else:
                return self.config.location["unwilling_boost"] / 10.0 # 0.2
        else:
            return self.config.location["intl_boost"] # 0.05

    def get_behavioral_multiplier(self, cand):
        """Combines multiple engagement and availability multipliers."""
        signals = cand.get("redrob_signals", {})
        
        # 1. Notice Period
        notice_days = signals.get("notice_period_days", 0)
        notice_cfg = self.config.behavior["notice_period"]
        if notice_days <= 30:
            m_notice = notice_cfg["under_30"]
        elif notice_days <= 60:
            m_notice = notice_cfg["under_60"]
        elif notice_days <= 90:
            m_notice = notice_cfg["under_90"]
        else:
            m_notice = notice_cfg["above_90"]
            
        # 2. Activity Recency
        last_active = signals.get("last_active_date")
        recency_cfg = self.config.behavior["recency"]
        m_recency = recency_cfg["above_180"] # Default to inactive
        if last_active:
            try:
                last_act_dt = datetime.strptime(last_active, "%Y-%m-%d")
                days_inactive = (self.current_date - last_act_dt).days
                if days_inactive <= 30:
                    m_recency = recency_cfg["under_30"]
                elif days_inactive <= 90:
                    m_recency = recency_cfg["under_90"]
                elif days_inactive <= 180:
                    m_recency = recency_cfg["under_180"]
                else:
                    m_recency = recency_cfg["above_180"]
            except:
                pass
                
        # 3. Open to work flag
        otw = signals.get("open_to_work_flag", False)
        m_otw = self.config.behavior["open_to_work"]["true_val"] if otw else self.config.behavior["open_to_work"]["false_val"]
        
        # 4. Recruiter response rate
        resp_rate = signals.get("recruiter_response_rate", 0.0)
        m_resp = 0.4 + 0.6 * resp_rate # range 0.4 to 1.0
        
        # 5. Interview completion rate
        int_rate = signals.get("interview_completion_rate", 0.0)
        min_int = self.config.behavior["interview_completion"]["min_multiplier"] # 0.7
        m_int = min_int + (1.0 - min_int) * int_rate # range 0.7 to 1.0
        
        return m_notice * m_recency * m_otw * m_resp * m_int

    def rerank_score(self, cand, base_score):
        """Applies location and behavioral multipliers to base score."""
        loc_mult = self.get_location_multiplier(cand)
        behavior_mult = self.get_behavioral_multiplier(cand)
        
        final_score = base_score * loc_mult * behavior_mult
        return final_score

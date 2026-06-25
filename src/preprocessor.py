import re
from datetime import datetime

CURRENT_DATE = datetime(2026, 6, 25)

def clean_text(text):
    """Clean and normalize text fields."""
    if not text:
        return ""
    text = text.lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text

def detect_honeypot(cand, current_date=None):
    """
    Implements 4 disjoint dynamic boundary checks to detect impossible profiles (honeypots).
    Returns True if the candidate is classified as a honeypot/fake profile.
    """
    if current_date is None:
        current_date = datetime(2026, 6, 25)
        
    profile = cand.get('profile', {})
    career = cand.get('career_history', [])
    skills = cand.get('skills', [])
    certs = cand.get('certifications', [])
    
    # Check 1: Job duration anomalies (claimed duration exceeds actual dates)
    for job in career:
        start_s = job.get('start_date')
        end_s = job.get('end_date')
        duration = job.get('duration_months', 0)
        if start_s:
            try:
                start_dt = datetime.strptime(start_s, '%Y-%m-%d')
                end_dt = datetime.strptime(end_s, '%Y-%m-%d') if end_s else current_date
                elapsed = (end_dt.year - start_dt.year) * 12 + (end_dt.month - start_dt.month)
                if duration > elapsed + 2:
                    return True
            except:
                pass
                
    # Check 2: Expert skills with 0 duration
    expert_0_skills = [
        s['name'] for s in skills 
        if s.get('proficiency') == 'expert' and s.get('duration_months', 0) == 0
    ]
    if len(expert_0_skills) >= 3:
        return True
        
    # Check 3: Experience mismatches
    years_num = profile.get('years_of_experience', 0)
    
    # 3a. History vs Profile mismatch
    total_months = sum(job.get('duration_months', 0) for job in career)
    years_history = total_months / 12.0
    if abs(years_history - years_num) > 5.0:
        return True
        
    # 3b. Text description vs Profile mismatch
    headline = profile.get('headline', '')
    summary = profile.get('summary', '')
    
    hl_years = None
    m = re.search(r'([0-9.]+)\+\s*yrs', headline)
    if m:
        hl_years = float(m.group(1))
        
    sum_years = None
    m2 = re.search(r'with\s+([0-9.]+)\+?\s*years', summary)
    if m2:
        sum_years = float(m2.group(1))
        
    text_diff = 0
    if hl_years is not None:
        text_diff = max(text_diff, abs(hl_years - years_num))
    if sum_years is not None:
        text_diff = max(text_diff, abs(sum_years - years_num))
        
    if text_diff > 5.0:
        return True
        
    # Check 4: Future certifications
    for c in certs:
        year = c.get('year')
        if year and year > current_date.year:
            return True
            
    return False

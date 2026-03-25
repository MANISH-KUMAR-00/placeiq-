# -*- coding: utf-8 -*-
"""
                                                        
    PlaceIQ   Campus Placement Predictor v2             
    HOW TO RUN:                                         
    1. pip install flask scikit-learn pandas numpy      
       pdfplumber python-docx                           
    2. python run_app.py                                
    3. Open http://localhost:5000                       
                                                        
"""

import sys
import io
import subprocess
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


#    Auto-install missing packages                       
REQUIRED = ["flask", "sklearn", "pandas", "numpy", "pdfplumber", "docx"]
INSTALL_MAP = {
    "flask": "flask",
    "sklearn": "scikit-learn",
    "pandas": "pandas",
    "numpy": "numpy",
    "pdfplumber": "pdfplumber",
    "docx": "python-docx",
}

print("[*] Checking dependencies...")
missing = []
for pkg, pip_name in INSTALL_MAP.items():
    try:
        __import__(pkg)
    except ImportError:
        missing.append(pip_name)

if missing:
    print(f"[+] Installing: {', '.join(missing)}")
    subprocess.check_call([sys.executable, "-m", "pip", "install"] + missing + ["--quiet"])
    print("[OK] Installed successfully!\n")
else:
    print("[OK] All dependencies found!\n")

#    Now run the app                                       
import os, json, re, pickle
import numpy as np
from flask import Flask, render_template_string, request, jsonify
from werkzeug.utils import secure_filename

#                                                         
#  COURSE CONFIGURATION
#                                                         
COURSE_CONFIG = {
    "B.Tech": {
        "features": ["cgpa","backlogs","internships","projects","hackathons",
                     "aptitude_score","coding_score","technical_score",
                     "communication_score","soft_skills_score",
                     "workshops_certifications","github_projects",
                     "competitive_programming","extracurricular","gender"],
        "weights": {
            "cgpa":0.20,"coding_score":0.18,"technical_score":0.15,
            "internships":0.12,"aptitude_score":0.10,"projects":0.08,
            "hackathons":0.06,"competitive_programming":0.05,
            "communication_score":0.04,"github_projects":0.03,
            "backlogs":-0.08,"soft_skills_score":0.02,
            "workshops_certifications":0.02,"extracurricular":0.01
        },
        "avg_salary": 6.5, "placement_rate": 0.72
    },
    "M.Tech": {
        "features": ["cgpa","backlogs","internships","research_papers",
                     "technical_score","coding_score","aptitude_score",
                     "communication_score","soft_skills_score",
                     "workshops_certifications","industry_experience_months",
                     "projects","extracurricular","github_projects","gender"],
        "weights": {
            "cgpa":0.22,"research_papers":0.18,"technical_score":0.16,
            "industry_experience_months":0.12,"coding_score":0.10,
            "aptitude_score":0.08,"internships":0.06,"projects":0.05,
            "communication_score":0.04,"backlogs":-0.08,
            "workshops_certifications":0.02,"github_projects":0.02,
            "soft_skills_score":0.01,"extracurricular":0.01
        },
        "avg_salary": 9.5, "placement_rate": 0.78
    },
    "MCA": {
        "features": ["cgpa","backlogs","internships","projects",
                     "coding_score","technical_score","aptitude_score",
                     "communication_score","soft_skills_score",
                     "workshops_certifications","database_skills",
                     "web_dev_skills","extracurricular","github_projects","gender"],
        "weights": {
            "cgpa":0.18,"coding_score":0.18,"technical_score":0.14,
            "internships":0.12,"aptitude_score":0.10,"projects":0.09,
            "database_skills":0.07,"web_dev_skills":0.06,
            "communication_score":0.04,"github_projects":0.03,
            "backlogs":-0.08,"soft_skills_score":0.02,
            "workshops_certifications":0.02,"extracurricular":0.01
        },
        "avg_salary": 5.5, "placement_rate": 0.68
    },
    "BCA": {
        "features": ["cgpa","backlogs","internships","projects",
                     "coding_score","aptitude_score","communication_score",
                     "soft_skills_score","workshops_certifications",
                     "web_dev_skills","database_skills",
                     "extracurricular","github_projects","gender"],
        "weights": {
            "cgpa":0.20,"coding_score":0.18,"aptitude_score":0.14,
            "internships":0.12,"web_dev_skills":0.10,"projects":0.09,
            "communication_score":0.06,"database_skills":0.04,
            "github_projects":0.03,"backlogs":-0.08,
            "soft_skills_score":0.03,"workshops_certifications":0.02,
            "extracurricular":0.01
        },
        "avg_salary": 4.2, "placement_rate": 0.62
    },
    "MBA": {
        "features": ["cgpa","backlogs","internships","communication_score",
                     "leadership_score","aptitude_score","soft_skills_score",
                     "group_discussion_score","case_study_score",
                     "work_experience_months","extracurricular",
                     "sports_achievements","workshops_certifications","gender"],
        "weights": {
            "communication_score":0.20,"leadership_score":0.18,
            "group_discussion_score":0.15,"aptitude_score":0.12,
            "work_experience_months":0.10,"cgpa":0.08,
            "case_study_score":0.07,"soft_skills_score":0.06,
            "internships":0.04,"extracurricular":0.03,
            "backlogs":-0.06,"sports_achievements":0.02,
            "workshops_certifications":0.01
        },
        "avg_salary": 8.5, "placement_rate": 0.80
    },
    "BBA": {
        "features": ["cgpa","backlogs","internships","communication_score",
                     "leadership_score","aptitude_score","soft_skills_score",
                     "group_discussion_score","extracurricular",
                     "sports_achievements","workshops_certifications",
                     "case_study_score","work_experience_months","gender"],
        "weights": {
            "communication_score":0.22,"leadership_score":0.18,
            "aptitude_score":0.14,"group_discussion_score":0.12,
            "cgpa":0.10,"soft_skills_score":0.08,
            "internships":0.06,"extracurricular":0.04,
            "case_study_score":0.03,"backlogs":-0.06,
            "sports_achievements":0.02,"workshops_certifications":0.01,
            "work_experience_months":0.01
        },
        "avg_salary": 4.8, "placement_rate": 0.65
    }
}

#                                                         
#  DATASET GENERATION & MODEL TRAINING
#                                                         
MODEL_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "placeiq_models.pkl")

def generate_course_dataset(course, n=20000):
    from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, VotingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.metrics import accuracy_score

    np.random.seed(hash(course) % (2**32))
    cfg = COURSE_CONFIG[course]
    data = {}
    data['cgpa'] = np.round(np.clip(np.random.normal(7.4,1.3,n),4.0,10.0),2)
    data['backlogs'] = np.random.choice([0,1,2,3,4],n,p=[0.55,0.22,0.12,0.07,0.04])
    data['internships'] = np.random.choice([0,1,2,3],n,p=[0.25,0.40,0.25,0.10])
    data['projects'] = np.random.choice([0,1,2,3,4,5],n,p=[0.05,0.20,0.30,0.25,0.15,0.05])
    data['aptitude_score'] = np.round(np.clip(np.random.normal(63,16,n),20,100),1)
    data['communication_score'] = np.round(np.clip(np.random.normal(6.5,1.8,n),1,10),1)
    data['soft_skills_score'] = np.round(np.clip(np.random.normal(6.5,1.8,n),1,10),1)
    data['extracurricular'] = np.random.choice([0,1],n,p=[0.38,0.62])
    data['workshops_certifications'] = np.random.choice([0,1,2,3,4],n,p=[0.15,0.30,0.30,0.18,0.07])
    data['gender'] = np.random.choice([0,1],n,p=[0.45,0.55])
    data['coding_score'] = np.round(np.clip(np.random.normal(6.0,2.0,n),1,10),1)
    data['technical_score'] = np.round(np.clip(np.random.normal(6.5,1.8,n),1,10),1)
    data['hackathons'] = np.random.choice([0,1,2,3],n,p=[0.40,0.35,0.18,0.07])
    data['github_projects'] = np.random.choice([0,1,2,3,4,5],n,p=[0.10,0.20,0.30,0.25,0.10,0.05])
    data['competitive_programming'] = np.random.choice([0,1,2,3],n,p=[0.40,0.30,0.20,0.10])
    data['research_papers'] = np.random.choice([0,1,2,3],n,p=[0.60,0.25,0.10,0.05])
    data['industry_experience_months'] = np.random.choice([0,6,12,18,24,36],n,p=[0.30,0.20,0.25,0.12,0.08,0.05])
    data['database_skills'] = np.round(np.clip(np.random.normal(6.0,2.0,n),1,10),1)
    data['web_dev_skills'] = np.round(np.clip(np.random.normal(6.0,2.0,n),1,10),1)
    data['leadership_score'] = np.round(np.clip(np.random.normal(6.5,1.8,n),1,10),1)
    data['group_discussion_score'] = np.round(np.clip(np.random.normal(6.5,1.8,n),1,10),1)
    data['case_study_score'] = np.round(np.clip(np.random.normal(6.0,1.8,n),1,10),1)
    data['work_experience_months'] = np.random.choice([0,6,12,18,24,36,48],n,p=[0.35,0.15,0.20,0.12,0.10,0.05,0.03])
    data['sports_achievements'] = np.random.choice([0,1,2,3],n,p=[0.50,0.30,0.15,0.05])

    import pandas as pd
    df = pd.DataFrame(data)
    features = cfg["features"]
    weights = cfg["weights"]
    score = np.zeros(n)
    for feat, w in weights.items():
        if feat in df.columns:
            col = df[feat].values
            col_norm = (col - col.min()) / (col.max() - col.min() + 1e-9)
            score += w * col_norm * 100
    noise = np.random.normal(0, 4, n)
    score += noise
    threshold = np.percentile(score, (1 - cfg["placement_rate"]) * 100)
    df['placed'] = (score > threshold).astype(int)
    return df[features], df['placed']

def train_all_models():
    from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, VotingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.metrics import accuracy_score

    models, accuracies, importances = {}, {}, {}
    for course in COURSE_CONFIG:
        print(f"  [#] Training {course}...", end=" ", flush=True)
        X, y = generate_course_dataset(course, 20000)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        gb = GradientBoostingClassifier(n_estimators=200, learning_rate=0.08, max_depth=5, subsample=0.85, random_state=42)
        rf = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42, n_jobs=-1)
        lr = LogisticRegression(C=1.5, max_iter=1000, random_state=42)
        ensemble = VotingClassifier([('gb',gb),('rf',rf),('lr',lr)], voting='soft', weights=[3,2,1])
        pipe = Pipeline([('scaler', StandardScaler()), ('clf', ensemble)])
        pipe.fit(X_train, y_train)

        acc = accuracy_score(y_test, pipe.predict(X_test))
        accuracies[course] = round(acc * 100, 2)
        fi = dict(zip(COURSE_CONFIG[course]['features'], pipe.named_steps['clf'].estimators_[1].feature_importances_))
        importances[course] = fi
        models[course] = pipe
        print(f"[OK] {acc*100:.1f}% accuracy")

    with open(MODEL_FILE, 'wb') as f:
        pickle.dump({'models': models, 'accuracies': accuracies, 'importances': importances}, f)
    return models, accuracies, importances

def load_or_train():
    if os.path.exists(MODEL_FILE):
        print("[OK] Loading saved models...")
        with open(MODEL_FILE, 'rb') as f:
            d = pickle.load(f)
        return d['models'], d['accuracies'], d['importances']
    print("[..] Training models on 20,000 students per course (first run only, ~60 seconds)...")
    return train_all_models()

#                                                         
#  RESUME ANALYSIS
#                                                         
SKILL_KEYWORDS = {
    "programming_languages": ["python","java","c++","c#","javascript","typescript","golang","kotlin","swift","php","ruby","scala","matlab","sql"],
    "web_frameworks": ["react","angular","vue","django","flask","spring","node.js","express","fastapi","laravel","nextjs"],
    "databases": ["mysql","postgresql","mongodb","redis","oracle","sqlite","cassandra","firebase","elasticsearch"],
    "cloud_devops": ["aws","azure","gcp","docker","kubernetes","jenkins","terraform","ansible","linux","git","github","gitlab"],
    "data_ml": ["machine learning","deep learning","tensorflow","pytorch","keras","scikit-learn","pandas","numpy","tableau","power bi","data science","nlp","neural network"],
    "management_skills": ["leadership","project management","agile","scrum","team management","strategic planning","business development","marketing","sales","finance","crm","erp"],
    "soft_skills": ["communication","teamwork","problem solving","critical thinking","time management","adaptability","creativity"]
}

def analyze_resume(filepath, course):
    text = ""
    ext = os.path.splitext(filepath)[1].lower()
    try:
        if ext == '.pdf':
            import pdfplumber
            with pdfplumber.open(filepath) as pdf:
                text = " ".join(p.extract_text() or "" for p in pdf.pages)
        elif ext in ['.docx', '.doc']:
            from docx import Document
            doc = Document(filepath)
            text = " ".join(p.text for p in doc.paragraphs)
        elif ext == '.txt':
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                text = f.read()
    except Exception as e:
        return {"error": f"Could not read file: {e}"}

    if len(text.strip()) < 50:
        return {"error": "Could not extract meaningful text. Try a different file format."}

    tl = text.lower()
    found_skills = {cat: [kw for kw in kws if kw in tl] for cat, kws in SKILL_KEYWORDS.items()}
    total_found = sum(len(v) for v in found_skills.values())
    total_max = sum(len(v) for v in SKILL_KEYWORDS.values())
    skill_coverage = round(total_found / total_max * 100, 1)

    sections = {"education":["education","academic","qualification"],"experience":["experience","internship","employment"],
                "skills":["skills","competencies","technologies"],"achievements":["achievement","award","certification"],
                "projects":["project","portfolio"],"research":["research","publication","thesis"]}
    detected = [s for s, hdrs in sections.items() if any(h in tl for h in hdrs)]

    wc = len(text.split())
    email = bool(re.search(r'\b[\w._%+-]+@[\w.-]+\.[A-Za-z]{2,}\b', text))
    phone = bool(re.search(r'[\+] [\d\s\-\(\)]{10,}', text))
    linkedin = "linkedin" in tl
    github = "github" in tl
    has_nums = bool(re.search(r'\b\d+[%+x]\b', text))

    course_rel = {"B.Tech":["programming_languages","web_frameworks","databases","cloud_devops","data_ml"],
                  "M.Tech":["programming_languages","data_ml","cloud_devops"],
                  "MCA":["programming_languages","web_frameworks","databases"],
                  "BCA":["programming_languages","web_frameworks","databases"],
                  "MBA":["management_skills","soft_skills"],"BBA":["management_skills","soft_skills"]}
    rel_cats = course_rel.get(course, list(SKILL_KEYWORDS.keys()))
    rel_found = sum(len(found_skills.get(c,[])) for c in rel_cats)
    rel_max = sum(len(SKILL_KEYWORDS.get(c,[])) for c in rel_cats)
    course_fit = round(rel_found / max(rel_max,1) * 100, 1)

    overall = min(100, round(
        len(detected)/len(sections)*25 + min(skill_coverage*0.35,35) +
        (email+phone+linkedin+github)*3 + min(max(wc-150,0)/10,15) + (5 if has_nums else 0), 1))

    strengths, gaps = [], []
    if len(found_skills.get("programming_languages",[])) >= 3: strengths.append("Strong multi-language programming background")
    if len(found_skills.get("data_ml",[])) >= 3: strengths.append("Good exposure to Data Science / ML tools")
    if github: strengths.append("GitHub presence mentioned")
    if linkedin: strengths.append("LinkedIn profile included")
    if has_nums: strengths.append("Uses quantifiable achievements")
    if len(detected) >= 5: strengths.append("Well-structured with all key sections")
    if not email: gaps.append("No email detected   add contact info")
    if not github and course in ["B.Tech","MCA","BCA","M.Tech"]: gaps.append("Add GitHub profile link")
    if not linkedin: gaps.append("Add LinkedIn profile URL")
    if wc < 200: gaps.append("Resume too short   add more detail")
    if wc > 1200: gaps.append("Resume too long   aim for 1 2 pages")
    if not has_nums: gaps.append("Add quantifiable achievements (e.g. 'improved by 40%')")
    if "skills" not in detected: gaps.append("Add a dedicated Skills section")
    if not found_skills.get("programming_languages",[]) and course in ["B.Tech","MCA","BCA","M.Tech"]:
        gaps.append("No programming languages detected   list your tech stack")

    ats_kws = {"B.Tech":["github","agile","rest api","data structures","algorithms"],
               "M.Tech":["research","thesis","system design","advanced algorithms"],
               "MCA":["full stack","database","sql","web development"],
               "BCA":["html","css","javascript","programming"],
               "MBA":["strategy","roi","stakeholder","budget"],"BBA":["marketing","sales","customer","crm"]}
    ats_missing = [k for k in ats_kws.get(course,[]) if k not in tl]

    return {"overall_score":overall,"course_fit_score":course_fit,"skill_coverage":skill_coverage,
            "word_count":wc,"detected_sections":detected,
            "found_skills":{k:v for k,v in found_skills.items() if v},
            "strengths":strengths[:5],"gaps":gaps[:6],"ats_missing":ats_missing,
            "contact_info":{"email":email,"phone":phone,"linkedin":linkedin,"github":github}}

#                                                         
#  ANALYTICS
#                                                         
FN = {"cgpa":"CGPA","backlogs":"Backlogs","internships":"Internships","projects":"Projects",
      "hackathons":"Hackathons","aptitude_score":"Aptitude Score","coding_score":"Coding Skills",
      "technical_score":"Technical Skills","communication_score":"Communication",
      "soft_skills_score":"Soft Skills","workshops_certifications":"Workshops & Certs",
      "github_projects":"GitHub Projects","competitive_programming":"Competitive Prog.",
      "research_papers":"Research Papers","industry_experience_months":"Industry Exp.",
      "database_skills":"Database Skills","web_dev_skills":"Web Dev Skills",
      "leadership_score":"Leadership","group_discussion_score":"Group Discussion",
      "case_study_score":"Case Study","work_experience_months":"Work Experience",
      "sports_achievements":"Sports Achievements","extracurricular":"Extracurricular","gender":"Gender"}

MAXES = {"cgpa":10,"backlogs":4,"internships":3,"projects":5,"hackathons":3,"aptitude_score":100,
         "coding_score":10,"technical_score":10,"communication_score":10,"soft_skills_score":10,
         "workshops_certifications":4,"github_projects":5,"competitive_programming":3,"research_papers":3,
         "industry_experience_months":36,"database_skills":10,"web_dev_skills":10,"leadership_score":10,
         "group_discussion_score":10,"case_study_score":10,"work_experience_months":48,"sports_achievements":3}

BENCHMARKS = {
    "B.Tech":{"min_cgpa":6.5,"rec_cgpa":7.5,"min_internships":1,"avg_package":"6-12 LPA"},
    "M.Tech":{"min_cgpa":7.0,"rec_cgpa":8.0,"min_internships":1,"avg_package":"8-18 LPA"},
    "MCA":{"min_cgpa":6.5,"rec_cgpa":7.5,"min_internships":1,"avg_package":"5-10 LPA"},
    "BCA":{"min_cgpa":6.0,"rec_cgpa":7.0,"min_internships":1,"avg_package":"3-7 LPA"},
    "MBA":{"min_cgpa":6.0,"rec_cgpa":7.0,"min_internships":2,"avg_package":"6-15 LPA"},
    "BBA":{"min_cgpa":6.0,"rec_cgpa":7.0,"min_internships":1,"avg_package":"3-8 LPA"},
}

def get_analytics(course, inp, prob):
    cfg = COURSE_CONFIG[course]
    feats = [f for f in cfg["features"] if f != "gender"]
    wts = cfg["weights"]
    pos = {k:v for k,v in wts.items() if v>0 and k in feats}
    tot = sum(pos.values())
    importance = [{"feature":FN.get(k,k),"importance":round(v/tot*100,1)}
                  for k,v in sorted(pos.items(),key=lambda x:-x[1]) if k!="gender"][:8]
    breakdown = []
    for f in feats[:8]:
        if f in inp:
            v = inp[f]; mx = MAXES.get(f,10)
            pct = max(0,(1-v/4)*100) if f=="backlogs" else min(v/mx*100,100) if mx else 0
            breakdown.append({"feature":FN.get(f,f),"value":v,"percent":round(pct,1),"weight":round(abs(wts.get(f,0))*100,1)})
    return {"importance":importance,"score_breakdown":breakdown,
            "benchmark":BENCHMARKS.get(course,{}),"placement_rate":cfg["placement_rate"]*100,
            "avg_salary":cfg["avg_salary"],"probability":prob}

#                                                         
#  TIPS ENGINE
#                                                         
def generate_tips(course, inp, prob):
    tips = {"good":[],"improve":[],"critical":[]}
    cgpa = inp.get("cgpa",0); backlogs = inp.get("backlogs",0)
    internships = inp.get("internships",0); comm = inp.get("communication_score",0)
    apt = inp.get("aptitude_score",0); coding = inp.get("coding_score",0)
    leadership = inp.get("leadership_score",0)

    if backlogs > 0: tips["critical"].append(f"You have {int(backlogs)} backlog(s)   clear them immediately, they block shortlisting")
    if cgpa < 6.5: tips["critical"].append("CGPA below 6.5 blocks many company shortlists   focus on academics now")
    if course in ["B.Tech","MCA","BCA","M.Tech"]:
        if coding >= 7: tips["good"].append("Strong coding skills   well positioned for tech interviews")
        elif coding < 5: tips["improve"].append("Practice DSA daily on LeetCode/HackerRank   coding rounds are gatekeepers")
        if inp.get("github_projects",0) == 0: tips["improve"].append("Create GitHub profile and push your projects publicly")
        if inp.get("hackathons",0) >= 2: tips["good"].append("Hackathon experience shows initiative   highlight it in resume")
    if course in ["MBA","BBA"]:
        if comm >= 8: tips["good"].append("Excellent communication skills   crucial for management roles")
        elif comm < 6: tips["improve"].append("Join debate club or Toastmasters to sharpen communication")
        if leadership >= 8: tips["good"].append("Strong leadership   emphasize in GD/PI rounds")
        elif leadership < 5: tips["improve"].append("Take leadership roles in college events/clubs")
    if internships >= 2: tips["good"].append("Strong internship record   major differentiator from peers")
    elif internships == 0: tips["improve"].append("Apply for internships urgently   even 1 transforms your profile")
    if apt >= 75: tips["good"].append("High aptitude score   you'll clear most written tests")
    elif apt < 55: tips["improve"].append("Practice aptitude daily   R.S. Aggarwal or IndiaBix, 30 min/day")
    if prob >= 80: tips["good"].append("Overall profile is very competitive   focus on interview prep now")
    elif prob < 40: tips["critical"].append("Profile needs significant improvement   focus on top 2 3 weak areas first")
    return tips

#                                                         
#  FLASK APP + HTML
#                                                         
UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "resume_uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024

# Load models
MODELS, ACCURACIES, IMPORTANCES = load_or_train()
print(f"\n[>>] PlaceIQ is ready!")
print(f"   Accuracies: { {c: f'{a}%' for c,a in ACCURACIES.items()} }")

COURSE_INFO_JSON = json.dumps({
    course: {"accuracy": ACCURACIES[course],
             "features": cfg["features"],
             "placement_rate": cfg["placement_rate"] * 100,
             "avg_salary": cfg["avg_salary"]}
    for course, cfg in COURSE_CONFIG.items()
})

HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>PlaceIQ — Campus Placement Predictor</title>
<link href="https://fonts.googleapis.com/css2?family=Cabinet+Grotesk:wght@400;500;600;700;800&family=Instrument+Sans:wght@300;400;500&display=swap" rel="stylesheet"/>
<style>
:root{--bg:#06080f;--surface:#0d1117;--card:#111820;--card2:#161e2a;--border:#1c2940;--border2:#243352;--accent:#4f9cf9;--accent-dim:rgba(79,156,249,0.12);--green:#22d3a0;--green-dim:rgba(34,211,160,0.1);--red:#ff6b6b;--red-dim:rgba(255,107,107,0.1);--amber:#fbbf24;--purple:#a78bfa;--text:#dce8f7;--text2:#8fabc7;--muted:#4a6382;--r:14px;}
*{margin:0;padding:0;box-sizing:border-box;}html{scroll-behavior:smooth;}
body{background:var(--bg);color:var(--text);font-family:'Instrument Sans',sans-serif;min-height:100vh;overflow-x:hidden;}
.bg-mesh{position:fixed;inset:0;z-index:0;pointer-events:none;background:radial-gradient(ellipse 60% 40% at 10% 20%,rgba(79,156,249,0.06),transparent 60%),radial-gradient(ellipse 50% 40% at 90% 80%,rgba(167,139,250,0.06),transparent 60%);}
.bg-dots{position:fixed;inset:0;z-index:0;pointer-events:none;background-image:radial-gradient(rgba(79,156,249,0.05) 1px,transparent 1px);background-size:32px 32px;}
.wrap{position:relative;z-index:1;max-width:1000px;margin:0 auto;padding:40px 20px 100px;}
header{text-align:center;padding:20px 0 48px;}
.logo{display:inline-flex;align-items:center;gap:10px;margin-bottom:24px;background:var(--card);border:1px solid var(--border);border-radius:100px;padding:8px 20px 8px 12px;font-size:13px;color:var(--accent);}
.logo-dot{width:28px;height:28px;background:linear-gradient(135deg,var(--accent),var(--purple));border-radius:8px;display:flex;align-items:center;justify-content:center;font-size:14px;}
h1{font-family:'Cabinet Grotesk',sans-serif;font-size:clamp(2rem,6vw,3.6rem);font-weight:800;line-height:1.08;letter-spacing:-0.02em;background:linear-gradient(160deg,#dce8f7,var(--accent) 50%,var(--purple));-webkit-background-clip:text;-webkit-text-fill-color:transparent;margin-bottom:14px;}
.sub{color:var(--text2);font-size:15px;max-width:500px;margin:0 auto;line-height:1.7;}
.tabs{display:flex;gap:4px;background:var(--card);border:1px solid var(--border);border-radius:14px;padding:4px;margin-bottom:28px;}
.tab{flex:1;padding:12px;border:none;background:transparent;color:var(--muted);font-family:'Cabinet Grotesk',sans-serif;font-size:14px;font-weight:600;border-radius:10px;cursor:pointer;transition:all .2s;}
.tab.active{background:var(--card2);color:var(--text);box-shadow:0 2px 8px rgba(0,0,0,.3);}
.tab:hover:not(.active){color:var(--text2);}
.panel{display:none;animation:fadeIn .35s ease;}.panel.active{display:block;}
@keyframes fadeIn{from{opacity:0;transform:translateY(10px)}to{opacity:1;transform:translateY(0)}}
.card{background:var(--card);border:1px solid var(--border);border-radius:20px;padding:28px;}.card+.card{margin-top:16px;}
.course-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:10px;margin-bottom:24px;}
@media(max-width:560px){.course-grid{grid-template-columns:repeat(2,1fr);}}
.course-btn{padding:14px 8px;background:var(--card2);border:1px solid var(--border);border-radius:12px;color:var(--text2);font-family:'Cabinet Grotesk',sans-serif;font-size:13px;font-weight:600;cursor:pointer;transition:all .2s;text-align:center;}
.course-btn span.ci{font-size:20px;display:block;margin-bottom:4px;}
.course-btn .cr{font-size:10px;color:var(--muted);font-weight:400;}
.course-btn.selected{background:var(--accent-dim);border-color:var(--accent);color:var(--accent);}
.course-btn:hover:not(.selected){border-color:var(--border2);color:var(--text);}
.sec{font-family:'Cabinet Grotesk',sans-serif;font-size:11px;font-weight:700;color:var(--accent);text-transform:uppercase;letter-spacing:.12em;margin-bottom:16px;display:flex;align-items:center;gap:10px;}
.sec::after{content:'';flex:1;height:1px;background:linear-gradient(90deg,var(--border),transparent);}
.fields{display:grid;grid-template-columns:repeat(auto-fill,minmax(200px,1fr));gap:14px;margin-bottom:20px;}
.field{display:flex;flex-direction:column;gap:6px;}
label{font-size:11px;font-weight:500;color:var(--text2);text-transform:uppercase;letter-spacing:.06em;}
input[type=number],select{background:var(--surface);border:1px solid var(--border);border-radius:10px;padding:10px 12px;color:var(--text);font-size:14px;font-family:'Instrument Sans',sans-serif;outline:none;transition:all .2s;width:100%;-moz-appearance:textfield;}
input:focus,select:focus{border-color:var(--accent);box-shadow:0 0 0 3px rgba(79,156,249,.1);}
input::-webkit-outer-spin-button,input::-webkit-inner-spin-button{-webkit-appearance:none;}
select option{background:var(--surface);}
.sl-wrap{display:flex;align-items:center;gap:10px;}
input[type=range]{flex:1;-webkit-appearance:none;height:3px;background:var(--border);border-radius:99px;border:none;padding:0;cursor:pointer;outline:none;}
input[type=range]::-webkit-slider-thumb{-webkit-appearance:none;width:16px;height:16px;background:var(--accent);border-radius:50%;border:2px solid var(--bg);cursor:pointer;}
.sl-val{min-width:26px;text-align:center;font-family:'Cabinet Grotesk',sans-serif;font-weight:700;font-size:14px;color:var(--accent);}
.tog{display:flex;gap:6px;}
.tog-btn{flex:1;padding:10px;background:var(--surface);border:1px solid var(--border);border-radius:9px;color:var(--muted);font-size:12px;cursor:pointer;transition:all .2s;font-family:'Instrument Sans',sans-serif;}
.tog-btn.on{background:var(--green-dim);border-color:var(--green);color:var(--green);}
.tog-btn.off{background:var(--red-dim);border-color:var(--red);color:var(--red);}
.predict-btn{width:100%;padding:16px;background:linear-gradient(135deg,var(--accent),var(--purple));border:none;border-radius:13px;color:#fff;font-family:'Cabinet Grotesk',sans-serif;font-size:15px;font-weight:700;cursor:pointer;transition:all .3s;letter-spacing:.03em;}
.predict-btn:hover{transform:translateY(-2px);box-shadow:0 12px 30px rgba(79,156,249,.25);}
.predict-btn:disabled{opacity:.5;cursor:not-allowed;transform:none;}
.result{display:none;margin-top:20px;border-radius:16px;padding:24px;border:1px solid;animation:fadeIn .5s ease;}
.result.placed{background:var(--green-dim);border-color:rgba(34,211,160,.3);}
.result.not-placed{background:var(--red-dim);border-color:rgba(255,107,107,.3);}
.res-top{display:flex;align-items:flex-start;gap:14px;margin-bottom:20px;}
.res-emoji{font-size:38px;line-height:1;}
.res-text h2{font-family:'Cabinet Grotesk',sans-serif;font-size:18px;font-weight:700;}
.res-text p{font-size:12px;color:var(--text2);margin-top:3px;}
.acc-badge{margin-left:auto;background:rgba(79,156,249,.1);border:1px solid rgba(79,156,249,.2);border-radius:8px;padding:4px 10px;font-size:11px;color:var(--accent);font-family:'Cabinet Grotesk',sans-serif;font-weight:600;white-space:nowrap;}
.prob-bars{display:grid;gap:10px;margin-bottom:20px;}
.prob-row{display:flex;flex-direction:column;gap:5px;}
.prob-label{display:flex;justify-content:space-between;font-size:11px;color:var(--muted);}
.prob-label span:last-child{font-weight:600;font-size:12px;color:var(--text);}
.bar-bg{background:var(--border);border-radius:99px;height:7px;overflow:hidden;}
.bar-fill{height:100%;border-radius:99px;transition:width 1.2s cubic-bezier(.16,1,.3,1);width:0;}
.bar-g{background:linear-gradient(90deg,#22d3a0,#6ee7b7);}
.bar-r{background:linear-gradient(90deg,#ff6b6b,#fca5a5);}
.tips-grid{display:grid;grid-template-columns:1fr 1fr;gap:10px;}
@media(max-width:560px){.tips-grid{grid-template-columns:1fr;}}
.tip-box{background:var(--card2);border-radius:12px;padding:13px;border:1px solid var(--border);}
.tip-box-title{font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:.1em;margin-bottom:9px;}
.tip-box.good .tip-box-title{color:var(--green);}
.tip-box.improve .tip-box-title{color:var(--amber);}
.tip-box.critical .tip-box-title{color:var(--red);}
.tip-item{display:flex;gap:8px;font-size:11px;color:var(--text2);margin-bottom:6px;line-height:1.5;align-items:flex-start;}
.tip-dot{width:5px;height:5px;border-radius:50%;margin-top:4px;flex-shrink:0;}
.tip-dot.g{background:var(--green);}.tip-dot.a{background:var(--amber);}.tip-dot.r{background:var(--red);}
.stat-cards{display:grid;grid-template-columns:repeat(3,1fr);gap:10px;margin-bottom:14px;}
@media(max-width:480px){.stat-cards{grid-template-columns:repeat(2,1fr);}}
.stat-card{background:var(--card2);border:1px solid var(--border);border-radius:12px;padding:13px;}
.stat-card-val{font-family:'Cabinet Grotesk',sans-serif;font-size:22px;font-weight:800;color:var(--accent);}
.stat-card-label{font-size:10px;color:var(--muted);text-transform:uppercase;letter-spacing:.06em;margin-top:2px;}
.analytics-grid{display:grid;grid-template-columns:1fr 1fr;gap:14px;margin-top:14px;}
@media(max-width:660px){.analytics-grid{grid-template-columns:1fr;}}
.mini-card{background:var(--card2);border:1px solid var(--border);border-radius:14px;padding:16px;}
.mini-title{font-size:12px;font-weight:600;color:var(--text2);margin-bottom:12px;}
.feat-bar{display:flex;flex-direction:column;gap:9px;}
.feat-row{display:flex;flex-direction:column;gap:4px;}
.feat-label{display:flex;justify-content:space-between;font-size:11px;color:var(--muted);}
.feat-label span:last-child{color:var(--accent);font-weight:600;}
.feat-bg{background:var(--border);border-radius:99px;height:5px;overflow:hidden;}
.feat-fill{height:100%;border-radius:99px;background:linear-gradient(90deg,var(--accent),var(--purple));transition:width 1s cubic-bezier(.16,1,.3,1);width:0;}
.upload-zone{border:2px dashed var(--border2);border-radius:16px;padding:44px 24px;text-align:center;cursor:pointer;transition:all .3s;background:var(--surface);}
.upload-zone:hover,.upload-zone.drag{border-color:var(--accent);background:var(--accent-dim);}
.upload-zone.has-file{border-color:var(--green);background:var(--green-dim);}
#fileInput{display:none;}
.upload-btn{margin-top:16px;width:100%;padding:14px;background:var(--card2);border:1px solid var(--border2);border-radius:12px;color:var(--accent);font-family:'Cabinet Grotesk',sans-serif;font-size:14px;font-weight:700;cursor:pointer;transition:all .2s;}
.upload-btn:hover{background:var(--accent-dim);}.upload-btn:disabled{opacity:.5;cursor:not-allowed;}
.resume-result{display:none;margin-top:16px;animation:fadeIn .4s ease;}
.score-ring-wrap{display:flex;justify-content:center;gap:28px;margin-bottom:20px;flex-wrap:wrap;}
.score-ring{text-align:center;}
.ring-val{font-family:'Cabinet Grotesk',sans-serif;font-size:26px;font-weight:800;}
.ring-label{font-size:10px;color:var(--muted);text-transform:uppercase;letter-spacing:.08em;margin-top:2px;}
.ring-val.good{color:var(--green);}.ring-val.ok{color:var(--amber);}.ring-val.bad{color:var(--red);}
.skills-chips{display:flex;flex-wrap:wrap;gap:5px;margin-bottom:10px;}
.chip{background:var(--card2);border:1px solid var(--border);border-radius:100px;padding:3px 10px;font-size:11px;color:var(--accent);}
.section-tags{display:flex;flex-wrap:wrap;gap:6px;}
.stag{background:var(--accent-dim);border:1px solid rgba(79,156,249,.2);border-radius:8px;padding:3px 9px;font-size:11px;color:var(--accent);}
.contact-row{display:flex;gap:8px;flex-wrap:wrap;margin-top:8px;}
.contact-chip{padding:4px 11px;border-radius:100px;font-size:11px;font-weight:500;}
.contact-chip.found{background:var(--green-dim);border:1px solid rgba(34,211,160,.3);color:var(--green);}
.contact-chip.missing{background:var(--red-dim);border:1px solid rgba(255,107,107,.3);color:var(--red);}
.list-items{display:flex;flex-direction:column;gap:7px;}
.list-item{display:flex;align-items:flex-start;gap:8px;font-size:11px;color:var(--text2);line-height:1.5;}
@media(max-width:560px){.card{padding:18px 14px;}.fields{grid-template-columns:1fr;}h1{font-size:1.9rem;}}
.spinner{display:inline-block;width:14px;height:14px;border:2px solid rgba(255,255,255,.3);border-top-color:#fff;border-radius:50%;animation:spin .6s linear infinite;vertical-align:middle;margin-right:6px;}
@keyframes spin{to{transform:rotate(360deg)}}
</style>
</head>
<body>
<div class="bg-mesh"></div><div class="bg-dots"></div>
<div class="wrap">
<header>
  <div class="logo"><div class="logo-dot">🎓</div>PlaceIQ — ML Placement Predictor</div>
  <h1>Predict Your<br/>Placement Odds</h1>
  <p class="sub">6 course-specific models · 20,000 students each · Resume AI · Analytics Dashboard</p>
</header>
<div class="tabs">
  <button class="tab active" onclick="switchTab('predict')">📊 Predict</button>
  <button class="tab" onclick="switchTab('analytics')">📈 Analytics</button>
  <button class="tab" onclick="switchTab('resume')">📄 Resume AI</button>
</div>

<!-- PREDICT -->
<div class="panel active" id="tab-predict">
  <div class="card">
    <div class="sec">Choose Your Course</div>
    <div class="course-grid" id="courseGrid"></div>
    <div id="formArea" style="display:none">
      <div id="formFields"></div>
      <button class="predict-btn" id="predictBtn" onclick="predict()">⚡ Predict My Placement Chances</button>
    </div>
  </div>
  <div class="result" id="resultPanel">
    <div class="res-top">
      <div class="res-emoji" id="resEmoji"></div>
      <div class="res-text"><h2 id="resTitle"></h2><p id="resSub"></p></div>
      <div class="acc-badge" id="accBadge"></div>
    </div>
    <div class="prob-bars">
      <div class="prob-row">
        <div class="prob-label"><span>[OK] Placement Probability</span><span id="pctP"></span></div>
        <div class="bar-bg"><div class="bar-fill bar-g" id="barG"></div></div>
      </div>
      <div class="prob-row">
        <div class="prob-label"><span>❌ Not Placed</span><span id="pctN"></span></div>
        <div class="bar-bg"><div class="bar-fill bar-r" id="barR"></div></div>
      </div>
    </div>
    <div id="tipsArea"></div>
  </div>
</div>

<!-- ANALYTICS -->
<div class="panel" id="tab-analytics">
  <div class="card" id="aEmpty">
    <div style="text-align:center;padding:40px 0;color:var(--muted)">
      <div style="font-size:44px;margin-bottom:14px">📊</div>
      <div style="font-family:'Cabinet Grotesk',sans-serif;font-size:15px;font-weight:600;color:var(--text2)">Run a prediction first</div>
      <div style="font-size:12px;margin-top:6px">Your analytics will appear here after predicting.</div>
    </div>
  </div>
  <div id="aCont" style="display:none"></div>
</div>

<!-- RESUME -->
<div class="panel" id="tab-resume">
  <div class="card">
    <div class="sec">AI Resume Analyzer</div>
    <div style="margin-bottom:14px">
      <label style="font-size:11px;color:var(--text2);text-transform:uppercase;letter-spacing:.06em;display:block;margin-bottom:7px">Course</label>
      <select id="resumeCourse" style="max-width:240px">
        <option>B.Tech</option><option>M.Tech</option><option>MCA</option>
        <option>BCA</option><option>MBA</option><option>BBA</option>
      </select>
    </div>
    <div class="upload-zone" id="uploadZone" onclick="document.getElementById('fileInput').click()"
         ondragover="event.preventDefault();this.classList.add('drag')"
         ondragleave="this.classList.remove('drag')" ondrop="handleDrop(event)">
      <div style="font-size:38px;margin-bottom:10px" id="uploadIcon">📎</div>
      <div style="font-size:14px;color:var(--text2)" id="uploadText">Drop resume here or click to browse</div>
      <div style="font-size:11px;color:var(--muted);margin-top:5px">PDF, DOCX, TXT · Max 5MB</div>
    </div>
    <input type="file" id="fileInput" accept=".pdf,.docx,.doc,.txt" onchange="fileSelected(event)"/>
    <button class="upload-btn" id="analyzeBtn" onclick="analyzeResume()" disabled>[*] Analyze Resume</button>
  </div>
  <div class="resume-result" id="resumeResult"></div>
</div>
</div>

<script>
const CI = COURSE_INFO_PLACEHOLDER;
const ICONS = {"B.Tech":"⚙️","M.Tech":"🔬","MCA":"💻","BCA":"🖥️","MBA":"📈","BBA":"🏢"};
const FD = {
  cgpa:{label:"CGPA (out of 10)",type:"number",min:4,max:10,step:0.01,placeholder:"e.g. 7.8"},
  backlogs:{label:"Backlogs / Arrears",type:"select",options:["0 — None","1","2","3","4+"],values:[0,1,2,3,4]},
  internships:{label:"Internships Done",type:"select",options:["0","1","2","3+"],values:[0,1,2,3]},
  projects:{label:"Projects Completed",type:"select",options:["0","1","2","3","4","5+"],values:[0,1,2,3,4,5]},
  hackathons:{label:"Hackathons",type:"select",options:["0","1","2","3+"],values:[0,1,2,3]},
  aptitude_score:{label:"Aptitude Score (%)",type:"range",min:20,max:100,vid:"aptV"},
  coding_score:{label:"Coding Skills (1–10)",type:"range",min:1,max:10,vid:"codV"},
  technical_score:{label:"Technical Skills (1–10)",type:"range",min:1,max:10,vid:"techV"},
  communication_score:{label:"Communication (1–10)",type:"range",min:1,max:10,vid:"comV"},
  soft_skills_score:{label:"Soft Skills (1–10)",type:"range",min:1,max:10,vid:"softV"},
  leadership_score:{label:"Leadership (1–10)",type:"range",min:1,max:10,vid:"leadV"},
  group_discussion_score:{label:"Group Discussion (1–10)",type:"range",min:1,max:10,vid:"gdV"},
  case_study_score:{label:"Case Study (1–10)",type:"range",min:1,max:10,vid:"csV"},
  workshops_certifications:{label:"Workshops / Certs",type:"select",options:["0","1","2","3","4+"],values:[0,1,2,3,4]},
  github_projects:{label:"GitHub Projects",type:"select",options:["0","1","2","3","4","5+"],values:[0,1,2,3,4,5]},
  competitive_programming:{label:"Competitive Programming",type:"select",options:["None","Beginner","Intermediate","Expert"],values:[0,1,2,3]},
  research_papers:{label:"Research Papers",type:"select",options:["0","1","2","3+"],values:[0,1,2,3]},
  industry_experience_months:{label:"Industry Exp. (months)",type:"select",options:["0","6","12","18","24","36+"],values:[0,6,12,18,24,36]},
  database_skills:{label:"Database Skills (1–10)",type:"range",min:1,max:10,vid:"dbV"},
  web_dev_skills:{label:"Web Dev (1–10)",type:"range",min:1,max:10,vid:"wdV"},
  work_experience_months:{label:"Work Exp. (months)",type:"select",options:["0","6","12","18","24","36","48+"],values:[0,6,12,18,24,36,48]},
  sports_achievements:{label:"Sports Achievements",type:"select",options:["None","College","State","National"],values:[0,1,2,3]},
  extracurricular:{label:"Extracurricular Activities",type:"toggle"},
  gender:{label:"Gender",type:"select",options:["Female","Male"],values:[0,1]},
};
const FG = {
  "B.Tech":[{t:"[#] Academic",f:["cgpa","backlogs","internships","projects","hackathons"]},{t:"💡 Technical",f:["coding_score","technical_score","aptitude_score","github_projects","competitive_programming"]},{t:"🤝 Soft Skills",f:["communication_score","soft_skills_score","workshops_certifications","extracurricular","gender"]}],
  "M.Tech":[{t:"[#] Academic",f:["cgpa","backlogs","internships","research_papers","industry_experience_months"]},{t:"💡 Technical",f:["technical_score","coding_score","aptitude_score","github_projects","projects"]},{t:"🤝 Soft Skills",f:["communication_score","soft_skills_score","workshops_certifications","extracurricular","gender"]}],
  "MCA":[{t:"[#] Academic",f:["cgpa","backlogs","internships","projects"]},{t:"💡 Technical",f:["coding_score","technical_score","aptitude_score","database_skills","web_dev_skills"]},{t:"🤝 Soft Skills",f:["communication_score","soft_skills_score","github_projects","workshops_certifications","extracurricular","gender"]}],
  "BCA":[{t:"[#] Academic",f:["cgpa","backlogs","internships","projects"]},{t:"💡 Technical",f:["coding_score","aptitude_score","web_dev_skills","database_skills","github_projects"]},{t:"🤝 Soft Skills",f:["communication_score","soft_skills_score","workshops_certifications","extracurricular","gender"]}],
  "MBA":[{t:"[#] Academic",f:["cgpa","backlogs","internships","work_experience_months"]},{t:"🗣️ Management",f:["communication_score","leadership_score","group_discussion_score","case_study_score","aptitude_score"]},{t:"🤝 Personality",f:["soft_skills_score","extracurricular","sports_achievements","workshops_certifications","gender"]}],
  "BBA":[{t:"[#] Academic",f:["cgpa","backlogs","internships","work_experience_months"]},{t:"🗣️ Management",f:["communication_score","leadership_score","group_discussion_score","case_study_score","aptitude_score"]},{t:"🤝 Personality",f:["soft_skills_score","extracurricular","sports_achievements","workshops_certifications","gender"]}],
};
const DEF = {cgpa:7.5,backlogs:0,internships:1,projects:2,hackathons:0,aptitude_score:65,coding_score:6,technical_score:6,communication_score:7,soft_skills_score:7,leadership_score:6,group_discussion_score:6,case_study_score:5,workshops_certifications:1,github_projects:1,competitive_programming:0,research_papers:0,industry_experience_months:0,database_skills:6,web_dev_skills:5,work_experience_months:0,sports_achievements:0,extracurricular:1,gender:1};

let sc="",ev=1,selFile=null;

function init(){
  const g=document.getElementById("courseGrid");
  for(const[c,i] of Object.entries(CI)){
    const b=document.createElement("button");b.className="course-btn";b.dataset.course=c;
    b.innerHTML=`<span class="ci">${ICONS[c]||"[#]"}</span>${c}<div class="cr">${i.placement_rate.toFixed(0)}% placed · ₹${i.avg_salary}L avg</div>`;
    b.onclick=()=>selectCourse(c);g.appendChild(b);
  }
}

function selectCourse(c){
  sc=c;
  document.querySelectorAll(".course-btn").forEach(b=>b.classList.toggle("selected",b.dataset.course===c));
  buildForm(c);
  document.getElementById("formArea").style.display="block";
  document.getElementById("resultPanel").style.display="none";
  document.getElementById("resumeCourse").value=c;
}

function buildForm(c){
  const gs=FG[c]||[];let h="";
  for(const g of gs){
    h+=`<div class="sec" style="margin-bottom:14px">${g.t}</div><div class="fields">`;
    for(const f of g.f) h+=buildField(f);
    h+=`</div>`;
  }
  document.getElementById("formFields").innerHTML=h;setExtra(ev);
}

function buildField(f){
  const d=FD[f];if(!d)return"";const id=`f_${f}`;let c="";
  if(d.type==="number") c=`<input type="number" id="${id}" min="${d.min}" max="${d.max}" step="${d.step||1}" placeholder="${d.placeholder||''}" value="${DEF[f]||0}"/>`;
  else if(d.type==="select"){const o=d.options.map((o,i)=>`<option value="${d.values[i]}"${d.values[i]==(DEF[f]||0)?'selected':''}>${o}</option>`).join("");c=`<select id="${id}">${o}</select>`;}
  else if(d.type==="range"){const v=DEF[f]||d.min;c=`<div class="sl-wrap"><input type="range" id="${id}" min="${d.min}" max="${d.max}" value="${v}" oninput="document.getElementById('${d.vid}').textContent=this.value"/><span class="sl-val" id="${d.vid}">${v}</span></div>`;}
  else if(d.type==="toggle") c=`<div class="tog"><button class="tog-btn" id="togY" onclick="setExtra(1)">✓ Yes</button><button class="tog-btn" id="togN" onclick="setExtra(0)">✗ No</button></div><input type="hidden" id="${id}" value="${DEF[f]||0}"/>`;
  return`<div class="field"><label>${d.label}</label>${c}</div>`;
}

function setExtra(v){
  ev=v;const h=document.getElementById("f_extracurricular"),ty=document.getElementById("togY"),tn=document.getElementById("togN");
  if(h)h.value=v;if(ty)ty.className="tog-btn"+(v===1?" on":"");if(tn)tn.className="tog-btn"+(v===0?" off":"");
}

async function predict(){
  if(!sc){alert("Please select a course first.");return;}
  const btn=document.getElementById("predictBtn");btn.disabled=true;
  btn.innerHTML=`<span class="spinner"></span>Analyzing...`;
  const p={course:sc};
  for(const f of CI[sc].features){const el=document.getElementById(`f_${f}`);p[f]=el?parseFloat(el.value):0;}
  try{
    const r=await fetch("/predict",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(p)});
    const d=await r.json();
    if(d.error){alert("Error: "+d.error);return;}
    showResult(d,p);showAnalytics(d.analytics,sc);
  }catch(e){alert("❌ Connection error — make sure the Flask server is running!\n\nRun: python run_app.py");}
  finally{btn.disabled=false;btn.innerHTML="⚡ Predict My Placement Chances";}
}

function showResult(d,inp){
  const pan=document.getElementById("resultPanel"),pl=d.placed===1;
  pan.className="result "+(pl?"placed":"not-placed");pan.style.display="block";
  document.getElementById("resEmoji").textContent=pl?"🎉":"📈";
  document.getElementById("resTitle").textContent=pl?"You're Likely to Get Placed!":"Profile Needs More Work";
  document.getElementById("resSub").textContent=`${d.placed_probability}% probability · ${sc} model`;
  document.getElementById("accBadge").textContent=`Accuracy: ${d.model_accuracy}%`;
  document.getElementById("pctP").textContent=d.placed_probability+"%";
  document.getElementById("pctN").textContent=d.not_placed_probability+"%";
  setTimeout(()=>{document.getElementById("barG").style.width=d.placed_probability+"%";document.getElementById("barR").style.width=d.not_placed_probability+"%";},100);
  const t=d.tips;let th=`<div class="tips-grid" style="margin-top:14px">`;
  if(t.good.length) th+=`<div class="tip-box good"><div class="tip-box-title">💚 Strengths</div><div class="list-items">${t.good.map(x=>`<div class="list-item"><div class="tip-dot g"></div>${x}</div>`).join("")}</div></div>`;
  if(t.improve.length) th+=`<div class="tip-box improve"><div class="tip-box-title">⚡ Improve</div><div class="list-items">${t.improve.map(x=>`<div class="list-item"><div class="tip-dot a"></div>${x}</div>`).join("")}</div></div>`;
  if(t.critical.length) th+=`<div class="tip-box critical" style="grid-column:1/-1"><div class="tip-box-title">🚨 Critical</div><div class="list-items">${t.critical.map(x=>`<div class="list-item"><div class="tip-dot r"></div>${x}</div>`).join("")}</div></div>`;
  th+=`</div>`;document.getElementById("tipsArea").innerHTML=th;
  pan.scrollIntoView({behavior:"smooth",block:"nearest"});
}

function showAnalytics(a,c){
  document.getElementById("aEmpty").style.display="none";
  const cont=document.getElementById("aCont");cont.style.display="block";
  const pc=a.probability>=70?"var(--green)":a.probability>=45?"var(--amber)":"var(--red)";
  const bm=a.benchmark;
  let h=`<div class="stat-cards"><div class="stat-card"><div class="stat-card-val" style="color:${pc}">${a.probability}%</div><div class="stat-card-label">Placement Prob.</div></div><div class="stat-card"><div class="stat-card-val">₹${a.avg_salary}L</div><div class="stat-card-label">Avg Package</div></div><div class="stat-card"><div class="stat-card-val">${a.placement_rate.toFixed(0)}%</div><div class="stat-card-label">Course Avg</div></div></div><div class="analytics-grid"><div class="mini-card"><div class="mini-title">📊 Feature Importance</div><div class="feat-bar" id="iBar"></div></div><div class="mini-card"><div class="mini-title">👤 Your Profile</div><div class="feat-bar" id="pBar"></div></div></div>`;
  if(bm&&Object.keys(bm).length) h+=`<div class="mini-card" style="margin-top:14px"><div class="mini-title">🏭 Industry Benchmarks — ${c}</div><div style="display:grid;grid-template-columns:repeat(auto-fill,minmax(160px,1fr));gap:12px;margin-top:10px"><div style="font-size:12px;color:var(--text2)">Min CGPA<br/><span style="font-family:'Cabinet Grotesk',sans-serif;font-size:18px;font-weight:700;color:var(--accent)">${bm.min_cgpa}</span></div><div style="font-size:12px;color:var(--text2)">Recommended<br/><span style="font-family:'Cabinet Grotesk',sans-serif;font-size:18px;font-weight:700;color:var(--green)">${bm.rec_cgpa}+</span></div><div style="font-size:12px;color:var(--text2)">Min Internships<br/><span style="font-family:'Cabinet Grotesk',sans-serif;font-size:18px;font-weight:700;color:var(--accent)">${bm.min_internships}</span></div><div style="font-size:12px;color:var(--text2)">Avg Package<br/><span style="font-family:'Cabinet Grotesk',sans-serif;font-size:18px;font-weight:700;color:var(--purple)">${bm.avg_package}</span></div></div></div>`;
  cont.innerHTML=h;
  document.getElementById("iBar").innerHTML=a.importance.slice(0,7).map(f=>`<div class="feat-row"><div class="feat-label"><span>${f.feature}</span><span>${f.importance}%</span></div><div class="feat-bg"><div class="feat-fill" data-w="${f.importance}" style="width:0"></div></div></div>`).join("");
  document.getElementById("pBar").innerHTML=a.score_breakdown.slice(0,7).map(f=>`<div class="feat-row"><div class="feat-label"><span>${f.feature}</span><span>${f.percent}%</span></div><div class="feat-bg"><div class="feat-fill" data-w="${f.percent}" style="width:0;background:${f.percent>=70?'linear-gradient(90deg,var(--green),#6ee7b7)':f.percent>=40?'linear-gradient(90deg,var(--amber),#fde68a)':'linear-gradient(90deg,var(--red),#fca5a5)'}"></div></div></div>`).join("");
  setTimeout(()=>{document.querySelectorAll(".feat-fill[data-w]").forEach(el=>el.style.width=el.dataset.w+"%");},100);
}

function fileSelected(e){
  selFile=e.target.files[0];if(!selFile)return;
  document.getElementById("uploadZone").classList.add("has-file");
  document.getElementById("uploadIcon").textContent="[OK]";
  document.getElementById("uploadText").textContent=selFile.name;
  document.getElementById("analyzeBtn").disabled=false;
}
function handleDrop(e){e.preventDefault();document.getElementById("uploadZone").classList.remove("drag");const f=e.dataTransfer.files[0];if(f){selFile=f;fileSelected({target:{files:[f]}});}}

async function analyzeResume(){
  if(!selFile){alert("Please select a file.");return;}
  const btn=document.getElementById("analyzeBtn");btn.disabled=true;btn.innerHTML=`<span class="spinner"></span>Analyzing...`;
  const course=document.getElementById("resumeCourse").value;
  const fd=new FormData();fd.append("resume",selFile);fd.append("course",course);
  try{
    const r=await fetch("/analyze-resume",{method:"POST",body:fd});
    const d=await r.json();
    if(d.error){alert("Error: "+d.error);return;}
    renderResume(d,course);
  }catch(e){alert("Connection error.");}
  finally{btn.disabled=false;btn.innerHTML="[*] Analyze Resume";}
}

function sc2(s){return s>=70?"good":s>=45?"ok":"bad";}

function renderResume(d,c){
  const el=document.getElementById("resumeResult");el.style.display="block";
  const allS=Object.values(d.found_skills||{}).flat();
  let sh="";
  for(const[cat,skills] of Object.entries(d.found_skills||{})){
    if(!skills.length)continue;
    const cl=cat.replace(/_/g," ").replace(/\b\w/g,x=>x.toUpperCase());
    sh+=`<div style="margin-bottom:12px"><div style="font-size:10px;color:var(--muted);text-transform:uppercase;letter-spacing:.08em;margin-bottom:5px">${cl}</div><div class="skills-chips">${skills.map(s=>`<div class="chip">${s}</div>`).join("")}</div></div>`;
  }
  const ct=d.contact_info||{};
  el.innerHTML=`<div class="card"><div class="sec">📄 Resume Analysis — ${c}</div>
  <div class="score-ring-wrap">
    <div class="score-ring"><div class="ring-val ${sc2(d.overall_score)}">${d.overall_score}/100</div><div class="ring-label">Overall</div></div>
    <div class="score-ring"><div class="ring-val ${sc2(d.course_fit_score)}">${d.course_fit_score}%</div><div class="ring-label">Course Fit</div></div>
    <div class="score-ring"><div class="ring-val ${sc2(d.skill_coverage)}">${d.skill_coverage}%</div><div class="ring-label">Skill Coverage</div></div>
    <div class="score-ring"><div class="ring-val" style="color:var(--accent)">${d.word_count}</div><div class="ring-label">Words</div></div>
  </div>
  <div style="margin-bottom:16px"><div class="mini-title" style="margin-bottom:7px">📋 Sections Detected</div><div class="section-tags">${(d.detected_sections||[]).map(s=>`<div class="stag">${s.charAt(0).toUpperCase()+s.slice(1)}</div>`).join("")||"<span style='color:var(--red);font-size:12px'>No standard sections found</span>"}</div></div>
  <div style="margin-bottom:16px"><div class="mini-title" style="margin-bottom:6px">📞 Contact Info</div><div class="contact-row">${["email","phone","linkedin","github"].map(k=>`<div class="contact-chip ${ct[k]?"found":"missing"}">${ct[k]?"✓":"✗"} ${k.charAt(0).toUpperCase()+k.slice(1)}</div>`).join("")}</div></div>
  ${sh?`<div style="margin-bottom:16px"><div class="mini-title" style="margin-bottom:8px">🛠️ Skills Found (${allS.length})</div>${sh}</div>`:""}
  <div class="tips-grid">
    ${d.strengths&&d.strengths.length?`<div class="tip-box good"><div class="tip-box-title">💚 Strengths</div><div class="list-items">${d.strengths.map(x=>`<div class="list-item"><div class="tip-dot g"></div>${x}</div>`).join("")}</div></div>`:""}
    ${d.gaps&&d.gaps.length?`<div class="tip-box improve"><div class="tip-box-title">⚠️ Gaps to Fix</div><div class="list-items">${d.gaps.map(x=>`<div class="list-item"><div class="tip-dot a"></div>${x}</div>`).join("")}</div></div>`:""}
  </div>
  ${d.ats_missing&&d.ats_missing.length?`<div style="margin-top:14px;background:rgba(251,191,36,.06);border:1px solid rgba(251,191,36,.2);border-radius:12px;padding:13px"><div class="tip-box-title" style="color:var(--amber);margin-bottom:7px">🤖 Missing ATS Keywords</div><div class="skills-chips">${d.ats_missing.map(k=>`<div class="chip" style="border-color:rgba(251,191,36,.3);color:var(--amber)">${k}</div>`).join("")}</div></div>`:""}
  </div>`;
  el.scrollIntoView({behavior:"smooth",block:"nearest"});
}

function switchTab(n){
  const ts=["predict","analytics","resume"];
  document.querySelectorAll(".tab").forEach((t,i)=>t.classList.toggle("active",ts[i]===n));
  document.querySelectorAll(".panel").forEach(p=>p.classList.toggle("active",p.id===`tab-${n}`));
}
init();
</script>
</body>
</html>"""

# Inject course info into HTML
HTML_RENDERED = HTML.replace("COURSE_INFO_PLACEHOLDER", COURSE_INFO_JSON)

@app.route('/')
def index():
    return render_template_string(HTML_RENDERED)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        course = data.get('course')
        if course not in MODELS:
            return jsonify({'error': f'Unknown course: {course}'}), 400
        features = COURSE_CONFIG[course]["features"]
        X = np.array([float(data.get(f, 0)) for f in features]).reshape(1, -1)
        model = MODELS[course]
        pred = model.predict(X)[0]
        proba = model.predict_proba(X)[0]
        pp = round(float(proba[1])*100, 1)
        np_ = round(float(proba[0])*100, 1)
        inp_dict = {f: float(data.get(f, 0)) for f in features}
        analytics = get_analytics(course, inp_dict, pp)
        tips = generate_tips(course, inp_dict, pp)
        return jsonify({'placed':int(pred),'placed_probability':pp,'not_placed_probability':np_,
                        'message':'Likely Placed!' if pred==1 else 'Needs Improvement',
                        'analytics':analytics,'tips':tips,'model_accuracy':ACCURACIES[course]})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/analyze-resume', methods=['POST'])
def analyze_resume_route():
    try:
        if 'resume' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        file = request.files['resume']
        course = request.form.get('course', 'B.Tech')
        if not file.filename:
            return jsonify({'error': 'No file selected'}), 400
        ext = os.path.splitext(file.filename)[1].lower()
        if ext not in ['.pdf','.docx','.doc','.txt']:
            return jsonify({'error': 'Use PDF, DOCX, or TXT files only'}), 400
        fname = secure_filename(file.filename)
        fpath = os.path.join(UPLOAD_FOLDER, fname)
        file.save(fpath)
        result = analyze_resume(fpath, course)
        try: os.remove(fpath)
        except: pass
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    print("\n" + "="*55)
    print("  [web] Open in browser: http://localhost:5000")
    print("="*55 + "\n")
    app.run(debug=False, host='0.0.0.0', port=5000)
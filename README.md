# 🚀 AI-Powered SEM Keyword Research Platform

**Enterprise-grade keyword research and competitive intelligence platform** powered by advanced NLP, semantic clustering, and Google Gemini AI.

Transform any website into a complete Google Ads campaign strategy with competitor analysis, budget optimization, and performance forecasting.

## 🎯 Key Features

### **Core Intelligence**
- 🕵️ **Competitor Analysis** - Deep website intelligence with Gemini AI
- 💰 **Budget Optimization** - Smart allocation across Search/Shopping/PMax campaigns  
- 📊 **Performance Forecasting** - ROAS predictions and conversion estimates
- 🎯 **Intent Classification** - Transactional vs informational keyword categorization

### **Advanced NLP Pipeline**
- 🧠 **Semantic Clustering** - Groups related keywords using Sentence-BERT embeddings
- 🔍 **Multi-source Expansion** - WordStream + Google Autocomplete + NER + KeyBERT
- 🎨 **Smart Deduplication** - Fuzzy matching removes near-duplicates
- 📈 **Multi-factor Scoring** - Volume + CPC + Intent + Competition analysis

### **Professional Outputs**
- 📋 **Google Ads Ready CSV** - Import directly into Google Ads Editor
- 📊 **Complete JSON Dataset** - All keyword data with analytics
- 📝 **Competitor Intelligence Reports** - Strategic insights in markdown
- 💡 **Performance Max Themes** - Asset group suggestions with copy
- 🎯 **Budget Allocation Reports** - ROI-optimized spend recommendations

## 🚀 Quick Start (5 Minutes)

### **Prerequisites**
- ✅ **Python 3.10+** (Download from [python.org](https://python.org))
- ✅ **Git** (For cloning - optional)
- ✅ **8GB+ RAM** (For NLP models)

### **Step 1: Environment Setup**
```bash
# Create virtual environment
python -m venv venv

# Activate environment
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Linux/Mac

# Upgrade pip
python -m pip install --upgrade pip
```

### **Step 2: Install Dependencies**
```bash
# Install all required packages
pip install -r requirements.txt

# This includes:
# - Advanced NLP: spacy, sentence-transformers, keybert
# - AI Integration: google-generativeai, openai
# - Data Processing: pandas, numpy, scikit-learn
# - Web Scraping: requests, beautifulsoup4
```

### **Step 3: Configure API Keys (REQUIRED for AI features)**

Edit `config.yaml` and add your API keys:

```yaml
# Basic LLM (Optional - fallback to rule-based)
llm:
  provider: "openai"
  model: "gpt-4o-mini"
  # Get key from: https://platform.openai.com/api-keys

# Gemini AI (REQUIRED for competitor analysis & budget optimization)
gemini:
  api_key: "YOUR_GEMINI_API_KEY_HERE"
  model: "gemini-2.0-flash-exp"
  # Get key from: https://makersuite.google.com/app/apikey
```

**⚠️ IMPORTANT**: Without Gemini API key, you'll only get basic keyword research (no competitor analysis or budget optimization).

### **Step 4: Customize Your Research**

Edit `config.yaml` for your business:

```yaml
website: "https://your-website.com"
competitors:
  - "https://competitor1.com"
  - "https://competitor2.com"
budget:
  total_monthly: 10000  # Your monthly ad budget
  conversion_rate: 0.02 # Expected conversion rate
business:
  industry: "your-industry"
  description: "Brief description of your business"
```

### **Step 5: Verify Setup (Recommended)**
```bash
# Run setup verification test
python test_setup.py

# Should show: "🎉 Setup successful! Ready to use."
```

### **Step 6: Run Analysis**
```bash
# Full analysis with AI features
python run.py --config config.yaml

# Test run without external calls
python run.py --config config.yaml --dry-run

# Debug mode for troubleshooting
python run.py --config config.yaml --debug
```

## 📊 What You Get (Sample Output)

### **🎯 Professional Reports Generated:**
```
outputs/run-20250808-094037/
├── 📋 search_adgroups.csv          # Google Ads ready keywords
├── 📊 keyword_data.json            # Complete dataset with analytics  
├── 🕵️ competitor_analysis.md       # Strategic competitor intelligence
├── 💰 budget_optimization.md       # ROI-focused budget recommendations
└── 🎨 pmax_themes.md              # Performance Max asset themes
```

### **📋 Sample Keywords Output (What Employers See):**

**File: `search_adgroups.csv` (Google Ads Ready)**
```csv
ad_group,keyword,intent,match_type,volume,cpc_low,cpc_high,competition,score
Brand Terms,your company name,navigational,exact,1500,0.50,1.20,0.25,0.92
Product Keywords,your main product,transactional,phrase,2100,1.80,4.50,0.73,0.85
Industry Terms,industry solution,commercial_investigation,broad,1200,0.90,2.30,0.58,0.78
Competitor Terms,competitor alternative,transactional,phrase,800,2.10,5.20,0.82,0.71
```

**What this gives you:**
- ✅ **Ready-to-use keywords** for Google Ads campaigns
- ✅ **Intent classification** - Know which keywords drive sales vs research
- ✅ **Match type suggestions** - Exact/Phrase/Broad recommendations  
- ✅ **Cost estimates** - CPC ranges for budget planning
- ✅ **Competition analysis** - How difficult each keyword is to rank for

### **🕵️ Sample Competitor Analysis:**
```markdown
## Competitor: GitLab
### 🎯 Value Propositions:
- Complete DevOps platform
- Built-in CI/CD pipelines  
- Security-first approach

### 🔍 Keyword Opportunities:
- "gitlab alternative"
- "devops automation platform"
- "enterprise git hosting"
```

### **💰 Sample Budget Optimization:**
```markdown
## 🎯 Recommended Budget Allocation
### Search Campaign: $5,000 (50%)
- High-intent keywords with precise targeting

### Shopping Campaign: $3,000 (30%) 
- Product showcase with visual appeal

### Performance Max: $2,000 (20%)
- Broad reach across Google properties

## 📈 Performance Forecast
- Estimated Clicks: 4,000
- Estimated Conversions: 80  
- Estimated ROAS: 4.0x
```

## ⚠️ Common Setup Issues & Solutions

### **Issue 1: Python Version**
```bash
# Check version
python --version  # Must be 3.10+

# If too old, download from python.org
# Or use pyenv/conda to manage versions
```

### **Issue 2: Virtual Environment Problems**
```bash
# Windows PowerShell execution policy
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# Alternative activation (Windows)
venv\Scripts\Activate.ps1

# Permission issues (Linux/Mac)
chmod +x venv/bin/activate
```

### **Issue 3: Package Installation Failures**
```bash
# Update pip first
python -m pip install --upgrade pip setuptools wheel

# Install with verbose output to see errors
pip install -r requirements.txt -v

# Missing Visual C++ (Windows)
# Download from: https://visualstudio.microsoft.com/visual-cpp-build-tools/
```

### **Issue 4: Memory Issues**
```bash
# Reduce model size in code (if needed)
# Or increase system RAM to 8GB+
# Close other applications during analysis
```

### **Issue 5: SSL Certificate Errors**
- The system handles most SSL issues automatically
- If you see certificate errors, they're usually non-blocking
- Check logs for "SSL" warnings but system continues

### **Issue 6: Invalid API Key Error**
```bash
ERROR - Gemini generation failed: 400 API key not valid
```
**Solution:**
1. Get valid Gemini API key from: https://makersuite.google.com/app/apikey
2. Replace `YOUR_GEMINI_API_KEY_HERE` in `config.yaml` with actual key
3. Or set environment variable: `export GEMINI_API_KEY=your_actual_key`

**Note:** System still works without API key, but you'll miss competitor analysis and budget optimization features.

### **Issue 7: Windows Unicode Display Problems**
```bash
UnicodeEncodeError: 'charmap' codec can't encode character
```
**Solution:** 
- System works fine, just emoji display issues
- Output shows `[SUCCESS]` instead of `✅`
- All functionality remains intact
- Optional: Set `PYTHONIOENCODING=utf-8` environment variable

### **Issue 8: API Rate Limits**  
- WordStream calls are rate-limited (1 req/2sec)
- System has built-in delays and retry logic
- If blocked, wait 5 minutes and retry

## 🔧 Advanced Configuration

### **Full `config.yaml` Reference:**
```yaml
# Target Analysis
website: "https://your-company.com"
competitors:
  - "https://competitor1.com" 
  - "https://competitor2.com"
locations: ["United States", "Canada"]
language: "en"
seeds: ["manual keyword 1", "manual keyword 2"]
max_keywords: 2000

# Quality Filters  
filters:
  min_search_volume: 100
  max_cpc: 20.0

# Data Sources
enrichers:
  use_wordstream: true
  use_serpapi: false
  serpapi_key: ""

# AI Configuration
llm:
  provider: "openai"
  model: "gpt-4o-mini" 
  max_tokens: 800

gemini:
  api_key: "YOUR_GEMINI_API_KEY"
  model: "gemini-2.0-flash-exp"
  max_tokens: 8192

# Business Context (for AI analysis)
budget:
  total_monthly: 10000
  conversion_rate: 0.02
  target_roas: 4.0

business:
  industry: "software"
  description: "AI-powered workflow automation"
  target_audience: ["business professionals", "tech teams"]

# Export Options
export:
  csv: true
  json: true
  googleads_json: false

# System Settings
output_dir: "./outputs"
cache_dir: "./cache"
logging:
  level: "INFO"
```

## 🚀 Ready for Production

This system is **enterprise-ready** with:
- ✅ **Error handling** - Graceful failures and retries
- ✅ **Caching system** - Avoids redundant API calls  
- ✅ **Rate limiting** - Respects service limits
- ✅ **Structured logging** - Full audit trail
- ✅ **Modular design** - Easy to extend and maintain

**Perfect for agencies, consultants, and businesses needing professional SEM intelligence.** 🎯
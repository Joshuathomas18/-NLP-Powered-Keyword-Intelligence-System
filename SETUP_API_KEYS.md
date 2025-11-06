# API Keys Setup Guide

## Quick Setup

### 1. Create `.env` file in project root

Create a file named `.env` in the same directory as `config.yaml`:

```bash
# OpenAI API Key (for LLM features)
OPENAI_API_KEY=sk-your-actual-openai-key-here

# Google Gemini API Key (optional, for competitor analysis)
GEMINI_API_KEY=your-actual-gemini-key-here

# SERP API Key (optional, for SERP data)
SERPAPI_KEY=your-serpapi-key-here
```

### 2. Get Your API Keys

**OpenAI API Key:**
1. Go to: https://platform.openai.com/api-keys
2. Sign in or create account
3. Click "Create new secret key"
4. Copy the key (starts with `sk-`)
5. Add to `.env` file: `OPENAI_API_KEY=sk-...`

**Gemini API Key (Optional):**
1. Go to: https://makersuite.google.com/app/apikey
2. Sign in with Google account
3. Click "Create API Key"
4. Copy the key
5. Add to `.env` file: `GEMINI_API_KEY=...`

### 3. Verify Setup

After creating `.env` file, restart your Streamlit app:
```bash
python -m streamlit run app.py
```

You should see:
- ✅ No "No OpenAI API key found" warnings
- ✅ LLM features working (intent classification, ad group naming)
- ✅ Competitor analysis working (if Gemini key added)

### 4. What Happens Without API Keys?

**Without OpenAI Key:**
- System uses **rule-based fallbacks** for:
  - Intent classification
  - Ad group naming
  - Match type suggestions
- Still works, but less accurate

**Without Gemini Key:**
- Competitor analysis skipped
- Budget optimization skipped
- Other features still work

## Important Notes

- ⚠️ **Never commit `.env` file to git** (it's already in .gitignore)
- ✅ Keep your API keys secret
- ✅ `.env` file loads automatically when app starts
- ✅ Keys are read from environment variables

## Troubleshooting

**If you still see "No OpenAI API key found":**
1. Make sure `.env` file is in project root (same folder as `app.py`)
2. Check file name is exactly `.env` (not `.env.txt`)
3. Restart Streamlit app after creating `.env`
4. Check no extra spaces in `.env` file: `OPENAI_API_KEY=sk-...` (no spaces)

**If Gemini key shows "API key not valid":**
1. Make sure you copied the full key
2. Check for any extra spaces or quotes
3. Get a new key from Google AI Studio if needed


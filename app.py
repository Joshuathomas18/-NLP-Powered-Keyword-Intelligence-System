"""
Streamlit Web UI for NLP-Powered SEM Keyword Intelligence System
Interactive workflow with background model loading and real-time pipeline execution.
"""

import streamlit as st
import json
import pandas as pd
import plotly.express as px
from pathlib import Path
from typing import Dict, List, Optional
import sys
import yaml
import time

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv not installed, skip

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from src.utils.config import Config
from src.utils.cache import CacheManager
from src.ui.model_loader import ModelLoader
from src.ui.pipeline_runner import PipelineRunner

# Page config
st.set_page_config(
    page_title="SEM Keyword Intelligence Dashboard",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 600;
        color: #1a1a1a;
        margin-bottom: 2rem;
        letter-spacing: -0.5px;
    }
    .status-ready {
        color: #16a34a;
        font-weight: 500;
        font-size: 0.95rem;
    }
    .status-loading {
        color: #ea580c;
        font-weight: 500;
        font-size: 0.95rem;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #6b7280;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        font-weight: 500;
    }
    .section-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #1a1a1a;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    </style>
""", unsafe_allow_html=True)


@st.cache_data
def load_keyword_data(json_path: str) -> Optional[Dict]:
    """Load keyword data from JSON file."""
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None


def get_all_runs() -> List[Path]:
    """Get all run directories."""
    output_dir = Path("outputs")
    if not output_dir.exists():
        return []
    return sorted([d for d in output_dir.iterdir() if d.is_dir() and d.name.startswith("run-")], 
                  reverse=True)


def extract_keywords_df(data: Dict) -> pd.DataFrame:
    """Extract all keywords into DataFrame."""
    keywords = []
    for ad_group in data.get('ad_groups', []):
        for kw in ad_group.get('keywords', []):
            kw_data = kw.copy()
            kw_data['ad_group'] = ad_group.get('ad_group_name', 'Unknown')
            keywords.append(kw_data)
    if keywords:
        return pd.DataFrame(keywords)
    return pd.DataFrame()


def load_config(website: str, max_keywords: int = 2000, dry_run: bool = False) -> Config:
    """Load configuration with custom website."""
    # Load base config
    config = Config.from_yaml("config.yaml")
    
    # Override with user input
    config.website = website
    config.max_keywords = max_keywords
    
    # Validate
    config.validate()
    
    return config


def display_results(result_data: Dict, run_id: str):
    """Display pipeline results with charts and data."""
    st.success("Analysis Complete")
    
    # Load keyword data
    json_path = Path(result_data['output_path']) / "keyword_data.json"
    if not json_path.exists():
        st.error("Results file not found")
        return
    
    data = load_keyword_data(str(json_path))
    if data is None:
        return
    
    df = extract_keywords_df(data)
    if df.empty:
        st.warning("No keyword data to display")
        return
    
    # Key metrics
    st.header("Results Summary")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Keywords", len(df))
    
    with col2:
        total_volume = df['volume'].sum()
        st.metric("Total Monthly Volume", f"{total_volume:,}")
    
    with col3:
        avg_cpc = df['cpc_high'].mean()
        st.metric("Average CPC", f"${avg_cpc:.2f}")
    
    with col4:
        st.metric("Ad Groups", len(data.get('ad_groups', [])))
    
    # Display static charts - Only Top 3 Most Relevant
    st.header("Visualization Charts")
    
    chart_dir = Path(f"charts/{run_id}")
    
    # Only the top 3 most relevant charts for SEM keyword intelligence
    chart_files = [
        ("Volume Distribution", "volume_distribution.png", "Search volume analysis - most critical metric for keyword selection"),
        ("CPC Analysis", "cpc_analysis.png", "Cost per click analysis - essential for budget planning"),
        ("Intent Analysis", "intent_analysis.png", "Intent classification - critical for targeting strategy")
    ]
    
    if chart_dir.exists():
        charts_found = 0
        for title, filename, description in chart_files:
            chart_path = chart_dir / filename
            if chart_path.exists():
                st.subheader(title)
                st.caption(description)
                st.image(str(chart_path), width='stretch')
                charts_found += 1
        
        if charts_found == 0:
            st.warning("Chart files not found. Charts will be generated after the next pipeline run.")
        elif charts_found < 3:
            st.info(f"Showing {charts_found} of 3 charts. Remaining charts will be generated shortly.")
    else:
        st.info("Charts are being generated... Please refresh in a moment.")
    
    # Keywords table
    st.header("Keywords")
    display_cols = ['keyword', 'ad_group', 'volume', 'cpc_high', 'competition', 
                   'intent', 'match_type', 'score']
    st.dataframe(df[display_cols], width='stretch', hide_index=True)
    
    # Ad groups summary
    st.header("Ad Groups")
    ad_groups_data = []
    for ad_group in data.get('ad_groups', []):
        keywords = ad_group.get('keywords', [])
        if keywords:
            kw_df = pd.DataFrame(keywords)
            ad_groups_data.append({
                'Ad Group': ad_group.get('ad_group_name', 'Unknown'),
                'Keywords': len(keywords),
                'Total Volume': ad_group.get('total_volume', 0),
                'Avg CPC': kw_df['cpc_high'].mean(),
                'Avg Score': kw_df['score'].mean()
            })
    
    if ad_groups_data:
        ag_df = pd.DataFrame(ad_groups_data)
        st.dataframe(ag_df, width='stretch', hide_index=True)
    
    # Download buttons
    st.header("Download Results")
    col1, col2 = st.columns(2)
    
    with col1:
        csv_path = Path(result_data['output_path']) / "search_adgroups.csv"
        if csv_path.exists():
            with open(csv_path, 'rb') as f:
                st.download_button("Download CSV", f.read(), 
                                 f"keywords_{run_id}.csv", "text/csv")
    
    with col2:
        if json_path.exists():
            with open(json_path, 'rb') as f:
                st.download_button("Download JSON", f.read(), 
                                 f"keywords_{run_id}.json", "application/json")


def main():
    """Main application."""
    # Initialize session state
    if 'model_loader' not in st.session_state:
        st.session_state.model_loader = ModelLoader()
        # Start background initialization immediately
        st.session_state.model_loader.initialize_models(background=True)
    
    if 'pipeline_running' not in st.session_state:
        st.session_state.pipeline_running = False
    
    if 'pipeline_result' not in st.session_state:
        st.session_state.pipeline_result = None
    
    # Header
    st.markdown('<h1 class="main-header">SEM Keyword Intelligence Dashboard</h1>', 
                unsafe_allow_html=True)
    
    # Model initialization check
    model_loader = st.session_state.model_loader
    model_status = model_loader.get_model_status()
    
    # Refresh status if loading
    if model_status['status'] == 'loading':
        model_status = model_loader.get_model_status()
    
    # Sidebar
    with st.sidebar:
        st.header("System Status")
        
        if model_status['status'] == 'ready':
            st.markdown('<p class="status-ready">Models Ready</p>', unsafe_allow_html=True)
            # Show model details
            with st.expander("Model Details"):
                progress = model_status['progress']
                for model_name, loaded in progress.items():
                    status = "Ready" if loaded else "Not Available"
                    st.write(f"{model_name.replace('_', ' ').title()}: {status}")
        elif model_status['status'] == 'loading':
            progress = model_loader.get_progress_percentage()
            st.progress(progress)
            
            current_model = model_status.get('current_model', 'Unknown')
            if current_model:
                st.markdown(f'<p class="status-loading">Loading: {current_model.replace("_", " ").title()} ({int(progress*100)}%)</p>', 
                           unsafe_allow_html=True)
            else:
                st.markdown(f'<p class="status-loading">Loading Models ({int(progress*100)}%)</p>', 
                           unsafe_allow_html=True)
            
            st.info("Models loading in background. You can start an analysis once ready.")
            
            # Add refresh button to update logs manually
            if st.button("Refresh Status", type="secondary"):
                st.rerun()
            
            # Show progress per model
            with st.expander("Loading Progress"):
                progress_dict = model_status['progress']
                for model_name, loaded in progress_dict.items():
                    if loaded:
                        st.write(f"{model_name.replace('_', ' ').title()}: Ready")
                    elif model_name == current_model:
                        st.write(f"{model_name.replace('_', ' ').title()}: Loading...")
                    else:
                        st.write(f"{model_name.replace('_', ' ').title()}: Pending")
        else:
            st.error("Model initialization failed")
            if model_status.get('error'):
                st.error(f"Error: {model_status['error']}")
        
        # Logs section
        if model_status['status'] == 'loading':
            with st.expander("Loading Logs", expanded=True):
                logs = model_status.get('logs', [])
                if logs:
                    log_text = "\n".join(logs[-30:])  # Show last 30 logs
                    st.code(log_text, language="text")
                else:
                    st.info("Waiting for logs...")
            
            # Troubleshooting tips
            with st.expander("Troubleshooting", expanded=False):
                st.markdown("""
                **If models are stuck loading:**
                1. Check the logs above to see which model is loading
                2. Sentence-BERT model download can take 2-5 minutes on first run
                3. If spaCy fails, install it: `python -m spacy download en_core_web_sm`
                4. If KeyBERT fails, it may need additional dependencies
                5. Check your internet connection (models download automatically)
                
                **Common issues:**
                - Slow internet → Model downloads take longer
                - Missing dependencies → Check error messages in logs
                - Disk space → Models need ~500MB free space
                """)
        elif model_status.get('logs'):
            with st.expander("Loading Logs", expanded=False):
                logs = model_status['logs']
                if logs:
                    log_text = "\n".join(logs[-20:])  # Show last 20 logs
                    st.code(log_text, language="text")
        
        st.header("Previous Runs")
        runs = get_all_runs()
        if runs:
            run_names = [run.name for run in runs]
            selected_run = st.selectbox("Select Run", run_names, index=0)
            
            if st.button("View Selected Run"):
                st.session_state.view_run = selected_run
                st.rerun()
    
    # Main tabs
    tabs = st.tabs(["Run Analysis", "View Results", "Charts", "Keywords", "Ad Groups"])
    
    with tabs[0]:  # Run Analysis
        st.header("Run Keyword Analysis")
        
        # Check if models are ready
        if model_status['status'] != 'ready':
            st.warning("Models are still loading. Please wait...")
            st.info("Models will be ready shortly. You can start an analysis once the status shows 'Ready'.")
            return
        
        # Configuration form
        with st.form("analysis_form"):
            website = st.text_input("Website URL", value="https://www.github.com", 
                                  help="Enter the website URL to analyze")
            
            col1, col2 = st.columns(2)
            
            with col1:
                max_keywords = st.number_input("Max Keywords", min_value=100, max_value=10000, 
                                              value=2000, step=100)
            
            with col2:
                dry_run = st.checkbox("Dry Run Mode (No external API calls)", value=False,
                                     help="Skip external API calls for faster testing")
            
            submitted = st.form_submit_button("Start Analysis", type="primary")
        
        # Check if pipeline is already running
        if st.session_state.pipeline_running:
            st.info("Pipeline is running...")
            return
        
        # Run pipeline
        if submitted:
            try:
                # Load config
                config = load_config(website, max_keywords, dry_run)
                cache_manager = CacheManager(config.cache_dir)
                
                # Create pipeline runner
                runner = PipelineRunner(config, cache_manager, dry_run=dry_run)
                
                # Progress display
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                result_data = None
                
                # Run pipeline with progress
                for update in runner.run_with_progress():
                    progress = update.get('progress', 0) / 100.0
                    phase_name = update.get('phase_name', '')
                    message = update.get('message', '')
                    phase_num = update.get('phase', 0)
                    
                    progress_bar.progress(progress)
                    status_text.markdown(f"**Phase {phase_num}/15: {phase_name}**\n\n{message}")
                    
                    if update.get('status') == 'complete':
                        result_data = update.get('result')
                        st.session_state.pipeline_result = result_data
                        st.session_state.pipeline_running = False
                        break
                    elif update.get('status') == 'error':
                        st.error(f"{update.get('message', 'Pipeline failed')}")
                        if update.get('error'):
                            st.error(f"Error: {update.get('error')}")
                        st.session_state.pipeline_running = False
                        st.stop()
                
                # Store result - will display on next rerun
                if result_data:
                    st.session_state.pipeline_running = False
                    st.success("Pipeline completed successfully")
                    st.rerun()
            
            except Exception as e:
                st.error(f"Pipeline failed: {str(e)}")
                import traceback
                st.code(traceback.format_exc())
                st.session_state.pipeline_running = False
        
        # Display results if available
        if st.session_state.pipeline_result:
            st.divider()
            result = st.session_state.pipeline_result
            display_results(result, result['run_id'])
    
    # Display results from previous run if selected
    if 'view_run' in st.session_state:
        with tabs[1]:  # View Results
            selected_run = st.session_state.view_run
            selected_path = Path("outputs") / selected_run
            json_path = selected_path / "keyword_data.json"
            
            if json_path.exists():
                data = load_keyword_data(str(json_path))
                if data:
                    df = extract_keywords_df(data)
                    if not df.empty:
                        display_results({
                            'output_path': str(selected_path),
                            'run_id': selected_run
                        }, selected_run)
                    else:
                        st.warning("No data in selected run.")
                else:
                    st.error("Failed to load data.")
            else:
                st.error("Run data not found.")
    
    # Legacy tabs for viewing existing runs
    runs = get_all_runs()
    if runs:
        run_names = [run.name for run in runs]
        selected_run = run_names[0] if run_names else None
        
        if selected_run:
            selected_path = Path("outputs") / selected_run
            json_path = selected_path / "keyword_data.json"
            
            if json_path.exists():
                data = load_keyword_data(str(json_path))
                if data:
                    df = extract_keywords_df(data)
                    
                    if not df.empty:
                        with tabs[2]:  # Charts
                            st.header("Interactive Charts")
                            
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.subheader("Intent Distribution")
                                intent_counts = df['intent'].value_counts()
                                fig = px.pie(intent_counts, values=intent_counts.values, names=intent_counts.index,
                                            color_discrete_map={
                                                'transactional': '#e74c3c',
                                                'commercial': '#3498db',
                                                'informational': '#2ecc71',
                                                'navigational': '#f39c12'
                                            })
                                st.plotly_chart(fig, width='stretch')
                            
                            with col2:
                                st.subheader("Volume Distribution")
                                fig = px.histogram(df, x='volume', nbins=30)
                                st.plotly_chart(fig, width='stretch')
                        
                        with tabs[3]:  # Keywords
                            st.header("Keywords")
                            st.dataframe(df[['keyword', 'ad_group', 'volume', 'cpc_high', 
                                            'competition', 'intent', 'match_type', 'score']], 
                                        width='stretch', hide_index=True)
                        
                        with tabs[4]:  # Ad Groups
                            st.header("Ad Groups")
                            ad_groups_data = []
                            for ad_group in data.get('ad_groups', []):
                                keywords = ad_group.get('keywords', [])
                                if keywords:
                                    kw_df = pd.DataFrame(keywords)
                                    ad_groups_data.append({
                                        'Ad Group': ad_group.get('ad_group_name', 'Unknown'),
                                        'Keywords': len(keywords),
                                        'Total Volume': ad_group.get('total_volume', 0),
                                        'Avg CPC': kw_df['cpc_high'].mean(),
                                        'Avg Score': kw_df['score'].mean()
                                    })
                            
                            if ad_groups_data:
                                ag_df = pd.DataFrame(ad_groups_data)
                                st.dataframe(ag_df, width='stretch', hide_index=True)


if __name__ == "__main__":
    main()

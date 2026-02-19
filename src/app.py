"""
app.py
------
Streamlit UI for OIC-LogLens — AI-Powered Error Resolution Engine
"""

import streamlit as st
import requests
import json
from datetime import datetime

# API Configuration
API_BASE_URL = "http://localhost:8000"

# Page Configuration
st.set_page_config(
    page_title="OIC-LogLens",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1F4E79;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        margin-bottom: 2rem;
    }
    .success-box {
        padding: 1rem;
        background-color: #D4EDDA;
        border-left: 4px solid #28A745;
        border-radius: 4px;
        margin: 1rem 0;
    }
    .error-box {
        padding: 1rem;
        background-color: #F8D7DA;
        border-left: 4px solid #DC3545;
        border-radius: 4px;
        margin: 1rem 0;
    }
    .info-box {
        padding: 1rem;
        background-color: #D1ECF1;
        border-left: 4px solid #17A2B8;
        border-radius: 4px;
        margin: 1rem 0;
    }
    .metric-card {
        background-color: #F8F9FA;
        padding: 1.5rem;
        border-radius: 8px;
        border: 1px solid #DEE2E6;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<div class="main-header">🔍 OIC-LogLens</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">AI-Powered Error Resolution Engine for Oracle Integration Cloud</div>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image("https://via.placeholder.com/200x80/1F4E79/FFFFFF?text=OIC-LogLens", width="stretch")
    st.markdown("---")
    
    page = st.radio(
        "Navigation",
        ["📥 Ingest Logs", "🔍 Search Duplicates", "📊 Dashboard"],
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    st.markdown("### ⚙️ API Status")
    
    # Check API health
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=2)
        if response.status_code == 200:
            st.success("✅ API Online")
        else:
            st.error("❌ API Error")
    except:
        st.error("❌ API Offline")
    
    st.markdown("---")
    st.markdown("### 📚 Resources")
    st.markdown("- [API Docs](http://localhost:8000/docs)")
    st.markdown("- [Testing Guide](TESTING.md)")
    st.markdown("- [GitHub Repo](https://github.com/bhagavansprasad/oic-log-lens)")


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 1: INGEST LOGS
# ─────────────────────────────────────────────────────────────────────────────

if page == "📥 Ingest Logs":
    st.header("📥 Ingest OIC Logs")
    st.markdown("Choose an ingestion method to add logs to the system.")
    
    tab1, tab2, tab3, tab4 = st.tabs(["📤 Upload File", "🌐 URL", "📝 Raw Text", "🗄️ Database"])
    
    # ── Tab 1: File Upload ──
    with tab1:
        st.subheader("Upload Log File")
        st.markdown("Browse and upload a log file from your computer.")
        
        uploaded_file = st.file_uploader(
            "Choose a JSON file",
            type=["json"],
            help="Select a JSON log file to upload"
        )
        
        if uploaded_file is not None:
            # Preview the file
            with st.expander("📄 Preview File Content"):
                file_content = uploaded_file.read().decode("utf-8")
                st.code(file_content[:500] + ("..." if len(file_content) > 500 else ""), language="json")
                uploaded_file.seek(0)  # Reset file pointer
            
            if st.button("🚀 Ingest Uploaded File", type="primary", key="ingest_upload"):
                with st.spinner("Ingesting log..."):
                    try:
                        # Read file content
                        log_content = uploaded_file.read().decode("utf-8")
                        
                        # Use /ingest/raw endpoint
                        response = requests.post(
                            f"{API_BASE_URL}/ingest/raw",
                            json={"log_content": log_content},
                            timeout=60
                        )
                        
                        if response.status_code == 200:
                            data = response.json()
                            st.markdown(f'<div class="success-box">✅ <b>Success!</b> Log ingested successfully.</div>', unsafe_allow_html=True)
                            
                            col1, col2 = st.columns(2)
                            with col1:
                                st.metric("Log ID", data["log_id"])
                            with col2:
                                st.metric("Jira ID", data["jira_id"].split("/")[-1])
                            
                            st.markdown(f"**Full Jira URL:** [{data['jira_id']}]({data['jira_id']})")
                        
                        elif response.status_code == 409:
                            st.markdown(f'<div class="error-box">⚠️ <b>Duplicate Detected!</b><br>{response.json()["detail"]}</div>', unsafe_allow_html=True)
                        
                        else:
                            st.markdown(f'<div class="error-box">❌ <b>Error!</b><br>{response.json()["detail"]}</div>', unsafe_allow_html=True)
                    
                    except Exception as e:
                        st.markdown(f'<div class="error-box">❌ <b>Request Failed!</b><br>{str(e)}</div>', unsafe_allow_html=True)
        else:
            st.info("👆 Please upload a JSON log file to begin.")
    
    # ── Tab 2: URL ──
    with tab2:
        st.subheader("Ingest from URL")
        st.markdown("Fetch a log file from a public HTTP/HTTPS URL.")
        
        url = st.text_input(
            "URL",
            value="https://storage.googleapis.com/promptlyai-public-bucket/oci_logs/01_flow-log.json",
            help="Direct link to a JSON log file"
        )
        
        if st.button("🚀 Ingest from URL", type="primary", key="ingest_url"):
            with st.spinner("Fetching and ingesting log..."):
                try:
                    response = requests.post(
                        f"{API_BASE_URL}/ingest/url",
                        json={"url": url},
                        timeout=60
                    )
                    
                    if response.status_code == 200:
                        data = response.json()
                        st.markdown(f'<div class="success-box">✅ <b>Success!</b> Log ingested from URL.</div>', unsafe_allow_html=True)
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Log ID", data["log_id"])
                        with col2:
                            st.metric("Jira ID", data["jira_id"].split("/")[-1])
                        
                        st.markdown(f"**Full Jira URL:** [{data['jira_id']}]({data['jira_id']})")
                    
                    elif response.status_code == 409:
                        st.markdown(f'<div class="error-box">⚠️ <b>Duplicate Detected!</b><br>{response.json()["detail"]}</div>', unsafe_allow_html=True)
                    
                    else:
                        st.markdown(f'<div class="error-box">❌ <b>Error!</b><br>{response.json()["detail"]}</div>', unsafe_allow_html=True)
                
                except Exception as e:
                    st.markdown(f'<div class="error-box">❌ <b>Request Failed!</b><br>{str(e)}</div>', unsafe_allow_html=True)
    
    # ── Tab 3: Raw Text ──
    with tab3:
        st.subheader("Ingest from Raw JSON")
        st.markdown("Paste the log content directly as JSON.")
        
        log_content = st.text_area(
            "Log Content (JSON Array)",
            value='[{"flowId": "test", "errorMessage": "sample error"}]',
            height=300,
            help="Paste the raw JSON log content"
        )
        
        if st.button("🚀 Ingest from Raw Text", type="primary", key="ingest_raw"):
            with st.spinner("Ingesting log..."):
                try:
                    response = requests.post(
                        f"{API_BASE_URL}/ingest/raw",
                        json={"log_content": log_content},
                        timeout=60
                    )
                    
                    if response.status_code == 200:
                        data = response.json()
                        st.markdown(f'<div class="success-box">✅ <b>Success!</b> Log ingested from raw text.</div>', unsafe_allow_html=True)
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Log ID", data["log_id"])
                        with col2:
                            st.metric("Jira ID", data["jira_id"].split("/")[-1])
                        
                        st.markdown(f"**Full Jira URL:** [{data['jira_id']}]({data['jira_id']})")
                    
                    elif response.status_code == 409:
                        st.markdown(f'<div class="error-box">⚠️ <b>Duplicate Detected!</b><br>{response.json()["detail"]}</div>', unsafe_allow_html=True)
                    
                    else:
                        st.markdown(f'<div class="error-box">❌ <b>Error!</b><br>{response.json()["detail"]}</div>', unsafe_allow_html=True)
                
                except Exception as e:
                    st.markdown(f'<div class="error-box">❌ <b>Request Failed!</b><br>{str(e)}</div>', unsafe_allow_html=True)
    
    # ── Tab 4: Database ──
    with tab4:
        st.subheader("Ingest from Database")
        st.markdown("Query a log from Oracle or other database.")
        
        col1, col2 = st.columns(2)
        
        with col1:
            connection_string = st.text_input(
                "Connection String",
                value="EA_APP/jnjnuh@localhost/FREEPDB1",
                help="Database connection string"
            )
        
        with col2:
            query = st.text_input(
                "SQL Query",
                value="SELECT LOG_JSON FROM TEST_LOGS WHERE LOG_ID = 1",
                help="SQL query to fetch the log"
            )
        
        if st.button("🚀 Ingest from Database", type="primary", key="ingest_db"):
            with st.spinner("Querying database and ingesting log..."):
                try:
                    response = requests.post(
                        f"{API_BASE_URL}/ingest/database",
                        json={
                            "connection_string": connection_string,
                            "query": query
                        },
                        timeout=60
                    )
                    
                    if response.status_code == 200:
                        data = response.json()
                        
                        # Handle batch response
                        if data["status"] == "success":
                            st.markdown(f'<div class="success-box">✅ <b>Success!</b> {data["message"]}</div>', unsafe_allow_html=True)
                        elif data["status"] == "partial_success":
                            st.markdown(f'<div class="info-box">⚠️ <b>Partial Success!</b> {data["message"]}</div>', unsafe_allow_html=True)
                        else:
                            st.markdown(f'<div class="error-box">❌ <b>Error!</b> {data["message"]}</div>', unsafe_allow_html=True)
                        
                        # Show summary metrics
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Total", data["total_logs"])
                        with col2:
                            st.metric("✅ Successful", data["successful"])
                        with col3:
                            st.metric("⚠️ Duplicates", data["duplicates"])
                        with col4:
                            st.metric("❌ Failed", data["failed"])
                        
                        # Show individual results
                        if data.get("results"):
                            with st.expander("📋 View Individual Results", expanded=True):
                                for i, result in enumerate(data["results"], 1):
                                    if result["status"] == "success":
                                        st.success(f"**Log {i}:** {result['message']}")
                                        st.markdown(f"  - Log ID: `{result['log_id']}`")
                                        st.markdown(f"  - Jira: [{result['jira_id'].split('/')[-1]}]({result['jira_id']})")
                                    elif result["status"] == "duplicate":
                                        st.warning(f"**Log {i}:** {result['message']}")
                                    else:
                                        st.error(f"**Log {i}:** {result['message']}")
                    
                    elif response.status_code == 409:
                        st.markdown(f'<div class="error-box">⚠️ <b>Duplicate Detected!</b><br>{response.json()["detail"]}</div>', unsafe_allow_html=True)
                    
                    else:
                        st.markdown(f'<div class="error-box">❌ <b>Error!</b><br>{response.json()["detail"]}</div>', unsafe_allow_html=True)
                
                except Exception as e:
                    st.markdown(f'<div class="error-box">❌ <b>Request Failed!</b><br>{str(e)}</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 2: SEARCH DUPLICATES
# ─────────────────────────────────────────────────────────────────────────────

elif page == "🔍 Search Duplicates":
    st.header("🔍 Search for Duplicate Logs")
    st.markdown("Find similar logs using semantic similarity search.")
    
    # Add tabs for file upload vs text input
    search_tab1, search_tab2 = st.tabs(["📤 Upload File", "📝 Paste Text"])
    
    # Use session state to track which tab's content to use
    if 'search_method' not in st.session_state:
        st.session_state.search_method = 'text'
    
    uploaded_content = None
    text_content = None
    
    with search_tab1:
        st.markdown("Upload a log file to search for similar logs.")
        uploaded_search_file = st.file_uploader(
            "Choose a JSON log file",
            type=["json"],
            help="Select a JSON log file to search",
            key="search_upload"
        )
        
        if uploaded_search_file is not None:
            st.session_state.search_method = 'upload'
            # Preview
            with st.expander("📄 Preview File Content"):
                file_content = uploaded_search_file.read().decode("utf-8")
                st.code(file_content[:500] + ("..." if len(file_content) > 500 else ""), language="json")
                uploaded_search_file.seek(0)
            
            uploaded_content = uploaded_search_file.read().decode("utf-8")
    
    with search_tab2:
        st.markdown("Paste the raw log content as JSON.")
        text_content = st.text_area(
            "Log Content (JSON Array)",
            value='[{"flowId": "test", "errorMessage": "sample error"}]',
            height=300,
            help="Paste the log you want to search for",
            key="search_text"
        )
        if text_content and text_content != '[{"flowId": "test", "errorMessage": "sample error"}]':
            st.session_state.search_method = 'text'
    
    # Use the correct content based on which tab was used
    if st.session_state.search_method == 'upload' and uploaded_content:
        log_content = uploaded_content
    else:
        log_content = text_content
    
    if st.button("🔍 Search Similar Logs", type="primary") and log_content:
        with st.spinner("Searching for similar logs..."):
            try:
                response = requests.post(
                    f"{API_BASE_URL}/search",
                    json={"log_content": log_content},
                    timeout=60
                )
                
                if response.status_code == 200:
                    data = response.json()
                    
                    st.markdown(f'<div class="success-box">✅ <b>{data["message"]}</b></div>', unsafe_allow_html=True)
                    
                    if data["matches"]:
                        st.markdown("### 📊 Search Results")
                        
                        for i, match in enumerate(data["matches"], 1):
                            # Determine badge color based on classification
                            classification = match.get('classification', 'UNKNOWN')
                            if classification == 'EXACT_DUPLICATE':
                                badge_color = "🟢"
                                badge_text = "EXACT DUPLICATE"
                            elif classification == 'SIMILAR_ROOT_CAUSE':
                                badge_color = "🟡"
                                badge_text = "SIMILAR ROOT CAUSE"
                            elif classification == 'RELATED':
                                badge_color = "🟠"
                                badge_text = "RELATED"
                            else:
                                badge_color = "⚪"
                                badge_text = "NOT RELATED"
                            
                            # New format: CLASSIFICATION - LLM's confidence X% (Vector: Y%) - JIRA_ID
                            confidence = match.get('confidence', 'N/A')
                            if confidence != 'N/A':
                                confidence_str = f"LLM's confidence {confidence}%"
                            else:
                                confidence_str = "No confidence score"
                            
                            jira_short = match['jira_id'].split('/')[-1]
                            jira_url = match['jira_id']
                            title = f"**Rank {match.get('rank', i)} — {badge_text} - {confidence_str} (Vector: {match['similarity_score']}%) - [{jira_short}]({jira_url})**"
                            
                            with st.expander(title, expanded=(i==1)):
                                # Classification banner
                                if match.get('reasoning'):
                                    st.info(f"**💡 Why this match:** {match['reasoning']}")
                                
                                col1, col2, col3, col4 = st.columns(4)
                                
                                with col1:
                                    st.metric("Similarity", f"{match['similarity_score']}%")
                                with col2:
                                    st.metric("Confidence", f"{match.get('confidence', 'N/A')}")
                                with col3:
                                    st.metric("Flow Code", match['flow_code'])
                                with col4:
                                    st.metric("Error Code", match['error_code'] or "N/A")
                                
                                st.markdown(f"**Jira ID:** [{match['jira_id']}]({match['jira_id']})")
                                st.markdown(f"**Trigger Type:** {match['trigger_type'] or 'N/A'}")
                                st.markdown(f"**Error Summary:**")
                                st.code(match['error_summary'], language=None)
                    else:
                        st.info("No similar logs found.")
                
                else:
                    st.markdown(f'<div class="error-box">❌ <b>Error!</b><br>{response.json()["detail"]}</div>', unsafe_allow_html=True)
            
            except Exception as e:
                st.markdown(f'<div class="error-box">❌ <b>Request Failed!</b><br>{str(e)}</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 3: DASHBOARD
# ─────────────────────────────────────────────────────────────────────────────

elif page == "📊 Dashboard":
    st.header("📊 System Dashboard")
    st.markdown("Overview of the OIC-LogLens system.")
    
    # Metrics (mock data - replace with real queries)
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Total Logs", "12", "+3")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Unique Jira IDs", "8", "+2")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Duplicates Detected", "4", "+1")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col4:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Avg Similarity", "78%", "-2%")
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # System Info
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### ⚙️ System Configuration")
        st.markdown("""
        - **API Endpoint:** `http://localhost:8000`
        - **Database:** Oracle 26ai (FREEPDB1)
        - **LLM Model:** Gemini 2.0 Flash
        - **Embedding Model:** gemini-embedding-001 (3072 dims)
        - **Vector Index:** HNSW Cosine (95% accuracy)
        """)
    
    with col2:
        st.markdown("### 📈 Recent Activity")
        st.markdown(f"""
        - **Last Ingestion:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        - **Last Search:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        - **API Uptime:** 99.9%
        - **Avg Response Time:** 12.3s
        """)
    
    st.markdown("---")
    st.markdown("### 🔗 Quick Links")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📖 API Documentation", width="stretch"):
            st.markdown("[Open API Docs](http://localhost:8000/docs)")
    
    with col2:
        if st.button("🧪 Testing Guide", width="stretch"):
            st.markdown("[View TESTING.md](TESTING.md)")
    
    with col3:
        if st.button("💻 GitHub Repository", width="stretch"):
            st.markdown("[View on GitHub](https://github.com/bhagavansprasad/oic-log-lens)")

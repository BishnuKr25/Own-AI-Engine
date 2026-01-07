"""Production Streamlit Frontend for Sovereign AI Suite"""
import streamlit as st
import requests
import json
from datetime import datetime
import pandas as pd
import time
from pathlib import Path
import base64
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, Any, Optional

# Page configuration
st.set_page_config(
    page_title="Sovereign AI Suite",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'About': "Sovereign AI Suite - Enterprise Self-Hosted AI System"
    }
)

# Custom CSS for production UI
st.markdown("""
<style>
    /* Main theme */
    .main-header {
        font-size: 3.5rem;
        font-weight: bold;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 2rem 0;
        margin-bottom: 2rem;
    }
    
    /* Card styles */
    .metric-card {
        background: white;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        border-left: 4px solid #667eea;
        margin-bottom: 1rem;
    }
    
    /* Button styles */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 25px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
    }
    
    /* Expert badges */
    .expert-badge {
        display: inline-block;
        padding: 0.3rem 0.8rem;
        margin: 0.2rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 500;
    }
    
    .gemma-badge {
        background: linear-gradient(135deg, #4285f4 0%, #34a853 100%);
        color: white;
    }
    
    .coding-badge {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
    }
    
    .general-badge {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
    
    /* Response container */
    .response-container {
        background: #f8f9fa;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        border-left: 4px solid #667eea;
    }
    
    /* Status indicators */
    .status-online {
        color: #10b981;
        font-weight: 600;
    }
    
    .status-offline {
        color: #ef4444;
        font-weight: 600;
    }
    
    /* Loading animation */
    .loading-text {
        animation: pulse 1.5s ease-in-out infinite;
    }
    
    @keyframes pulse {
        0% { opacity: 0.6; }
        50% { opacity: 1; }
        100% { opacity: 0.6; }
    }
</style>
""", unsafe_allow_html=True)

# Session state initialization
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False
if 'api_key' not in st.session_state:
    st.session_state.api_key = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'user_info' not in st.session_state:
    st.session_state.user_info = None
if 'metrics' not in st.session_state:
    st.session_state.metrics = {
        'total_queries': 0,
        'avg_response_time': 0,
        'models_used': set()
    }

# API Configuration
API_BASE_URL = st.secrets.get("API_BASE_URL", "http://localhost:8000/api/v1")

class APIClient:
    """API client for backend communication"""
    
    @staticmethod
    def check_health() -> bool:
        """Check if API is healthy"""
        try:
            response = requests.get(f"{API_BASE_URL.replace('/api/v1', '')}/health", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    @staticmethod
    def process_query(
        query: str,
        api_key: str,
        **kwargs
    ) -> Dict[str, Any]:
        """Process a query through the API"""
        headers = {"X-API-Key": api_key}
        payload = {
            "query": query,
            "context": kwargs.get("context"),
            "task_type": kwargs.get("task_type"),
            "num_experts": kwargs.get("num_experts", 3),
            "temperature": kwargs.get("temperature", 0.7),
            "max_tokens": kwargs.get("max_tokens", 2048),
            "use_rag": kwargs.get("use_rag", True),
            "fast_mode": kwargs.get("fast_mode", False)
        }
        
        try:
            response = requests.post(
                f"{API_BASE_URL}/query",
                json=payload,
                headers=headers,
                timeout=120
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            return {"error": str(e)}
    
    @staticmethod
    def get_models(api_key: str) -> Dict[str, Any]:
        """Get available models"""
        headers = {"X-API-Key": api_key}
        try:
            response = requests.get(
                f"{API_BASE_URL}/models",
                headers=headers,
                timeout=10
            )
            response.raise_for_status()
            return response.json()
        except:
            return {"models": []}
    
    @staticmethod
    def upload_file(
        file,
        api_key: str,
        query: Optional[str] = None,
        process_type: str = "analyze"
    ) -> Dict[str, Any]:
        """Upload and process a file"""
        headers = {"X-API-Key": api_key}
        files = {"file": (file.name, file.getvalue(), file.type)}
        data = {"query": query or "", "process_type": process_type}
        
        try:
            response = requests.post(
                f"{API_BASE_URL}/upload",
                files=files,
                data=data,
                headers=headers,
                timeout=120
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            return {"error": str(e)}

# Initialize API client
api_client = APIClient()

def render_header():
    """Render application header"""
    st.markdown('<h1 class="main-header">🚀 Sovereign AI Suite</h1>', unsafe_allow_html=True)
    
    # Check API health
    api_healthy = api_client.check_health()
    
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        st.markdown("**Enterprise Self-Hosted AI System**")
    with col2:
        if api_healthy:
            st.markdown('<span class="status-online">● System Online</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span class="status-offline">● System Offline</span>', unsafe_allow_html=True)
    with col3:
        st.markdown(f"**v1.0.0** | {datetime.now().strftime('%Y-%m-%d')}")

def render_sidebar():
    """Render sidebar with settings and information"""
    with st.sidebar:
        st.markdown("## 🔐 Authentication")
        
        if not st.session_state.authenticated:
            api_key = st.text_input(
                "API Key",
                type="password",
                placeholder="sk-...",
                help="Enter your API key to access the system"
            )
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🔗 Connect", use_container_width=True):
                    if api_key:
                        # Validate API key
                        models = api_client.get_models(api_key)
                        if "error" not in models:
                            st.session_state.api_key = api_key
                            st.session_state.authenticated = True
                            st.success("✅ Connected successfully!")
                            time.sleep(1)
                            st.rerun()
                        else:
                            st.error("❌ Invalid API key")
                    else:
                        st.warning("Please enter an API key")
            
            with col2:
                if st.button("🔑 Use Demo", use_container_width=True):
                    st.session_state.api_key = "demo-key-123"
                    st.session_state.authenticated = True
                    st.info("Using demo mode")
                    time.sleep(1)
                    st.rerun()
        else:
            st.success("✅ Authenticated")
            if st.button("🚪 Disconnect", use_container_width=True):
                st.session_state.authenticated = False
                st.session_state.api_key = None
                st.session_state.chat_history = []
                st.rerun()
        
        st.markdown("---")
        
        if st.session_state.authenticated:
            st.markdown("## ⚙️ Configuration")
            
            # Model settings
            with st.expander("🤖 Model Settings", expanded=True):
                fast_mode = st.checkbox(
                    "⚡ Fast Mode",
                    value=False,
                    help="Use Gemma 3 for ultra-fast responses"
                )
                
                if not fast_mode:
                    num_experts = st.slider(
                        "Number of Experts",
                        min_value=1,
                        max_value=5,
                        value=3,
                        help="More experts = higher quality but slower"
                    )
                else:
                    num_experts = 1
                    st.info("Fast mode uses Gemma 3 2B/9B")
                
                temperature = st.slider(
                    "Temperature",
                    min_value=0.0,
                    max_value=2.0,
                    value=0.7,
                    step=0.1,
                    help="Higher = more creative, Lower = more focused"
                )
                
                max_tokens = st.select_slider(
                    "Max Tokens",
                    options=[512, 1024, 2048, 4096, 8192],
                    value=2048,
                    help="Maximum response length"
                )
                
                use_rag = st.checkbox(
                    "📚 Use Knowledge Base",
                    value=True,
                    help="Search internal knowledge base for context"
                )
            
            # Available models display
            with st.expander("📊 Available Models"):
                models_data = api_client.get_models(st.session_state.api_key)
                if models_data.get("models"):
                    # Group by category
                    general_models = []
                    coding_models = []
                    multimodal_models = []
                    
                    for model in models_data["models"]:
                        if "coding" in model["categories"]:
                            coding_models.append(model)
                        elif "multimodal" in model["categories"]:
                            multimodal_models.append(model)
                        else:
                            general_models.append(model)
                    
                    st.markdown("**General & Analysis**")
                    for model in general_models[:5]:
                        badge_class = "gemma-badge" if "gemma" in model["key"] else "general-badge"
                        st.markdown(
                            f'<span class="expert-badge {badge_class}">{model["name"]}</span>',
                            unsafe_allow_html=True
                        )
                    
                    if coding_models:
                        st.markdown("**Coding Specialists**")
                        for model in coding_models[:5]:
                            st.markdown(
                                f'<span class="expert-badge coding-badge">{model["name"]}</span>',
                                unsafe_allow_html=True
                            )
            
            # Session metrics
            st.markdown("---")
            st.markdown("## 📈 Session Metrics")
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total Queries", st.session_state.metrics['total_queries'])
            with col2:
                st.metric(
                    "Avg Response Time",
                    f"{st.session_state.metrics['avg_response_time']:.1f}s"
                )
            
            if st.session_state.metrics['models_used']:
                st.markdown("**Models Used:**")
                for model in st.session_state.metrics['models_used']:
                    st.markdown(f"• {model}")
            
            if st.button("🔄 Reset Session", use_container_width=True):
                st.session_state.chat_history = []
                st.session_state.metrics = {
                    'total_queries': 0,
                    'avg_response_time': 0,
                    'models_used': set()
                }
                st.rerun()
        
        # Return configuration
        return {
            'fast_mode': locals().get('fast_mode', False),
            'num_experts': locals().get('num_experts', 3),
            'temperature': locals().get('temperature', 0.7),
            'max_tokens': locals().get('max_tokens', 2048),
            'use_rag': locals().get('use_rag', True)
        }

def render_chat_interface(config: Dict[str, Any]):
    """Render main chat interface"""
    
    # Chat history display
    for message in st.session_state.chat_history:
        if message["role"] == "user":
            with st.chat_message("user"):
                st.markdown(message["content"])
        else:
            with st.chat_message("assistant"):
                st.markdown(message["content"])
                
                # Show metadata if available
                if "metadata" in message:
                    with st.expander("📊 Details"):
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.markdown(f"**Type:** {message['metadata'].get('task_type', 'N/A')}")
                        with col2:
                            st.markdown(f"**Time:** {message['metadata'].get('time', 'N/A')}")
                        with col3:
                            st.markdown(f"**Confidence:** {message['metadata'].get('confidence', 0):.2f}")
                        with col4:
                            st.markdown(f"**Tokens:** {message['metadata'].get('tokens', 'N/A')}")
                        
                        if message['metadata'].get('experts'):
                            st.markdown("**Experts consulted:**")
                            for expert in message['metadata']['experts']:
                                st.markdown(f"• {expert}")
    
    # Chat input
    if prompt := st.chat_input("Ask anything... I can help with coding, analysis, creative writing, and more!"):
        # Add user message
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Get AI response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                start_time = time.time()
                
                # Process query
                response = api_client.process_query(
                    query=prompt,
                    api_key=st.session_state.api_key,
                    **config
                )
                
                processing_time = time.time() - start_time
                
                if "error" in response:
                    st.error(f"Error: {response['error']}")
                else:
                    # Display response
                    st.markdown(response["answer"])
                    
                    # Update metrics
                    st.session_state.metrics['total_queries'] += 1
                    current_avg = st.session_state.metrics['avg_response_time']
                    st.session_state.metrics['avg_response_time'] = (
                        (current_avg * (st.session_state.metrics['total_queries'] - 1) + processing_time) /
                        st.session_state.metrics['total_queries']
                    )
                    
                    # Track models used
                    for expert in response.get("experts_consulted", []):
                        st.session_state.metrics['models_used'].add(expert)
                    
                    # Add to chat history
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": response["answer"],
                        "metadata": {
                            "task_type": response.get("task_type"),
                            "time": f"{processing_time:.1f}s",
                            "confidence": response.get("confidence_score", 0),
                            "tokens": response.get("tokens_used", 0),
                            "experts": response.get("experts_consulted", [])
                        }
                    })

def render_file_upload_tab():
    """Render file upload interface"""
    st.markdown("### 📁 File Processing")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded_file = st.file_uploader(
            "Choose a file",
            type=['pdf', 'txt', 'docx', 'csv', 'xlsx', 'png', 'jpg', 'jpeg', 'mp3', 'wav'],
            help="Upload documents, images, or audio files for processing"
        )
    
    with col2:
        process_type = st.selectbox(
            "Process Type",
            ["analyze", "extract", "summarize", "add_to_knowledge"],
            help="How to process the uploaded file"
        )
    
    if uploaded_file:
        st.markdown(f"**File:** {uploaded_file.name} ({uploaded_file.size / 1024:.1f} KB)")
        
        query = st.text_area(
            "Query (optional)",
            placeholder="Ask a specific question about the file content",
            height=100
        )
        
        if st.button("🚀 Process File", use_container_width=True):
            with st.spinner("Processing file..."):
                result = api_client.upload_file(
                    uploaded_file,
                    st.session_state.api_key,
                    query,
                    process_type
                )
            
            if "error" in result:
                st.error(f"Error: {result['error']}")
            else:
                st.success("✅ File processed successfully!")
                
                # Display results
                st.markdown("### Results:")
                if isinstance(result.get("result"), str):
                    st.markdown(result["result"])
                elif isinstance(result.get("transcription"), str):
                    st.markdown("**Transcription:**")
                    st.markdown(result["transcription"])
                elif isinstance(result.get("text"), str):
                    st.markdown("**Extracted Text:**")
                    with st.expander("View full text"):
                        st.text(result["text"])
                
                # Show metadata
                if result.get("success"):
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Type", result.get("type", "N/A"))
                    with col2:
                        if result.get("duration"):
                            st.metric("Duration", f"{result['duration']:.1f}s")
                    with col3:
                        if result.get("length"):
                            st.metric("Length", f"{result['length']} chars")

def render_analytics_tab():
    """Render analytics dashboard"""
    st.markdown("### 📊 Usage Analytics")
    
    # Get analytics data (mock data for demo)
    analytics_data = {
        "queries_by_type": {
            "general": 45,
            "coding": 30,
            "analysis": 20,
            "creative": 15,
            "data_generation": 10
        },
        "response_times": [2.3, 3.1, 1.8, 4.2, 2.7, 3.5, 2.1, 3.8, 2.5, 3.0],
        "models_usage": {
            "gemma3-27b": 35,
            "llama3-70b": 25,
            "codestral-22b": 20,
            "mixtral-8x22b": 15,
            "gemma3-9b": 30,
            "gemma3-2b": 40
        }
    }
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Query types pie chart
        fig1 = px.pie(
            values=list(analytics_data["queries_by_type"].values()),
            names=list(analytics_data["queries_by_type"].keys()),
            title="Queries by Type",
            color_discrete_sequence=px.colors.sequential.Purples_r
        )
        st.plotly_chart(fig1, use_container_width=True)
        
        # Response time line chart
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            y=analytics_data["response_times"],
            mode='lines+markers',
            name='Response Time',
            line=dict(color='#667eea', width=2),
            marker=dict(size=8)
        ))
        fig2.update_layout(
            title="Response Times (last 10 queries)",
            xaxis_title="Query",
            yaxis_title="Time (seconds)",
            showlegend=False
        )
        st.plotly_chart(fig2, use_container_width=True)
    
    with col2:
        # Model usage bar chart
        fig3 = px.bar(
            x=list(analytics_data["models_usage"].values()),
            y=list(analytics_data["models_usage"].keys()),
            orientation='h',
            title="Model Usage Count",
            color=list(analytics_data["models_usage"].values()),
            color_continuous_scale='Purples'
        )
        fig3.update_layout(showlegend=False)
        st.plotly_chart(fig3, use_container_width=True)
        
        # Summary metrics
        st.markdown("### Summary Statistics")
        
        total_queries = sum(analytics_data["queries_by_type"].values())
        avg_response = sum(analytics_data["response_times"]) / len(analytics_data["response_times"])
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Queries", total_queries)
            st.metric("Avg Response Time", f"{avg_response:.2f}s")
        with col2:
            st.metric("Active Models", len(analytics_data["models_usage"]))
            st.metric("Success Rate", "99.5%")

def render_examples_section():
    """Render example queries section"""
    st.markdown("### 💡 Example Queries")
    
    examples = {
        "💻 Coding": [
            "Write a Python function to implement binary search with error handling",
            "Debug this code: def factorial(n): return n * factorial(n-1)",
            "Create a REST API with FastAPI for user authentication",
            "Optimize this SQL query for better performance",
            "Convert this Python code to TypeScript"
        ],
        "📊 Analysis": [
            "Analyze the pros and cons of microservices vs monolithic architecture",
            "Compare React, Vue, and Angular for enterprise applications",
            "What are the key performance metrics for a web application?",
            "Explain the CAP theorem with real-world examples"
        ],
        "✏️ Creative": [
            "Write a short story about an AI discovering consciousness",
            "Create a haiku about programming",
            "Generate creative names for a tech startup",
            "Write a motivational quote for developers"
        ],
        "📈 Data": [
            "Generate sample CSV data for a sales dashboard with 100 rows",
            "Create a JSON schema for a user profile system",
            "Generate SQL queries to create a blog database schema",
            "Create test data for an e-commerce platform"
        ]
    }
    
    for category, queries in examples.items():
        with st.expander(category):
            for query in queries:
                if st.button(query, key=f"example_{query[:20]}", use_container_width=True):
                    st.session_state.next_query = query
                    st.rerun()

def main():
    """Main application function"""
    
    # Render header
    render_header()
    
    # Render sidebar and get configuration
    config = render_sidebar()
    
    if not st.session_state.authenticated:
        # Welcome screen
        st.markdown("---")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.markdown("### 🧠 Advanced AI")
            st.markdown("""
            - Council of Experts system
            - Gemma 3 optimization
            - 10+ specialized models
            - Multimodal processing
            """)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.markdown("### 🔒 Complete Privacy")
            st.markdown("""
            - 100% self-hosted
            - No external APIs
            - Your data stays yours
            - Enterprise security
            """)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col3:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.markdown("### ⚡ High Performance")
            st.markdown("""
            - Sub-second responses
            - Parallel processing
            - Smart caching
            - Auto-scaling
            """)
            st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Examples section
        render_examples_section()
        
    else:
        # Main application tabs
        tab1, tab2, tab3, tab4 = st.tabs(["💬 Chat", "📁 Files", "📊 Analytics", "💡 Examples"])
        
        with tab1:
            render_chat_interface(config)
        
        with tab2:
            render_file_upload_tab()
        
        with tab3:
            render_analytics_tab()
        
        with tab4:
            render_examples_section()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 1rem;'>
        <p>Sovereign AI Suite v1.0.0 | Powered by Council of Experts with Gemma 3</p>
        <p>🔒 Self-Hosted | ⚡ High Performance | 🛡️ Enterprise Security</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
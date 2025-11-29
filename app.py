"""
SentiScope - AI-Powered Sentiment Analysis Tool
Track emotions and analyze feedback with AI
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from transformers import pipeline
import json

# Page configuration
st.set_page_config(
    page_title="SentiScope - AI Sentiment Analysis",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    /* FORCE LIGHT MODE - Override Streamlit dark theme */
    .main {
        background-color: #ffffff !important;
    }
    
    .stApp {
        background-color: #ffffff !important;
    }
    
    /* All text must be dark */
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Inter', sans-serif;
        color: #1e293b !important;
    }
    
    p, div, span, label {
        font-family: 'Inter', sans-serif;
        color: #1e293b !important;
    }
    
    .main * {
        color: #1e293b !important;
    }
    
    .main h1, .main h2, .main h3 {
        color: #1e293b !important;
    }
    
    .main p, .main div {
        color: #1e293b !important;
    }
    
    .sentiscope-header {
        background: linear-gradient(135deg, #8b5cf6 0%, #6366f1 100%);
        padding: 2.5rem;
        border-radius: 12px;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(139, 92, 246, 0.3);
    }
    
    .sentiscope-title {
        color: white;
        font-size: 48px;
        font-weight: 700;
        margin: 0;
        text-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    
    .sentiscope-subtitle {
        color: rgba(255, 255, 255, 0.95);
        font-size: 18px;
        margin: 8px 0 0 0;
        font-weight: 500;
    }
    
    .sentiment-positive {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: white;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
        margin: 0.5rem 0;
    }
    
    .sentiment-negative {
        background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
        color: white;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
        margin: 0.5rem 0;
    }
    
    .sentiment-neutral {
        background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%);
        color: white;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
        margin: 0.5rem 0;
    }
    
    .stTextArea textarea {
        font-size: 14px !important;
        line-height: 1.6 !important;
        background-color: #f8fafc !important;
        border: 2px solid #e2e8f0 !important;
        border-radius: 8px !important;
        color: #1e293b !important;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #8b5cf6 0%, #6366f1 100%) !important;
        color: white !important;
        font-weight: 600 !important;
        border-radius: 8px !important;
        padding: 12px 32px !important;
        border: none !important;
        font-size: 16px !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 6px rgba(139, 92, 246, 0.3) !important;
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #7c3aed 0%, #4f46e5 100%) !important;
        box-shadow: 0 6px 12px rgba(139, 92, 246, 0.4) !important;
        transform: translateY(-2px) !important;
    }
    
    /* Download button specific styling */
    .stDownloadButton > button {
        background: linear-gradient(135deg, #8b5cf6 0%, #6366f1 100%) !important;
        color: white !important;
        font-weight: 600 !important;
        border-radius: 8px !important;
        padding: 12px 32px !important;
        border: none !important;
        font-size: 16px !important;
    }
    
    .stDownloadButton > button:hover {
        background: linear-gradient(135deg, #7c3aed 0%, #4f46e5 100%) !important;
    }
    
    [data-testid="stMetricValue"] {
        font-size: 32px !important;
        font-weight: 700 !important;
        color: #1e293b !important;
    }
    
    [data-testid="stMetricLabel"] {
        font-size: 14px !important;
        color: #64748b !important;
        font-weight: 500 !important;
    }
    
    /* Sidebar Styling - Light text on dark background */
    [data-testid="stSidebar"] {
        background-color: #1e293b !important;
    }
    
    [data-testid="stSidebar"] * {
        color: #f8fafc !important;
    }
    
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        color: #ffffff !important;
    }
    
    [data-testid="stSidebar"] .stMarkdown {
        color: #e2e8f0 !important;
    }
    
    [data-testid="stSidebar"] hr {
        border-color: #475569 !important;
    }
    
    /* Sidebar radio buttons */
    [data-testid="stSidebar"] label {
        color: #f8fafc !important;
    }
    
    [data-testid="stSidebar"] [role="radiogroup"] label {
        color: #f8fafc !important;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="sentiscope-header">
    <h1 class="sentiscope-title">SentiScope</h1>
    <p class="sentiscope-subtitle">Track Emotions and Analyze Feedback with AI</p>
</div>
""", unsafe_allow_html=True)

# Initialize sentiment analyzer
@st.cache_resource
def load_model():
    """Load the sentiment analysis model with 3 classes (positive, neutral, negative)"""
    return pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment-latest")

# Initialize session state
if 'journal_entries' not in st.session_state:
    st.session_state.journal_entries = []

if 'batch_results' not in st.session_state:
    st.session_state.batch_results = None

# Sidebar - Mode Selection
with st.sidebar:
    st.markdown("### Choose Mode")
    mode = st.radio(
        "Select analysis mode:",
        ["Personal Journal", "Batch Analyzer"],
        help="Personal Journal: Track your own sentiment over time\nBatch Analyzer: Analyze multiple texts at once"
    )
    
    st.divider()
    
    st.markdown("### About SentiScope")
    st.markdown("""
    SentiScope uses state-of-the-art Deep Learning (Twitter-RoBERTa) to analyze sentiment in text with three classes: positive, neutral, and negative.
    
    **Personal Journal:**
    - Track your mood over time
    - Visualize sentiment trends
    - Gain insights into your emotional patterns
    
    **Batch Analyzer:**
    - Analyze reviews, tweets, feedback
    - Bulk sentiment analysis
    - Export results as CSV
    - Perfect for market research
    """)
    
    st.divider()
    
    st.markdown("### Powered By")
    st.markdown("""
    - Hugging Face Transformers
    - Twitter-RoBERTa Model
    - Streamlit
    - Plotly Visualizations
    """)

# Load model
with st.spinner("Loading AI model..."):
    sentiment_analyzer = load_model()

# ============================================================================
# MODE 1: PERSONAL JOURNAL
# ============================================================================

if mode == "Personal Journal":
    st.markdown("### Personal Sentiment Journal")
    st.markdown("Track your emotional journey with AI-powered sentiment analysis")
    
    # Input section
    col1, col2 = st.columns([3, 1])
    
    with col1:
        journal_text = st.text_area(
            "Write your thoughts:",
            height=200,
            placeholder="Today I felt... \n\nWhat's on your mind? Express yourself freely.",
            help="Write about your day, feelings, or anything on your mind"
        )
    
    with col2:
        st.markdown("#### Quick Stats")
        if st.session_state.journal_entries:
            st.metric("Total Entries", len(st.session_state.journal_entries))
            
            # Calculate average sentiment
            recent_sentiments = [e['sentiment'] for e in st.session_state.journal_entries[-7:]]
            positive_count = sum(1 for s in recent_sentiments if s == 'POSITIVE')
            st.metric("Positive (Last 7)", f"{positive_count}/7")
        else:
            st.info("No entries yet. Start writing!")
    
    if st.button("Analyze My Entry", use_container_width=True):
        if journal_text.strip():
            with st.spinner("Analyzing sentiment..."):
                # Analyze sentiment
                result = sentiment_analyzer(journal_text[:512])[0]
                
                # Store entry
                entry = {
                    'timestamp': datetime.now().isoformat(),
                    'text': journal_text,
                    'sentiment': result['label'],
                    'confidence': result['score']
                }
                st.session_state.journal_entries.append(entry)
                
                # Display result
                sentiment = result['label']
                confidence = result['score'] * 100
                
                if sentiment == 'positive':
                    st.markdown(f"""
                    <div class="sentiment-positive">
                        <h3 style="margin: 0; color: white;">😊 Positive Sentiment</h3>
                        <p style="margin: 8px 0 0 0; font-size: 18px;">Confidence: {confidence:.1f}%</p>
                    </div>
                    """, unsafe_allow_html=True)
                elif sentiment == 'negative':
                    st.markdown(f"""
                    <div class="sentiment-negative">
                        <h3 style="margin: 0; color: white;">😔 Negative Sentiment</h3>
                        <p style="margin: 8px 0 0 0; font-size: 18px;">Confidence: {confidence:.1f}%</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:  # neutral
                    st.markdown(f"""
                    <div class="sentiment-neutral">
                        <h3 style="margin: 0; color: white;">😐 Neutral Sentiment</h3>
                        <p style="margin: 8px 0 0 0; font-size: 18px;">Confidence: {confidence:.1f}%</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                st.success("Entry saved to your journal!")
        else:
            st.error("Please write something first!")
    
    # Display journal history
    if st.session_state.journal_entries:
        st.divider()
        st.markdown("### Your Sentiment Timeline")
        
        # Create dataframe
        df = pd.DataFrame(st.session_state.journal_entries)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['sentiment_numeric'] = df['sentiment'].map({'positive': 1, 'neutral': 0, 'negative': -1})
        
        # Sentiment over time chart
        fig = px.line(
            df, 
            x='timestamp', 
            y='sentiment_numeric',
            title='Sentiment Trend Over Time',
            labels={'sentiment_numeric': 'Sentiment', 'timestamp': 'Date'},
            markers=True
        )
        fig.update_yaxes(tickvals=[-1, 0, 1], ticktext=['Negative', 'Neutral', 'Positive'])
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        # Sentiment distribution
        col1, col2 = st.columns(2)
        
        with col1:
            sentiment_counts = df['sentiment'].value_counts()
            fig_pie = px.pie(
                values=sentiment_counts.values,
                names=sentiment_counts.index,
                title='Overall Sentiment Distribution',
                color=sentiment_counts.index,
                color_discrete_map={'positive': '#10b981', 'neutral': '#f59e0b', 'negative': '#ef4444'}
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        
        with col2:
            avg_confidence = df.groupby('sentiment')['confidence'].mean() * 100
            fig_bar = px.bar(
                x=avg_confidence.index,
                y=avg_confidence.values,
                title='Average Confidence by Sentiment',
                labels={'x': 'Sentiment', 'y': 'Confidence (%)'},
                color=avg_confidence.index,
                color_discrete_map={'positive': '#10b981', 'neutral': '#f59e0b', 'negative': '#ef4444'}
            )
            st.plotly_chart(fig_bar, use_container_width=True)
        
        # Recent entries
        st.markdown("### Recent Entries")
        for entry in reversed(st.session_state.journal_entries[-5:]):
            with st.expander(f"{entry['timestamp'][:10]} - {entry['sentiment']} ({entry['confidence']*100:.1f}%)"):
                st.write(entry['text'])
        
        # Export options
        st.markdown("### Export Journal")
        col1, col2 = st.columns(2)
        
        with col1:
            # JSON export
            json_data = json.dumps(st.session_state.journal_entries, indent=2)
            st.download_button(
                label="📥 Download as JSON",
                data=json_data,
                file_name=f"sentiscope_journal_{datetime.now().strftime('%Y%m%d')}.json",
                mime="application/json",
                use_container_width=True
            )
        
        with col2:
            # CSV export
            df_export = pd.DataFrame(st.session_state.journal_entries)
            csv_data = df_export.to_csv(index=False)
            st.download_button(
                label="📥 Download as CSV",
                data=csv_data,
                file_name=f"sentiscope_journal_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv",
                use_container_width=True
            )

# ============================================================================
# MODE 2: BATCH ANALYZER
# ============================================================================

else:  # Batch Analyzer mode
    st.markdown("### Batch Sentiment Analyzer")
    st.markdown("Analyze multiple texts at once - perfect for reviews, tweets, and feedback")
    
    # Input methods
    input_method = st.radio(
        "Choose input method:",
        ["Paste Multiple Texts", "Upload CSV"],
        horizontal=True
    )
    
    texts_to_analyze = []
    
    if input_method == "Paste Multiple Texts":
        st.markdown("**Enter texts (one per line):**")
        batch_text = st.text_area(
            "Texts to analyze:",
            height=300,
            placeholder="This product is amazing!\nTerrible customer service.\nPretty good overall.\n...",
            help="Enter each text on a new line"
        )
        
        if batch_text.strip():
            texts_to_analyze = [line.strip() for line in batch_text.split('\n') if line.strip()]
    
    else:  # Upload CSV
        st.markdown("**Upload CSV file with a 'text' column:**")
        uploaded_file = st.file_uploader("Choose CSV file", type=['csv'])
        
        if uploaded_file:
            df_upload = pd.read_csv(uploaded_file)
            if 'text' in df_upload.columns:
                texts_to_analyze = df_upload['text'].dropna().tolist()
                st.success(f"Loaded {len(texts_to_analyze)} texts from CSV")
            else:
                st.error("CSV must have a 'text' column!")
    
    if texts_to_analyze:
        st.info(f"Ready to analyze {len(texts_to_analyze)} texts")
        
        if st.button("Analyze All Texts", use_container_width=True):
            with st.spinner(f"Analyzing {len(texts_to_analyze)} texts..."):
                results = []
                
                progress_bar = st.progress(0)
                for i, text in enumerate(texts_to_analyze):
                    if text and len(text) > 0:
                        sentiment_result = sentiment_analyzer(text[:512])[0]
                        results.append({
                            'text': text,
                            'sentiment': sentiment_result['label'],
                            'confidence': sentiment_result['score'],
                            'confidence_pct': sentiment_result['score'] * 100
                        })
                    progress_bar.progress((i + 1) / len(texts_to_analyze))
                
                st.session_state.batch_results = pd.DataFrame(results)
                st.success(f"Analysis complete! Processed {len(results)} texts.")
    
    # Display results
    if st.session_state.batch_results is not None and len(st.session_state.batch_results) > 0:
        df_results = st.session_state.batch_results
        
        st.divider()
        st.markdown("### Analysis Results")
        
        # Summary metrics
        col1, col2, col3, col4, col5 = st.columns(5)
        
        total = len(df_results)
        positive = len(df_results[df_results['sentiment'] == 'positive'])
        neutral = len(df_results[df_results['sentiment'] == 'neutral'])
        negative = len(df_results[df_results['sentiment'] == 'negative'])
        avg_conf = df_results['confidence_pct'].mean()
        
        col1.metric("Total Analyzed", total)
        col2.metric("Positive", positive, f"{positive/total*100:.1f}%")
        col3.metric("Neutral", neutral, f"{neutral/total*100:.1f}%")
        col4.metric("Negative", negative, f"{negative/total*100:.1f}%")
        col5.metric("Avg Confidence", f"{avg_conf:.1f}%")
        
        # Visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            sentiment_counts = df_results['sentiment'].value_counts()
            fig_pie = px.pie(
                values=sentiment_counts.values,
                names=sentiment_counts.index,
                title='Sentiment Distribution',
                color=sentiment_counts.index,
                color_discrete_map={'positive': '#10b981', 'neutral': '#f59e0b', 'negative': '#ef4444'}
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        
        with col2:
            fig_box = px.box(
                df_results,
                x='sentiment',
                y='confidence_pct',
                title='Confidence Distribution by Sentiment',
                color='sentiment',
                color_discrete_map={'positive': '#10b981', 'neutral': '#f59e0b', 'negative': '#ef4444'}
            )
            st.plotly_chart(fig_box, use_container_width=True)
        
        # Data table
        st.markdown("### Detailed Results")
        
        # Filter options
        filter_sentiment = st.multiselect(
            "Filter by sentiment:",
            options=['positive', 'neutral', 'negative'],
            default=['positive', 'neutral', 'negative']
        )
        
        df_filtered = df_results[df_results['sentiment'].isin(filter_sentiment)]
        
        st.dataframe(
            df_filtered[['text', 'sentiment', 'confidence_pct']].rename(columns={
                'text': 'Text',
                'sentiment': 'Sentiment',
                'confidence_pct': 'Confidence (%)'
            }),
            use_container_width=True,
            height=400
        )
        
        # Export
        csv = df_results.to_csv(index=False)
        st.download_button(
            label="Download Results as CSV",
            data=csv,
            file_name=f"sentiscope_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #64748b; font-size: 14px; padding: 2rem 0;">
    <p style="margin: 0;">
        <strong>SentiScope</strong> • Built with Streamlit & Hugging Face Transformers
    </p>
    <p style="margin: 4px 0 0 0;">
        Deep Learning sentiment analysis powered by Twitter-RoBERTa
    </p>
</div>
""", unsafe_allow_html=True)

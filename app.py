import streamlit as st
import time
from typing import Dict, List, Tuple, Optional
from utils import extract_text_from_file
from summarizer import generate_summary
from question_answering import ask_question as default_ask_question, highlight_text
from challenge_mode import generate_questions, evaluate_answer
from ollama_qa import OllamaQA
import os
import json
import time

USE_OLLAMA = False
qa_model = None

try:
    import requests
    response = requests.get('http://localhost:11434/api/tags', timeout=5)
    if response.status_code == 200:
        print("Ollama is running!")
        try:
            qa_model = OllamaQA(model_name="llama3:instruct")
            print(" Successfully connected to Ollama with llama3:instruct model")
            USE_OLLAMA = True
        except Exception as e:
            print(f" Could not load model: {e}")
            print("Trying to pull the model...")
            import subprocess
            try:
                subprocess.run(["ollama", "pull", "llama3:instruct"], check=True)
                qa_model = OllamaQA(model_name="llama3:instruct")
                print(" Successfully pulled and loaded llama3:instruct model")
                USE_OLLAMA = True
            except Exception as pull_error:
                print(f" Failed to pull model: {pull_error}")
                USE_OLLAMA = False
except Exception as e:
    print(f" Could not connect to Ollama: {e}")
    print("Please make sure Ollama is installed and running")
    print("You can download it from: https://ollama.ai/download")
    print("Falling back to default Hugging Face model...")

def ask_question(document_text: str, question: str) -> Dict:
    """Wrapper function to use either Ollama or default QA model"""
    if USE_OLLAMA:
        return qa_model.ask_question(document_text, question)
    return default_ask_question(document_text, question)

# Set page config with new theme
st.set_page_config(
    page_title="GenAI Research Assistant",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced Custom CSS
st.markdown("""
<style>
    :root {
        --primary: #4a6fa5;
        --secondary: #166088;
        --accent: #4fc3f7;
        --success: #28a745;
        --warning: #ffc107;
        --danger: #dc3545;
        --light: #f8f9fa;
        --dark: #343a40;
        --text: #2c3e50;
        --bg: #ffffff;
        --card-bg: #f8fafc;
    }
    
    .main {
        max-width: 1200px;
        padding: 2rem;
    }
    
    /* Header styling */
    .header {
        background: linear-gradient(135deg, var(--primary), var(--secondary));
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    .header h1 {
        font-size: 2.5rem;
        margin-bottom: 0.5rem;
    }
    
    .header p {
        font-size: 1.1rem;
        opacity: 0.9;
        
    }
    
    /* Sidebar styling */
    .sidebar .sidebar-content {
        background: var(--card-bg);
        border-right: 1px solid #e0e0e0;
    }
    
    /* Button styling */
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        font-weight: bold;
        padding: 0.5rem 1rem;
        transition: all 0.3s ease;
        border: none;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.15);
    }
    
    .stButton>button:active {
        transform: translateY(0);
    }
    
    /* Primary button */
    .stButton>button.primary {
        background: linear-gradient(135deg, var(--primary), var(--secondary));
        color: white;
    }
    
    /* Secondary button */
    .stButton>button.secondary {
        background: white;
        color: var(--primary);
        border: 1px solid var(--primary);
    }
    
    /* Text areas */
    .stTextArea>div>div>textarea {
        min-height: 150px;
        border-radius: 8px;
        border: 1px solid #e0e0e0;
        padding: 0.75rem;
    }
    
    /* Cards/expanders */
    .stExpander {
        background: var(--card-bg);
        border-radius: 10px;
        border: 1px solid #e0e0e0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        margin-bottom: 1rem;
    }
    
    .stExpander .streamlit-expanderHeader {
        font-weight: bold;
        color: var(--primary);
    }
    
    /* Metrics */
    .stMetric {
        background: white;
        border-radius: 10px;
        padding: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        border: 1px solid #e0e0e0;
    }
    
    .stMetric label {
        font-size: 0.9rem;
        color: var(--text);
        opacity: 0.8;
    }
    
    .stMetric div {
        font-size: 1.5rem;
        font-weight: bold;
        color: var(--primary);
    }
    
    /* Chat messages */
    .stChatMessage {
        border-radius: 10px;
        padding: 1rem;
        margin-bottom: 1rem;
    }
    
    .stChatMessage[data-testid="user"] {
        background-color: #e3f2fd;
        border-left: 4px solid var(--primary);
    }
    
    .stChatMessage[data-testid="assistant"] {
        background-color: var(--card-bg);
        border-left: 4px solid var(--accent);
    }
    
    /* Confidence indicators */
    .confidence-high {
        color: var(--success);
        font-weight: bold;
    }
    
    .confidence-medium {
        color: var(--warning);
        font-weight: bold;
    }
    
    .confidence-low {
        color: var(--danger);
        font-weight: bold;
    }
    
    /* Highlight */
    .highlight {
        background-color: #fff3cd;
        padding: 0.1em 0.2em;
        border-radius: 3px;
        font-weight: bold;
    }
    
    /* Divider */
    .divider {
        height: 1px;
        background: linear-gradient(to right, transparent, #e0e0e0, transparent);
        margin: 2rem 0;
    }
    
    /* Section headers */
    .section-header {
        color: var(--primary);
        font-size: 1.5rem;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid var(--accent);
    }
    
    /* Mode cards */
    .mode-card {
        background: white;
        border-radius: 10px;
        padding: 1.5rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        border: 1px solid #e0e0e0;
        margin-bottom: 1.5rem;
        transition: all 0.3s ease;
    }
    
    .mode-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 6px 12px rgba(0,0,0,0.1);
    }
    
    .mode-card h3 {
        color: var(--primary);
        margin-top: 0;
    }
    
    /* Progress spinner */
    .stSpinner>div {
        border-color: var(--primary) transparent transparent transparent;
    }
    
    /* Info boxes */
    .stAlert {
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state (keep existing code)
if 'document_text' not in st.session_state:
    st.session_state.document_text = ""
if 'summary' not in st.session_state:
    st.session_state.summary = ""
if 'questions' not in st.session_state:
    st.session_state.questions = []
if 'show_questions' not in st.session_state:
    st.session_state.show_questions = False
if 'user_answers' not in st.session_state:
    st.session_state.user_answers = {}
if 'show_results' not in st.session_state:
    st.session_state.show_results = False
if "messages" not in st.session_state:
    st.session_state.messages = []

# App title and description with new header
st.markdown("""
<div class="header">
    <h1>🧠 GenAI Research Assistant</h1>
    <p>Your intelligent partner for document analysis and comprehension</p>
</div>
""", unsafe_allow_html=True)

# File uploader in sidebar with enhanced styling
with st.sidebar:
    st.markdown("""
    <style>
        .sidebar .sidebar-content {
            padding: 1.5rem;
        }
        .sidebar .stFileUploader {
            margin-bottom: 1.5rem;
        }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown("### 📄 Document Upload")
    uploaded_file = st.file_uploader(
        "Upload a PDF or TXT document", 
        type=["pdf", "txt"],
        help="Upload a research paper, article, or any text document",
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    st.markdown("### About")
    st.markdown("""
    This GenAI Research Assistant helps you:
    - Summarize documents
    - Answer questions about content
    - Test your understanding
    """)
    
    st.markdown("---")
    st.markdown("### Model Status")
    if USE_OLLAMA:
        st.success("Using Ollama (llama3:instruct)")
    else:
        st.info("Using default Hugging Face model")

# Document processing (keep existing functionality)
if uploaded_file and not st.session_state.document_text:
    with st.spinner(" Processing your document..."):
        try:
            st.session_state.document_text = extract_text_from_file(uploaded_file)
            st.session_state.summary = generate_summary(st.session_state.document_text)
            st.session_state.questions = []
            st.session_state.user_answers = {}
            st.session_state.show_questions = False
            st.session_state.show_results = False
            
            st.success("Document processed successfully!")
        except Exception as e:
            st.error(f"Error processing document: {str(e)}")

# Document information card
if st.session_state.document_text:
    with st.expander("Document Information", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Word Count", f"{len(st.session_state.document_text.split()):,}")
        with col2:
            st.metric("Characters", f"{len(st.session_state.document_text):,}")
    
    # Summary section with enhanced styling
    with st.expander("Summary (≤ 250 words)", expanded=True):
        st.markdown(f"""
        <div style="
            background: #f8fafc;
            padding: 1.5rem;
            border-radius: 8px;
            border-left: 4px solid var(--accent);
                    color: #000000; 
        ">
            {st.session_state.summary}
        </div>
        """, unsafe_allow_html=True)

# Interactive Modes Section
st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
st.markdown('<h2 class="section-header">Interactive Modes</h2>', unsafe_allow_html=True)

# Mode selection cards
col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    <div class="mode-card">
        <h3>Ask Anything</h3>
        <p style="color: #2c3e50;">Get instant answers to your questions about the document content.</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="mode-card">
        <h3>Challenge Mode</h3>
        <p style="color: #2c3e50;">Test your understanding with generated questions and get feedback.</p>
    </div>
    """, unsafe_allow_html=True)
# Ask Anything Mode
st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
st.markdown('<h2 class="section-header">Ask Anything About the Document</h2>', unsafe_allow_html=True)

# Display chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"], unsafe_allow_html=True)

# Chat input with enhanced styling
if prompt := st.chat_input("Ask a question about the document..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        with st.spinner("Analyzing document..."):
            result = ask_question(st.session_state.document_text, prompt)
            is_comprehensive = result.get('is_comprehensive', False)
            
            if is_comprehensive:
                response = f"""
                <div style="margin-bottom: 1em;">
                    <div style="font-weight: bold; margin-bottom: 0.5em; color: var(--primary);">Answer:</div>
                    <div style="margin-bottom: 1em; white-space: pre-line; line-height: 1.6;">{result['answer']}</div>
                </div>
                """
            else:
                if result['confidence'] > 70:
                    confidence_class = "confidence-high"
                elif result['confidence'] > 30:
                    confidence_class = "confidence-medium"
                else:
                    confidence_class = "confidence-low"
                
                response = f"""
                <div style="margin-bottom: 1em;">
                    <div style="font-weight: bold; margin-bottom: 0.5em; color: var(--primary);">Answer:</div>
                    <div style="margin-bottom: 1em; line-height: 1.6;">{result['answer']}</div>
                    
                    <div style="display: flex; align-items: center; margin-bottom: 1em;">
                        <div style="font-weight: bold; margin-right: 0.5em;">Confidence:</div>
                        <span class="{confidence_class}">{result['confidence']}%</span>
                    </div>
                """
                
                if result.get('context'):
                    response += f"""
                    <details style="margin-top: 1em; border: 1px solid #e0e0e0; border-radius: 4px; padding: 0.5em;">
                        <summary style="font-weight: bold; cursor: pointer; padding: 0.5em; color: var(--primary);">
                            View Source Context
                        </summary>
                        <div style="
                            background: #f8f9fa;
                            border-left: 4px solid var(--accent);
                            padding: 0.5em 1em;
                            margin: 0.5em 0;
                            border-radius: 0 4px 4px 0;
                            white-space: pre-wrap;
                            font-size: 0.9em;
                            line-height: 1.5;
                        ">
                            {result['context']}
                        </div>
                    </details>
                    """
                
                response += "</div>"
            
            for chunk in response.split():
                full_response += chunk + " "
                time.sleep(0.05)
                message_placeholder.markdown(full_response + "▌", unsafe_allow_html=True)
            
            message_placeholder.markdown(full_response, unsafe_allow_html=True)
    
    st.session_state.messages.append({"role": "assistant", "content": response})
    st.rerun()

# Challenge Mode
st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
st.markdown('<h2 class="section-header">Challenge Mode: Test Your Understanding</h2>', unsafe_allow_html=True)

if not st.session_state.document_text:
    st.info("ℹPlease upload a document first to use Challenge Mode.")
else:
    if st.button("Generate Challenge Questions", key="generate_questions", use_container_width=True):
        with st.spinner("Creating challenging questions..."):
            try:
                st.session_state.questions = generate_questions(st.session_state.document_text)
                st.session_state.show_questions = True
                st.session_state.show_results = False
                st.session_state.user_answers = {}
                st.success("Challenge questions generated!")
                st.rerun()
            except Exception as e:
                st.error(f"Error generating questions: {str(e)}")
                st.session_state.show_questions = False

    if st.session_state.show_questions and st.session_state.questions:
        st.markdown("### Answer the following questions:")
        
        if isinstance(st.session_state.questions[0], str):
            st.session_state.questions = [
                {'question': q, 'context': st.session_state.document_text[:1000]}
                for q in st.session_state.questions
            ]
        
        for i, question_data in enumerate(st.session_state.questions):
            if isinstance(question_data, dict):
                question_text = question_data.get('question', 'No question text available')
                question_context = question_data.get('context', '')
            else:
                question_text = str(question_data)
                question_context = st.session_state.document_text[:1000]
                
            st.markdown(f"""
            <div style="
                background: var(--card-bg);
                padding: 1rem;
                border-radius: 8px;
                margin-bottom: 1rem;
                border-left: 4px solid var(--primary);
            ">
                <h4 style="margin-top: 0; color: var(--primary);">Q{i+1}: {question_text}</h4>
            </div>
            """, unsafe_allow_html=True)
            
            if i not in st.session_state.user_answers:
                st.session_state.user_answers[i] = {
                    'answer': '',
                    'evaluation': None,
                    'context': question_data.get('context', '')
                }
            
            answer = st.text_area(
                f"Your answer for Q{i+1}:",
                value=st.session_state.user_answers[i]['answer'],
                key=f"answer_{i}",
                height=100,
                disabled=st.session_state.show_results,
                label_visibility="collapsed"
            )
            
            st.session_state.user_answers[i]['answer'] = answer
            
            if st.session_state.show_results and st.session_state.user_answers[i].get('evaluation'):
                eval_data = st.session_state.user_answers[i]['evaluation']
                if eval_data['is_correct']:
                    st.success(f" {eval_data['feedback']}")
                else:
                    st.error(f" {eval_data['feedback']}")
                
                with st.expander("🔍 View Reference", expanded=False):
                    st.markdown("**Relevant Document Excerpt:**")
                    st.markdown(f"> {eval_data['reference']}", unsafe_allow_html=True)
                    
                    if 'full_context' in eval_data:
                        with st.expander("📖 View Full Context"):
                            st.markdown(eval_data['full_context'])
            
            st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        
        if not st.session_state.show_results:
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Submit All Answers", key="submit_answers", use_container_width=True):
                    if not all(ans['answer'].strip() for ans in st.session_state.user_answers.values()):
                        st.warning("Please answer all questions before submitting.")
                    else:
                        for i, question_data in enumerate(st.session_state.questions):
                            user_answer = st.session_state.user_answers[i]['answer']
                            evaluation = evaluate_answer(question_data, user_answer)
                            
                            if isinstance(question_data, dict):
                                context = question_data.get('context', '')
                                if not context and 'context' in question_data:
                                    context = question_data['context']
                                evaluation['full_context'] = context or st.session_state.document_text[:1000]
                            else:
                                evaluation['full_context'] = st.session_state.document_text[:1000]
                            
                            st.session_state.user_answers[i]['evaluation'] = evaluation
                        st.session_state.show_results = True
                        st.rerun()
            
            with col2:
                if st.button(" Reset Answers", key="reset_answers", use_container_width=True):
                    st.session_state.user_answers = {}
                    st.rerun()
        else:
            if st.button(" Try Again with New Questions", key="new_questions", use_container_width=True):
                st.session_state.show_questions = False
                st.session_state.show_results = False
                st.rerun()
import streamlit as st
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
import re

# Set page config
st.set_page_config(
    page_title="Answer Comparison Tool",
    page_icon="📊",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E3A8A;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #2563EB;
        margin-bottom: 0.5rem;
    }
    .metric-card {
        background-color: #F3F4F6;
        border-radius: 0.5rem;
        padding: 1rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.12);
    }
    .highlight {
        background-color: #DBEAFE;
        padding: 0.1rem 0.2rem;
        border-radius: 0.2rem;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<p class="main-header">Student-Teacher Answer Comparison Tool</p>', unsafe_allow_html=True)
st.markdown("Upload JSON files containing student and teacher answers to compare them using ROUGE and BLEU scores.")

# Initialize session state variables
if 'uploaded_files' not in st.session_state:
    st.session_state.uploaded_files = []
if 'comparison_results' not in st.session_state:
    st.session_state.comparison_results = []
if 'processing_done' not in st.session_state:
    st.session_state.processing_done = False

# Function to clean and tokenize text
def preprocess_text(text):
    # Convert to string if not already
    if not isinstance(text, str):
        text = str(text)
    
    # Replace LaTeX-style math with placeholders
    text = re.sub(r'\\[a-zA-Z]+', 'MATHSYMBOL', text)
    
    # Replace Unicode math symbols with placeholders
    text = re.sub(r'[√∛∜∫∬∭∮∯∰∱∲∳⨌]', 'MATHSYMBOL', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

# Function to calculate BLEU score
def calculate_bleu(reference, candidate):
    if not reference or not candidate:
        return 0.0
    
    # Preprocess texts
    ref_processed = preprocess_text(reference)
    cand_processed = preprocess_text(candidate)
    
    # Tokenize
    reference_tokens = ref_processed.split()
    candidate_tokens = cand_processed.split()
    
    if not reference_tokens or not candidate_tokens:
        return 0.0
    
    # Calculate BLEU score with smoothing
    smoothie = SmoothingFunction().method1
    
    # Calculate for different n-gram weights
    weights_1 = (1, 0, 0, 0)  # Unigrams only
    weights_2 = (0.5, 0.5, 0, 0)  # Bigrams and unigrams
    weights_3 = (0.33, 0.33, 0.33, 0)  # Up to trigrams
    weights_4 = (0.25, 0.25, 0.25, 0.25)  # Up to 4-grams
    
    try:
        bleu_1 = sentence_bleu([reference_tokens], candidate_tokens, weights=weights_1, smoothing_function=smoothie)
        bleu_2 = sentence_bleu([reference_tokens], candidate_tokens, weights=weights_2, smoothing_function=smoothie)
        bleu_3 = sentence_bleu([reference_tokens], candidate_tokens, weights=weights_3, smoothing_function=smoothie)
        bleu_4 = sentence_bleu([reference_tokens], candidate_tokens, weights=weights_4, smoothing_function=smoothie)
        
        return {
            'bleu_1': bleu_1,
            'bleu_2': bleu_2,
            'bleu_3': bleu_3,
            'bleu_4': bleu_4,
            'bleu_avg': (bleu_1 + bleu_2 + bleu_3 + bleu_4) / 4
        }
    except Exception as e:
        st.error(f"Error calculating BLEU score: {e}")
        return {
            'bleu_1': 0.0,
            'bleu_2': 0.0,
            'bleu_3': 0.0,
            'bleu_4': 0.0,
            'bleu_avg': 0.0
        }

# Function to calculate ROUGE scores
def calculate_rouge(reference, candidate):
    if not reference or not candidate:
        return {
            'rouge1_precision': 0.0,
            'rouge1_recall': 0.0,
            'rouge1_f1': 0.0,
            'rougeL_precision': 0.0,
            'rougeL_recall': 0.0,
            'rougeL_f1': 0.0
        }
    
    # Preprocess texts
    ref_processed = preprocess_text(reference)
    cand_processed = preprocess_text(candidate)
    
    # Initialize Rouge scorer
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    
    try:
        scores = scorer.score(ref_processed, cand_processed)
        
        return {
            'rouge1_precision': scores['rouge1'].precision,
            'rouge1_recall': scores['rouge1'].recall,
            'rouge1_f1': scores['rouge1'].fmeasure,
            'rougeL_precision': scores['rougeL'].precision,
            'rougeL_recall': scores['rougeL'].recall,
            'rougeL_f1': scores['rougeL'].fmeasure
        }
    except Exception as e:
        st.error(f"Error calculating ROUGE score: {e}")
        return {
            'rouge1_precision': 0.0,
            'rouge1_recall': 0.0,
            'rouge1_f1': 0.0,
            'rougeL_precision': 0.0,
            'rougeL_recall': 0.0,
            'rougeL_f1': 0.0
        }

# Function to process uploaded JSON files
def process_files(uploaded_files):
    results = []
    
    for uploaded_file in uploaded_files:
        try:
            # Read JSON content
            content = uploaded_file.read()
            data = json.loads(content)
            
            # Extract relevant fields
            question = data.get('question', '')
            teacher_answer = data.get('Teacher_Answer', '')
            student_answer = data.get('Student_Answer', '')
            accuracy = data.get('Accuracy', '')
            overall_score = data.get('Overall Score', '')
            max_marks = data.get('MaximumMarks', '')
            
            # Calculate BLEU scores
            bleu_scores = calculate_bleu(teacher_answer, student_answer)
            
            # Calculate ROUGE scores
            rouge_scores = calculate_rouge(teacher_answer, student_answer)
            
            # Combine results
            result = {
                'filename': uploaded_file.name,
                'question': question,
                'teacher_answer': teacher_answer,
                'student_answer': student_answer,
                'accuracy': accuracy,
                'overall_score': overall_score,
                'max_marks': max_marks,
                **bleu_scores,
                **rouge_scores
            }
            
            results.append(result)
            
        except Exception as e:
            st.error(f"Error processing file {uploaded_file.name}: {e}")
    
    return results

# File uploader
with st.expander("Upload Files", expanded=True):
    uploaded_files = st.file_uploader(
        "Upload JSON files containing student and teacher answers",
        type=["json"],
        accept_multiple_files=True
    )
    
    col1, col2 = st.columns([1, 5])
    with col1:
        process_button = st.button("Process Files", type="primary")
    
    if process_button and uploaded_files:
        st.session_state.uploaded_files = uploaded_files
        with st.spinner("Processing files..."):
            st.session_state.comparison_results = process_files(uploaded_files)
            st.session_state.processing_done = True
            st.success(f"Successfully processed {len(uploaded_files)} files!")

# Display results if processing is done
if st.session_state.processing_done and st.session_state.comparison_results:
    results = st.session_state.comparison_results
    
    # Summary metrics
    st.markdown('<p class="sub-header">Summary Metrics</p>', unsafe_allow_html=True)
    
    # Create summary dataframe
    summary_df = pd.DataFrame([{
        'Filename': result['filename'],
        'BLEU-1': result['bleu_1'],
        'BLEU-2': result['bleu_2'],
        'BLEU-4': result['bleu_4'],
        'ROUGE-1 F1': result['rouge1_f1'],
        'ROUGE-L F1': result['rougeL_f1'],
        'Accuracy': result['accuracy'],
        'Score': f"{result['overall_score']}/{result['max_marks']}" if result['max_marks'] else result['overall_score']
    } for result in results])
    
    st.dataframe(summary_df, use_container_width=True)
    
    # Visualizations
    st.markdown('<p class="sub-header">Visualizations</p>', unsafe_allow_html=True)
    
    if len(results) > 1:
        # Multiple files - show comparisons
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # BLEU scores
        bleu_df = pd.DataFrame([{
            'Filename': result['filename'],
            'BLEU-1': result['bleu_1'],
            'BLEU-2': result['bleu_2'],
            'BLEU-3': result['bleu_3'],
            'BLEU-4': result['bleu_4'],
        } for result in results])
        
        bleu_df_melted = pd.melt(bleu_df, id_vars=['Filename'], var_name='Metric', value_name='Score')
        sns.barplot(x='Filename', y='Score', hue='Metric', data=bleu_df_melted, ax=ax1)
        ax1.set_title('BLEU Scores Comparison')
        ax1.set_ylim(0, 1)
        ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha='right')
        
        # ROUGE scores
        rouge_df = pd.DataFrame([{
            'Filename': result['filename'],
            'ROUGE-1 F1': result['rouge1_f1'],
            'ROUGE-L F1': result['rougeL_f1'],
        } for result in results])
        
        rouge_df_melted = pd.melt(rouge_df, id_vars=['Filename'], var_name='Metric', value_name='Score')
        sns.barplot(x='Filename', y='Score', hue='Metric', data=rouge_df_melted, ax=ax2)
        ax2.set_title('ROUGE Scores Comparison')
        ax2.set_ylim(0, 1)
        ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, ha='right')
        
        plt.tight_layout()
        st.pyplot(fig)
    
    # Detailed results for each file
    st.markdown('<p class="sub-header">Detailed Results</p>', unsafe_allow_html=True)
    
    for i, result in enumerate(results):
        with st.expander(f"File: {result['filename']}", expanded=i==0):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Question:**")
                st.markdown(f"<div class='metric-card'>{result['question']}</div>", unsafe_allow_html=True)
                
                st.markdown("**Teacher's Answer:**")
                st.markdown(f"<div class='metric-card'>{result['teacher_answer']}</div>", unsafe_allow_html=True)
                
                st.markdown("**Student's Answer:**")
                st.markdown(f"<div class='metric-card'>{result['student_answer']}</div>", unsafe_allow_html=True)
            
            with col2:
                # BLEU scores
                st.markdown("**BLEU Scores:**")
                bleu_df = pd.DataFrame({
                    'Metric': ['BLEU-1', 'BLEU-2', 'BLEU-3', 'BLEU-4', 'BLEU-Average'],
                    'Score': [
                        result['bleu_1'],
                        result['bleu_2'],
                        result['bleu_3'],
                        result['bleu_4'],
                        result['bleu_avg']
                    ]
                })
                st.dataframe(bleu_df, use_container_width=True)
                
                # ROUGE scores
                st.markdown("**ROUGE Scores:**")
                rouge_df = pd.DataFrame({
                    'Metric': ['ROUGE-1 Precision', 'ROUGE-1 Recall', 'ROUGE-1 F1', 
                              'ROUGE-L Precision', 'ROUGE-L Recall', 'ROUGE-L F1'],
                    'Score': [
                        result['rouge1_precision'],
                        result['rouge1_recall'],
                        result['rouge1_f1'],
                        result['rougeL_precision'],
                        result['rougeL_recall'],
                        result['rougeL_f1']
                    ]
                })
                st.dataframe(rouge_df, use_container_width=True)
                
                # Visualization
                fig, ax = plt.subplots(figsize=(8, 4))
                metrics = ['BLEU-1', 'BLEU-2', 'BLEU-3', 'BLEU-4', 'ROUGE-1 F1', 'ROUGE-L F1']
                scores = [
                    result['bleu_1'], 
                    result['bleu_2'], 
                    result['bleu_3'], 
                    result['bleu_4'], 
                    result['rouge1_f1'], 
                    result['rougeL_f1']
                ]
                
                bars = ax.bar(metrics, scores, color=['#1F77B4', '#1F77B4', '#1F77B4', '#1F77B4', '#FF7F0E', '#FF7F0E'])
                ax.set_ylim(0, 1)
                ax.set_title('Similarity Metrics')
                ax.set_ylabel('Score')
                
                # Add value labels
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                            f'{height:.2f}', ha='center', va='bottom')
                
                plt.xticks(rotation=45)
                plt.tight_layout()
                st.pyplot(fig)

# Settings and options
with st.sidebar:
    st.markdown("## Options")
    st.markdown("### Display Settings")
    
    # Theme options
    theme = st.selectbox(
        "Theme",
        ["Light", "Dark"],
        index=0
    )
    
    # Comparison method
    comparison_method = st.radio(
        "Primary Comparison Method",
        ["BLEU Score", "ROUGE Score", "Both Equally"],
        index=2
    )
    
    # Visualization type
    viz_type = st.selectbox(
        "Visualization Type",
        ["Bar Chart", "Radar Chart", "Heatmap"],
        index=0
    )
    
    st.markdown("### Advanced Settings")
    
    # BLEU settings
    st.markdown("**BLEU Settings**")
    bleu_smoothing = st.checkbox("Use Smoothing", value=True)
    
    # ROUGE settings
    st.markdown("**ROUGE Settings**")
    use_stemmer = st.checkbox("Use Stemmer", value=True)
    
    # Export options
    st.markdown("### Export Options")
    export_format = st.selectbox(
        "Export Format",
        ["CSV", "Excel", "JSON"],
        index=0
    )
    
    if st.session_state.processing_done and st.session_state.comparison_results:
        if st.button("Export Results"):
            # This would actually export the data in a real app
            st.success("Export feature would be implemented here!")
    
    # About section
    st.markdown("---")
    st.markdown("### About")
    st.markdown("""
    This tool helps educators compare student answers with teacher answers 
    using NLP metrics like BLEU and ROUGE scores.
    
    Built with Streamlit, NLTK, and Rouge-score.
    """)

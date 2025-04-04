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
    page_title="Model Prediction Comparison",
    page_icon="📊",
    layout="wide"
)

# Add custom styling
st.markdown("""
<style>
    .header-text {
        font-size: 2rem;
        font-weight: bold;
        color: #1E3A8A;
    }
    .metric-container {
        background-color: #F3F4F6;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        text-align: center;
    }
    .metric-label {
        font-size: 1rem;
        text-align: center;
        color: #6B7280;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<p class="header-text">Model Prediction Comparison Tool</p>', unsafe_allow_html=True)
st.markdown("Compare model predictions with golden data using ROUGE and BLEU scores")

# Function to preprocess text
def preprocess_text(text):
    if not isinstance(text, str):
        text = str(text)
    
    # Replace special characters and normalize whitespace
    text = re.sub(r'\\[a-zA-Z]+', ' ', text)  # Remove LaTeX commands
    text = re.sub(r'[^\w\s.,?!]', ' ', text)  # Keep basic punctuation
    text = re.sub(r'\s+', ' ', text).strip()  # Normalize whitespace
    
    return text

# Function to calculate BLEU score
def calculate_bleu(reference, candidate):
    if not reference or not candidate:
        return {
            'bleu_1': 0.0,
            'bleu_2': 0.0,
            'bleu_3': 0.0,
            'bleu_4': 0.0
        }
    
    # Preprocess texts
    ref_processed = preprocess_text(reference)
    cand_processed = preprocess_text(candidate)
    
    # Tokenize
    reference_tokens = ref_processed.split()
    candidate_tokens = cand_processed.split()
    
    if not reference_tokens or not candidate_tokens:
        return {
            'bleu_1': 0.0,
            'bleu_2': 0.0,
            'bleu_3': 0.0,
            'bleu_4': 0.0
        }
    
    # Calculate BLEU score with smoothing
    smoothie = SmoothingFunction().method1
    
    try:
        bleu_1 = sentence_bleu([reference_tokens], candidate_tokens, weights=(1, 0, 0, 0), smoothing_function=smoothie)
        bleu_2 = sentence_bleu([reference_tokens], candidate_tokens, weights=(0.5, 0.5, 0, 0), smoothing_function=smoothie)
        bleu_3 = sentence_bleu([reference_tokens], candidate_tokens, weights=(0.33, 0.33, 0.33, 0), smoothing_function=smoothie)
        bleu_4 = sentence_bleu([reference_tokens], candidate_tokens, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothie)
        
        return {
            'bleu_1': bleu_1,
            'bleu_2': bleu_2,
            'bleu_3': bleu_3,
            'bleu_4': bleu_4
        }
    except Exception as e:
        st.error(f"Error calculating BLEU score: {e}")
        return {
            'bleu_1': 0.0,
            'bleu_2': 0.0,
            'bleu_3': 0.0,
            'bleu_4': 0.0
        }

# Function to calculate ROUGE scores
def calculate_rouge(reference, candidate):
    if not reference or not candidate:
        return {
            'rouge1_f1': 0.0,
            'rouge2_f1': 0.0,
            'rougeL_f1': 0.0
        }
    
    # Preprocess texts
    ref_processed = preprocess_text(reference)
    cand_processed = preprocess_text(candidate)
    
    # Initialize Rouge scorer
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    
    try:
        scores = scorer.score(ref_processed, cand_processed)
        
        return {
            'rouge1_f1': scores['rouge1'].fmeasure,
            'rouge2_f1': scores['rouge2'].fmeasure,
            'rougeL_f1': scores['rougeL'].fmeasure
        }
    except Exception as e:
        st.error(f"Error calculating ROUGE score: {e}")
        return {
            'rouge1_f1': 0.0,
            'rouge2_f1': 0.0,
            'rougeL_f1': 0.0
        }

# Main app layout
st.sidebar.header("Upload Options")

# File upload widgets
golden_file = st.sidebar.file_uploader("Upload Golden Data (JSON)", type="json")
prediction_file = st.sidebar.file_uploader("Upload Model Predictions (JSON)", type="json")

# Comparison field selection
if golden_file and prediction_file:
    # Load data
    try:
        golden_data = json.load(golden_file)
        prediction_data = json.load(prediction_file)
        
        # If data is loaded as a list of dictionaries
        if isinstance(golden_data, list) and len(golden_data) > 0:
            # Show first item for field selection
            sample_golden = golden_data[0]
        else:
            # Single item case
            sample_golden = golden_data
            
        if isinstance(prediction_data, list) and len(prediction_data) > 0:
            sample_prediction = prediction_data[0]
        else:
            sample_prediction = prediction_data
        
        # Get available fields
        golden_fields = list(sample_golden.keys())
        prediction_fields = list(sample_prediction.keys())
        
        # Field selection
        st.sidebar.subheader("Select Fields to Compare")
        
        golden_field = st.sidebar.selectbox(
            "Golden Data Field",
            golden_fields,
            index=golden_fields.index("Teacher_Answer") if "Teacher_Answer" in golden_fields else 0
        )
        
        prediction_field = st.sidebar.selectbox(
            "Prediction Field",
            prediction_fields,
            index=prediction_fields.index("Student_Answer") if "Student_Answer" in prediction_fields else 0
        )
        
        # ID field for matching items (if data is in list format)
        has_multiple_items = (isinstance(golden_data, list) and len(golden_data) > 1) or \
                            (isinstance(prediction_data, list) and len(prediction_data) > 1)
        
        if has_multiple_items:
            st.sidebar.subheader("Match Items By")
            common_fields = list(set(golden_fields).intersection(set(prediction_fields)))
            id_field = st.sidebar.selectbox(
                "ID Field", 
                common_fields,
                index=common_fields.index("question") if "question" in common_fields else 0
            )
        else:
            id_field = None
        
        # Process data and generate comparison
        if st.sidebar.button("Run Comparison", type="primary"):
            with st.spinner("Processing data..."):
                results = []
                
                # Prepare data for processing
                if has_multiple_items:
                    # Create dictionaries keyed by ID field for easy matching
                    golden_dict = {item.get(id_field, i): item for i, item in enumerate(golden_data)} if isinstance(golden_data, list) else {0: golden_data}
                    prediction_dict = {item.get(id_field, i): item for i, item in enumerate(prediction_data)} if isinstance(prediction_data, list) else {0: prediction_data}
                    
                    # Find matching items
                    all_ids = set(golden_dict.keys()).intersection(set(prediction_dict.keys()))
                    
                    for item_id in all_ids:
                        golden_item = golden_dict[item_id]
                        prediction_item = prediction_dict[item_id]
                        
                        golden_text = golden_item.get(golden_field, "")
                        prediction_text = prediction_item.get(prediction_field, "")
                        
                        # Calculate scores
                        bleu_scores = calculate_bleu(golden_text, prediction_text)
                        rouge_scores = calculate_rouge(golden_text, prediction_text)
                        
                        results.append({
                            'id': item_id,
                            'golden_text': golden_text,
                            'prediction_text': prediction_text,
                            **bleu_scores,
                            **rouge_scores
                        })
                else:
                    # Single item comparison
                    golden_text = sample_golden.get(golden_field, "")
                    prediction_text = sample_prediction.get(prediction_field, "")
                    
                    # Calculate scores
                    bleu_scores = calculate_bleu(golden_text, prediction_text)
                    rouge_scores = calculate_rouge(golden_text, prediction_text)
                    
                    results.append({
                        'id': 'Single item',
                        'golden_text': golden_text,
                        'prediction_text': prediction_text,
                        **bleu_scores,
                        **rouge_scores
                    })
                
                # Display results
                if results:
                    # Overall metrics
                    avg_bleu1 = np.mean([r['bleu_1'] for r in results])
                    avg_bleu4 = np.mean([r['bleu_4'] for r in results])
                    avg_rouge1 = np.mean([r['rouge1_f1'] for r in results])
                    avg_rougeL = np.mean([r['rougeL_f1'] for r in results])
                    
                    # Display summary metrics
                    st.subheader("Summary Metrics")
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.markdown('<div class="metric-container">', unsafe_allow_html=True)
                        st.markdown(f'<p class="metric-value">{avg_bleu1:.3f}</p>', unsafe_allow_html=True)
                        st.markdown('<p class="metric-label">Average BLEU-1</p>', unsafe_allow_html=True)
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    with col2:
                        st.markdown('<div class="metric-container">', unsafe_allow_html=True)
                        st.markdown(f'<p class="metric-value">{avg_bleu4:.3f}</p>', unsafe_allow_html=True)
                        st.markdown('<p class="metric-label">Average BLEU-4</p>', unsafe_allow_html=True)
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    with col3:
                        st.markdown('<div class="metric-container">', unsafe_allow_html=True)
                        st.markdown(f'<p class="metric-value">{avg_rouge1:.3f}</p>', unsafe_allow_html=True)
                        st.markdown('<p class="metric-label">Average ROUGE-1</p>', unsafe_allow_html=True)
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    with col4:
                        st.markdown('<div class="metric-container">', unsafe_allow_html=True)
                        st.markdown(f'<p class="metric-value">{avg_rougeL:.3f}</p>', unsafe_allow_html=True)
                        st.markdown('<p class="metric-label">Average ROUGE-L</p>', unsafe_allow_html=True)
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Visualizations
                    st.subheader("Metric Visualization")
                    
                    # Create visualization dataframe
                    viz_data = pd.DataFrame([{
                        'ID': r['id'],
                        'BLEU-1': r['bleu_1'],
                        'BLEU-4': r['bleu_4'],
                        'ROUGE-1': r['rouge1_f1'],
                        'ROUGE-L': r['rougeL_f1']
                    } for r in results])
                    
                    # Create bar chart for multiple items or radar chart for single item
                    if len(results) > 1:
                        # Reshape for seaborn
                        viz_long = pd.melt(
                            viz_data, 
                            id_vars=['ID'], 
                            value_vars=['BLEU-1', 'BLEU-4', 'ROUGE-1', 'ROUGE-L'],
                            var_name='Metric', 
                            value_name='Score'
                        )
                        
                        # Create bar chart
                        fig, ax = plt.subplots(figsize=(10, 6))
                        sns.barplot(data=viz_long, x='ID', y='Score', hue='Metric', ax=ax)
                        ax.set_ylim(0, 1)
                        ax.set_title('Metric Comparison Across Items')
                        ax.set_ylabel('Score')
                        ax.set_xlabel('Item ID')
                        if len(results) > 5:  # Rotate labels if many items
                            plt.xticks(rotation=45, ha='right')
                        plt.tight_layout()
                        st.pyplot(fig)
                        
                        # Add histogram of scores
                        st.subheader("Score Distribution")
                        hist_data = pd.melt(
                            viz_data, 
                            value_vars=['BLEU-1', 'BLEU-4', 'ROUGE-1', 'ROUGE-L'],
                            var_name='Metric', 
                            value_name='Score'
                        )
                        
                        fig, ax = plt.subplots(figsize=(10, 6))
                        sns.histplot(data=hist_data, x='Score', hue='Metric', kde=True, bins=20, ax=ax)
                        ax.set_title('Distribution of Scores')
                        ax.set_xlim(0, 1)
                        plt.tight_layout()
                        st.pyplot(fig)
                    else:
                        # Radar chart for single item
                        categories = ['BLEU-1', 'BLEU-4', 'ROUGE-1', 'ROUGE-L']
                        values = [
                            results[0]['bleu_1'],
                            results[0]['bleu_4'],
                            results[0]['rouge1_f1'],
                            results[0]['rougeL_f1']
                        ]
                        
                        # Create radar chart
                        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
                        
                        # Compute angle for each category
                        angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False).tolist()
                        values += values[:1]  # Close the polygon
                        angles += angles[:1]  # Close the polygon
                        categories += categories[:1]  # Close the polygon
                        
                        ax.plot(angles, values, 'o-', linewidth=2)
                        ax.fill(angles, values, alpha=0.25)
                        ax.set_thetagrids(np.degrees(angles[:-1]), categories[:-1])
                        ax.set_ylim(0, 1)
                        ax.grid(True)
                        ax.set_title('Metric Scores')
                        plt.tight_layout()
                        st.pyplot(fig)
                    
                    # Detailed results
                    st.subheader("Detailed Results")
                    
                    for i, result in enumerate(results):
                        with st.expander(f"Item {result['id']}", expanded=i==0):
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.markdown("**Golden Data:**")
                                st.text_area("", value=result['golden_text'], height=150, key=f"golden_{i}")
                            
                            with col2:
                                st.markdown("**Model Prediction:**")
                                st.text_area("", value=result['prediction_text'], height=150, key=f"pred_{i}")
                            
                            # Metrics table
                            st.markdown("**Similarity Metrics:**")
                            metrics_df = pd.DataFrame({
                                'Metric': ['BLEU-1', 'BLEU-2', 'BLEU-3', 'BLEU-4', 
                                          'ROUGE-1 (F1)', 'ROUGE-2 (F1)', 'ROUGE-L (F1)'],
                                'Score': [
                                    result['bleu_1'],
                                    result['bleu_2'],
                                    result['bleu_3'],
                                    result['bleu_4'],
                                    result['rouge1_f1'],
                                    result['rouge2_f1'],
                                    result['rougeL_f1']
                                ]
                            })
                            st.dataframe(metrics_df, use_container_width=True)
                    
                    # Export options
                    st.subheader("Export Results")
                    export_format = st.selectbox("Export Format", ["CSV", "Excel", "JSON"])
                    
                    if st.button("Export Data"):
                        # Convert results to a more export-friendly format
                        export_data = pd.DataFrame([{
                            'ID': r['id'],
                            'Golden_Text': r['golden_text'],
                            'Prediction_Text': r['prediction_text'],
                            'BLEU_1': r['bleu_1'],
                            'BLEU_2': r['bleu_2'],
                            'BLEU_3': r['bleu_3'],
                            'BLEU_4': r['bleu_4'],
                            'ROUGE_1_F1': r['rouge1_f1'],
                            'ROUGE_2_F1': r['rouge2_f1'],
                            'ROUGE_L_F1': r['rougeL_f1']
                        } for r in results])
                        
                        if export_format == "CSV":
                            st.download_button(
                                label="Download CSV",
                                data=export_data.to_csv(index=False),
                                file_name="prediction_comparison_results.csv",
                                mime="text/csv"
                            )
                        elif export_format == "Excel":
                            # In a real app, you would use a BytesIO buffer with pandas to_excel
                            st.error("Excel export would be implemented here in a production app")
                        elif export_format == "JSON":
                            st.download_button(
                                label="Download JSON",
                                data=json.dumps(results, indent=2),
                                file_name="prediction_comparison_results.json",
                                mime="application/json"
                            )
                    
    except Exception as e:
        st.error(f"Error processing files: {e}")
        st.error("Please make sure the uploaded files are valid JSON")
else:
    # Show instructions when files are not yet uploaded
    st.info("Please upload golden data and model prediction files to begin comparison")
    
    st.markdown("""
    ### How to Use This Tool
    
    1. Upload your golden data (ground truth) JSON file using the sidebar
    2. Upload your model predictions JSON file using the sidebar
    3. Select the fields to compare from each file
    4. If your files contain multiple items, select a common ID field to match them
    5. Click "Run Comparison" to generate metrics and visualizations
    
    ### Supported Metrics
    
    - **BLEU scores** (1-gram to 4-gram)
    - **ROUGE scores** (ROUGE-1, ROUGE-2, ROUGE-L)
    
    ### Expected JSON Format
    
    Your JSON files should contain either a single object or an array of objects.
    Each object should have fields for the text you want to compare and ideally a common ID field.
    
    Example format:
    ```json
    [
      {
        "question": "What is the capital of France?",
        "Teacher_Answer": "The capital of France is Paris.",
        "Student_Answer": "Paris is the capital of France."
      },
      ...
    ]
    ```
    """)

# Footer
st.markdown("---")
st.markdown(
    "This tool evaluates model predictions against golden data using standard NLP metrics."
)

import streamlit as st
import json
import pandas as pd
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
import nltk
import re

# Download NLTK data if needed
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# Page setup
st.set_page_config(page_title="Text Comparison Scores", layout="wide")
st.title("ROUGE and BLEU Score Calculator")

# Text preprocessing function
def preprocess_text(text):
    if not isinstance(text, str):
        text = str(text)
    
    # Clean special characters and normalize whitespace
    text = re.sub(r'\\[a-zA-Z]+', ' ', text)  # Remove LaTeX commands
    text = re.sub(r'\s+', ' ', text).strip()  # Normalize whitespace
    
    return text

# Score calculation functions
def calculate_bleu(reference, candidate):
    ref_processed = preprocess_text(reference)
    cand_processed = preprocess_text(candidate)
    
    # Tokenize
    reference_tokens = ref_processed.split()
    candidate_tokens = cand_processed.split()
    
    if not reference_tokens or not candidate_tokens:
        return 0.0, 0.0, 0.0, 0.0
    
    # Calculate BLEU scores with smoothing
    smoothie = SmoothingFunction().method1
    
    bleu_1 = sentence_bleu([reference_tokens], candidate_tokens, weights=(1, 0, 0, 0), smoothing_function=smoothie)
    bleu_2 = sentence_bleu([reference_tokens], candidate_tokens, weights=(0.5, 0.5, 0, 0), smoothing_function=smoothie)
    bleu_3 = sentence_bleu([reference_tokens], candidate_tokens, weights=(0.33, 0.33, 0.33, 0), smoothing_function=smoothie)
    bleu_4 = sentence_bleu([reference_tokens], candidate_tokens, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothie)
    
    return bleu_1, bleu_2, bleu_3, bleu_4

def calculate_rouge(reference, candidate):
    ref_processed = preprocess_text(reference)
    cand_processed = preprocess_text(candidate)
    
    # Initialize Rouge scorer
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(ref_processed, cand_processed)
    
    return scores['rouge1'].fmeasure, scores['rouge2'].fmeasure, scores['rougeL'].fmeasure

# File upload section
st.header("Upload Files")
col1, col2 = st.columns(2)

with col1:
    golden_file = st.file_uploader("Upload Golden Data (JSON)", type="json")

with col2:
    prediction_file = st.file_uploader("Upload Model Predictions (JSON)", type="json")

# Process files when both are uploaded
if golden_file and prediction_file:
    try:
        # Load the JSON data
        golden_data = json.load(golden_file)
        prediction_data = json.load(prediction_file)
        
        # Handle different data structures
        if isinstance(golden_data, list):
            golden_items = golden_data
        else:
            golden_items = [golden_data]
            
        if isinstance(prediction_data, list):
            prediction_items = prediction_data
        else:
            prediction_items = [prediction_data]
        
        # Get field options
        golden_fields = list(golden_items[0].keys())
        prediction_fields = list(prediction_items[0].keys())
        
        # Field selection
        st.header("Select Fields to Compare")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            golden_field = st.selectbox(
                "Golden Data Field",
                golden_fields,
                index=golden_fields.index("Teacher_Answer") if "Teacher_Answer" in golden_fields else 0
            )
        
        with col2:
            prediction_field = st.selectbox(
                "Prediction Field",
                prediction_fields,
                index=prediction_fields.index("Student_Answer") if "Student_Answer" in prediction_fields else 0
            )
        
        with col3:
            id_field = st.selectbox(
                "ID Field (for matching items)",
                [field for field in golden_fields if field in prediction_fields],
                index=0
            )
        
        # Calculate scores button
        if st.button("Calculate Scores", type="primary"):
            with st.spinner("Calculating scores..."):
                # Create dictionaries for easy matching
                golden_dict = {item.get(id_field, i): item for i, item in enumerate(golden_items)}
                prediction_dict = {item.get(id_field, i): item for i, item in enumerate(prediction_items)}
                
                # Find common IDs
                common_ids = set(golden_dict.keys()).intersection(set(prediction_dict.keys()))
                
                if not common_ids:
                    st.error("No matching items found between the golden data and predictions.")
                else:
                    # Calculate scores for each matching item
                    results = []
                    
                    for item_id in common_ids:
                        golden_text = golden_dict[item_id].get(golden_field, "")
                        prediction_text = prediction_dict[item_id].get(prediction_field, "")
                        
                        # Skip empty texts
                        if not golden_text or not prediction_text:
                            continue
                        
                        # Calculate scores
                        bleu_1, bleu_2, bleu_3, bleu_4 = calculate_bleu(golden_text, prediction_text)
                        rouge1_f1, rouge2_f1, rougeL_f1 = calculate_rouge(golden_text, prediction_text)
                        
                        results.append({
                            'id': item_id,
                            'bleu_1': bleu_1,
                            'bleu_2': bleu_2,
                            'bleu_3': bleu_3,
                            'bleu_4': bleu_4,
                            'rouge1_f1': rouge1_f1,
                            'rouge2_f1': rouge2_f1,
                            'rougeL_f1': rougeL_f1
                        })
                    
                    if results:
                        # Display results as a table
                        st.header("Comparison Scores")
                        
                        # Create dataframe for display
                        df = pd.DataFrame([{
                            'ID': r['id'],
                            'BLEU-1': r['bleu_1'],
                            'BLEU-2': r['bleu_2'],
                            'BLEU-3': r['bleu_3'],
                            'BLEU-4': r['bleu_4'],
                            'ROUGE-1': r['rouge1_f1'],
                            'ROUGE-2': r['rouge2_f1'],
                            'ROUGE-L': r['rougeL_f1']
                        } for r in results])
                        
                        st.dataframe(df)
                        
                        # Show averages
                        st.header("Average Scores")
                        
                        avg_df = pd.DataFrame({
                            'Metric': ['BLEU-1', 'BLEU-2', 'BLEU-3', 'BLEU-4', 'ROUGE-1', 'ROUGE-2', 'ROUGE-L'],
                            'Average Score': [
                                df['BLEU-1'].mean(),
                                df['BLEU-2'].mean(),
                                df['BLEU-3'].mean(),
                                df['BLEU-4'].mean(),
                                df['ROUGE-1'].mean(),
                                df['ROUGE-2'].mean(),
                                df['ROUGE-L'].mean()
                            ]
                        })
                        
                        st.dataframe(avg_df)
                        
                        # Export options
                        st.download_button(
                            "Download Results as CSV",
                            data=df.to_csv(index=False),
                            file_name="comparison_scores.csv",
                            mime="text/csv"
                        )
                    else:
                        st.error("No valid text comparisons could be made.")
    
    except Exception as e:
        st.error(f"Error processing files: {e}")
else:
    st.info("Please upload both golden data and model prediction files to calculate scores.")

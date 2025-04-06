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
        
        col1, col2 = st.columns(2)
        
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
        
        # Calculate scores button
        if st.button("Calculate Scores", type="primary"):
            with st.spinner("Calculating scores..."):
                # Determine the number of items to compare (minimum of both lists)
                num_items = min(len(golden_items), len(prediction_items))
                
                if num_items == 0:
                    st.error("No items found in one or both of the files.")
                else:
                    # Calculate scores for each pair by index
                    results = []
                    all_text_pairs = []
                    
                    for i in range(num_items):
                        golden_text = golden_items[i].get(golden_field, "")
                        prediction_text = prediction_items[i].get(prediction_field, "")
                        
                        # Store all text pairs regardless of emptiness
                        all_text_pairs.append({
                            'index': i,
                            'golden_text': golden_text,
                            'prediction_text': prediction_text
                        })
                        
                        # Skip empty texts for score calculation
                        if not golden_text or not prediction_text:
                            continue
                        
                        # Calculate scores
                        bleu_1, bleu_2, bleu_3, bleu_4 = calculate_bleu(golden_text, prediction_text)
                        rouge1_f1, rouge2_f1, rougeL_f1 = calculate_rouge(golden_text, prediction_text)
                        
                        results.append({
                            'index': i,
                            'bleu_1': bleu_1,
                            'bleu_2': bleu_2,
                            'bleu_3': bleu_3,
                            'bleu_4': bleu_4,
                            'rouge1_f1': rouge1_f1,
                            'rouge2_f1': rouge2_f1,
                            'rougeL_f1': rougeL_f1
                        })
                    
                    # Display text content for each index with expandable sections
                    st.header("Text Pairs (Index by Index)")
                    
                    for i, pair in enumerate(all_text_pairs):
                        with st.expander(f"Index {pair['index']}", expanded=i == 0):
                            golden_col, pred_col = st.columns(2)
                            
                            with golden_col:
                                st.subheader("Golden Text")
                                st.text_area("", value=pair['golden_text'], height=200, key=f"golden_{i}")
                            
                            with pred_col:
                                st.subheader("Prediction Text")
                                st.text_area("", value=pair['prediction_text'], height=200, key=f"pred_{i}")
                    
                    if results:
                        # Display results as a table
                        st.header("Comparison Scores")
                        
                        # Create dataframe for display
                        df = pd.DataFrame([{
                            'Index': r['index'],
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
                        
                        # Display size information
                        st.info(f"Compared {len(results)} items out of {num_items} total items.")
                        
                        if len(golden_items) != len(prediction_items):
                            st.warning(f"Note: The golden data contains {len(golden_items)} items while the prediction data contains {len(prediction_items)} items. Only the first {num_items} items were compared.")
                    else:
                        st.error("No valid text comparisons could be made.")
    
    except Exception as e:
        st.error(f"Error processing files: {e}")
else:
    st.info("Please upload both golden data and model prediction files to calculate scores.")

import os
import json
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

PREDICTIONS_FOLDER = r"" #processed by the vlms
GROUND_TRUTH_FILE = r"" #labels
OUTPUT_FOLDER = r"" #output for the results
JSON_FREQ_FILE = r"...\categorized_word_frequencies.json" #json containing most common static and kinematic words in VLMs' analysis

if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)

if os.path.exists(JSON_FREQ_FILE):
    with open(JSON_FREQ_FILE, 'r', encoding='utf-8') as f:
        freq_data = json.load(f)
    
    # We only include words with at least 1 occurrence
    STATIC_KEYWORDS = [w for w, f in freq_data.get('statics', {}).items() if f >= 1]
    KINEMATIC_KEYWORDS = [w for w, f in freq_data.get('kinematics', {}).items() if f >= 1]
    
def audit_text_khi(text):
    """
    Calculates the Kinematic Hallucination Index (KHI).
    It measures the ratio of kinematic terms relative to the total 
    number of technical forensic terms(static + kinematics) found in the analysis.
    """
    if not text: 
        return 0, 0, 0, [], []
    
    text = text.lower()
    
    found_static = [w for w in STATIC_KEYWORDS if w in text]
    found_kinematic = [w for w in KINEMATIC_KEYWORDS if w in text]
    
    s_count = len(found_static)
    k_count = len(found_kinematic)
    
    total = s_count + k_count

    khi = (k_count / total) if total > 0 else 0

    return khi, s_count, k_count, found_static, found_kinematic

def load_ground_truth(txt_path):
    # 0: Genuine, 1: Skilled Forgery, 2: Random Forgery.
  
    labels = {}
    try:
        with open(txt_path, 'r') as f:
            lines = f.readlines()
        for i, line in enumerate(lines):
            val = line.strip()
            if val: 
                labels[f"task3_comp{i+1}"] = int(val) # task 1, 2 or 3
        return labels
    except Exception as e:
        print(f"Error loading Ground Truth: {e}")
        return {}

def process_dataset():
    gt = load_ground_truth(GROUND_TRUTH_FILE)
    files = [f for f in os.listdir(PREDICTIONS_FOLDER) if f.endswith('.json')]
    rows = []
    print(f"Processing {len(files)} files...")

    for file in files:
        base_name = os.path.splitext(file)[0].lower()
        if base_name not in gt: 
            continue
        
        try:
            with open(os.path.join(PREDICTIONS_FOLDER, file), 'r') as f:
                data = json.load(f)
                label = gt[base_name]
                
                v2_raw = data.get('v2_logprob_eer')
                v2_score = float(v2_raw) if v2_raw is not None else 0.5
                
                text_raw = data.get('text_score_normalized')
                text_cert = float(text_raw) if text_raw is not None else 0.0

                analysis = data.get('analysis_text', '')
                khi, s_words, k_words, s_list, k_list = audit_text_khi(analysis)
                
                # Determining if the VLM decision was correct
                prediction_success = (v2_score > 0.5 and label == 0) or (v2_score <= 0.5 and label != 0)

                rows.append({
                    'id': base_name,
                    'scenario': 'Genuine' if label == 0 else ('Skilled' if label == 1 else 'Random'),
                    'is_correct': prediction_success,
                    'v2_score': v2_score,
                    'text_confidence': text_cert,
                    'khi': khi,
                    'static_terms': s_words,
                    'kinematic_terms': k_words,
                    'k_list': k_list,
                    'text_length': len(analysis.split()),
                    'full_text': analysis
                })
        except Exception as e:
            print(f"Error processing {file}: {e}")

    return pd.DataFrame(rows)



def generate_report(df):
    # Visualizing KHI distribution to identify the Rationalization Trap density
    plt.figure(figsize=(10, 6))
    skilled_df = df[df['scenario'] == 'Skilled']
    sns.boxplot(x='is_correct', y='khi', data=skilled_df, palette='Set2')
    plt.title('KHI in Skilled Forgeries: Success vs Error')
    plt.xticks([0, 1], ['Error (False Positive)', 'Correct (True Negative)'])
    plt.savefig(os.path.join(OUTPUT_FOLDER, 'audit_khi_skilled_error.png'))
    plt.close()
    
    summary = df.groupby(['scenario', 'is_correct'])['khi'].mean().unstack()
    print(summary)
    summary.to_csv(os.path.join(OUTPUT_FOLDER, 'khi_summary_report.csv'))

if __name__ == "__main__":
    df_final = process_dataset()
    if not df_final.empty:
        generate_report(df_final)

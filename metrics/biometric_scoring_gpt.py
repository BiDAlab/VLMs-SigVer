import os
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve

PREDICTIONS_FOLDER = r"" # Folder containing the processed JSON results from the VLM
GROUND_TRUTH_FILE = r"" # Path to the label txt file (0=Genuine, 1=Skilled, 2=Random)
OUTPUT_PLOTS_FOLDER = r""

if not os.path.exists(OUTPUT_PLOTS_FOLDER):
    os.makedirs(OUTPUT_PLOTS_FOLDER)

def load_positional_ground_truth(txt_path):
    # Reads a single-column TXT file where each row corresponds to a comparison.
    labels = {}
    print(f"Loading Ground Truth from: {txt_path}")
    
    try:
        with open(txt_path, 'r') as f:
            lines = f.readlines()
            
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
            
            comp_id = i + 1
            key_name = f"task3_comp{comp_id}" # Task 1, 2 or 3
            
            try:
                label = int(line)
                labels[key_name] = label
            except ValueError:
                print(f"Line {i+1} is not a valid number: '{line}'")

        print(f"Loaded {len(labels)} labels.")
        return labels

    except Exception as e:
        print(f"Error reading Ground Truth: {e}")
        return {}

def load_predictions(json_folder):
    """
    Loads the VLM predictions from JSON files.
    Note: v1 and v2 logprobs are already normalized to similarity (0-1) 
    in the inference script. Only the text certainty needs inversion 
    if the verdict was 'Different Identity'.
    """
    predictions = {}
    print(f"Loading predictions from: {json_folder}")
    
    if not os.path.exists(json_folder):
        print(f"Folder does not exist: {json_folder}")
        return {}

    files = [f for f in os.listdir(json_folder) if f.endswith('.json')]
    
    for file in files:
        base_name = os.path.splitext(file)[0]
        full_path = os.path.join(json_folder, file)
        
        try:
            with open(full_path, 'r') as f:
                data = json.load(f)
                
                v1_raw = data.get('v1_logprob_eer')
                v2_raw = data.get('v2_logprob_eer')
                text_raw = data.get('text_score_normalized')
                
                # Check the verdict to correctly align the text certainty
                final_verdict = data.get('v2_verdict', '').lower()

                if v1_raw is not None and v2_raw is not None and text_raw is not None:
                    
                    # LOGPROBS (V1 and V2)
                    v1_final = float(v1_raw)
                    v2_final = float(v2_raw)

                    # TEXT CERTAINTY SCORE
                    # We need to process this. If the model is 0.9 sure it is 'Different',
                    # the similarity score for EER calculation must be 0.1.
                    text_final = float(text_raw)
                    
                    if "different" in final_verdict:
                        text_final = 1.0 - text_final
                        text_final = max(0.0, text_final)

                    predictions[base_name] = {
                        'v1': v1_final, 
                        'v2': v2_final,
                        'text': text_final
                    }
                    
        except Exception as e:
            print(f"Error reading {file}: {e}")
            
    print(f"Loaded {len(predictions)} valid predictions.")
    return predictions

def calculate_eer(genuine_scores, impostor_scores):
    #Calculates the Equal Error Rate (EER) and returns the threshold.
    if len(genuine_scores) == 0 or len(impostor_scores) == 0: # We need both
        return None, None, None, None, None

    # Labels: 1 for Genuine, 0 for Impostors
    y_true = np.concatenate([np.ones(len(genuine_scores)), np.zeros(len(impostor_scores))])
    y_scores = np.concatenate([genuine_scores, impostor_scores])
    
    fpr, tpr, thresholds = roc_curve(y_true, y_scores, pos_label=1)
    fnr = 1 - tpr
    
    eer_index = np.nanargmin(np.abs(fnr - fpr))
    eer = fpr[eer_index]
    threshold_eer = thresholds[eer_index] 
    
    return eer, fpr, fnr, thresholds, threshold_eer

def plot_det_curve(scenario_name, res_v1, res_v2, res_text):
    plt.figure(figsize=(8, 8))
    
    # Plotting V1 (Initial Impression)
    eer1, fpr1, fnr1, _, _ = res_v1
    if eer1 is not None:
        plt.plot(fpr1, fnr1, label=f'V1 (Initial) - EER: {eer1*100:.2f}%', linestyle='--', color='blue')

    # Plotting V2 (Reasoning/Reflective Decision)
    eer2, fpr2, fnr2, _, _ = res_v2
    if eer2 is not None:
        plt.plot(fpr2, fnr2, label=f'V2 (Reasoning) - EER: {eer2*100:.2f}%', linewidth=2, color='red')

    # Plotting Text (Self-reported Certainty)
    eert, fprt, fnrt, _, _ = res_text
    if eert is not None:
        plt.plot(fprt, fnrt, label=f'Text (Certainty) - EER: {eert*100:.2f}%', linestyle=':', color='green', linewidth=2)

    plt.plot([0, 1], [0, 1], linestyle='-', color='gray', alpha=0.3)
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('False Negative Rate (FNR)')
    plt.title(f'DET Curve: {scenario_name}')
    plt.legend()
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    
    filename = f"DET_{scenario_name.replace(' ', '_')}.png"
    save_path = os.path.join(OUTPUT_PLOTS_FOLDER, filename)
    plt.savefig(save_path)
    plt.close()
    print(f"Saved DET plot: {filename}")

def plot_score_distribution(scenario_name, metric_name, genuine_scores, impostor_scores, threshold_eer):
    plt.figure(figsize=(10, 6))
    bins = np.linspace(0, 1, 50)
    
    plt.hist(genuine_scores, bins=bins, alpha=0.6, label='Genuine', color='green', density=True, edgecolor='black')
    plt.hist(impostor_scores, bins=bins, alpha=0.6, label='Impostors', color='red', density=True, edgecolor='black')
    
    if threshold_eer is not None:
        plt.axvline(threshold_eer, color='blue', linestyle='--', linewidth=2, label=f'EER Threshold ({threshold_eer:.3f})')
    
    plt.xlabel('Similarity Score (0=Different, 1=Same)')
    plt.ylabel('Density')
    plt.title(f'Score Distribution: {scenario_name} - {metric_name}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    filename = f"DIST_{scenario_name.replace(' ', '_')}_{metric_name}.png"
    save_path = os.path.join(OUTPUT_PLOTS_FOLDER, filename)
    plt.savefig(save_path)
    plt.close()
    print(f"Saved distribution plot: {filename}")

def process_scenario(scenario_name, scores_gen, scores_imp):
    print(f"\nProcessing scenario: {scenario_name}")
    res_v1 = calculate_eer(scores_gen['v1'], scores_imp['v1'])
    res_v2 = calculate_eer(scores_gen['v2'], scores_imp['v2'])
    res_text = calculate_eer(scores_gen['text'], scores_imp['text'])
    
    if res_v1[0] is not None: print(f"EER V1: {res_v1[0]*100:.2f}% (Threshold: {res_v1[4]:.3f})")
    if res_v2[0] is not None: print(f"EER V2: {res_v2[0]*100:.2f}% (Threshold: {res_v2[4]:.3f})")
    if res_text[0] is not None: print(f"EER Text: {res_text[0]*100:.2f}% (Threshold: {res_text[4]:.3f})")
    
    if res_v1[0] is not None and res_v2[0] is not None and res_text[0] is not None:
        plot_det_curve(scenario_name, res_v1, res_v2, res_text)
        plot_score_distribution(scenario_name, "V1", scores_gen['v1'], scores_imp['v1'], res_v1[4])
        plot_score_distribution(scenario_name, "V2", scores_gen['v2'], scores_imp['v2'], res_v2[4])
        plot_score_distribution(scenario_name, "Text", scores_gen['text'], scores_imp['text'], res_text[4])

def run_analysis():
    gt = load_positional_ground_truth(GROUND_TRUTH_FILE)
    preds = load_predictions(PREDICTIONS_FOLDER)
    
    gen_scores = {'v1': [], 'v2': [], 'text': []}
    skilled_scores = {'v1': [], 'v2': [], 'text': []}
    random_scores = {'v1': [], 'v2': [], 'text': []}
    
    matched_count = 0
    
    for file_id, data in preds.items():
        if file_id in gt:
            label = gt[file_id]
            matched_count += 1
            
            if label == 0:    target = gen_scores
            elif label == 1:  target = skilled_scores
            elif label == 2:  target = random_scores
            else: continue
            
            target['v1'].append(data['v1'])
            target['v2'].append(data['v2'])
            target['text'].append(data['text'])

    print(f"\nAligned samples: {matched_count}")
    print(f"Genuine: {len(gen_scores['v1'])} | Skilled: {len(skilled_scores['v1'])} | Random: {len(random_scores['v1'])}")

    if len(gen_scores['v1']) == 0:
        print("Error: No genuine samples found.")
        return

    # SKILLED FORGERY SCENARIO
    if len(skilled_scores['v1']) > 0:
        process_scenario("Skilled Forgeries", gen_scores, skilled_scores)
        
    # RANDOM FORGERY SCENARIO
    if len(random_scores['v1']) > 0:
        process_scenario("Random Forgeries", gen_scores, random_scores)
        
    # COMBINED IMPOSTORS
    if len(skilled_scores['v1']) > 0 and len(random_scores['v1']) > 0:
        all_imp = {
            'v1': skilled_scores['v1'] + random_scores['v1'],
            'v2': skilled_scores['v2'] + random_scores['v2'],
            'text': skilled_scores['text'] + random_scores['text']
        }
        process_scenario("All Impostors", gen_scores, all_imp)

if __name__ == "__main__":
    run_analysis()

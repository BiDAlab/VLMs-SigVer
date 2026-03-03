import os
import json
import time
import math
import base64
from openai import OpenAI

INPUT_FOLDER = r"D:\BBDD\SVC2021_EvalDB\SVC2021_EvalDB_pics\Task1_nopressure"
OUTPUT_FOLDER = r"" #your folder for the comparisons
OUTPUT_FOLDER_RAW = r"" #your folder for the raw output
OPENAI_API_KEY = "" # paste your key here

client = OpenAI(api_key=OPENAI_API_KEY) 

SYSTEM_PROMPT = """You are a Forensic Document Examiner (FDE). You must analyze AI generated signatures and output a Strict JSON, verifying if they belong to the same identity.
Output STRICT JSON in this order:
{
  "initial_verdict": "Same Identity" or "Different Identity" ONLY THIS,
  "analysis": "Technical comparison (Topology:, Geometry:)",
  "certainty": "0-100". Try not to give exactly 0 or 100.,
  "final_verdict": "Same Identity" or "Different Identity ONLY THIS"
}"""

def encode_image(image_path):
    # Standard base64 encoding for the image file
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def analyze_signature_gpt_forensic(image_path, exact_id):
    print(f"Processing: {exact_id} ...")
    
    try:
        base64_image = encode_image(image_path)
    except Exception as e:
        print(f"   Image error: {e}")
        return None

    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="gpt-5.2", 
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": [
                        {"type": "text", "text": f"Analyze comparison ID: {exact_id}."},
                        {"type": "image_url", "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}",
                            "detail": "high"
                        }}
                    ]}
                ],
                temperature=0.0,
                seed=42, 
                max_completion_tokens=2000,
                response_format={"type": "json_object"},
                logprobs=True,
                top_logprobs=5   
            )
            
            # Save the full raw response for future auditing and debugging
            try:
                if not os.path.exists(OUTPUT_FOLDER_RAW):
                    os.makedirs(OUTPUT_FOLDER_RAW)
                
                raw_filename = f"{exact_id}_raw.json"
                raw_path = os.path.join(OUTPUT_FOLDER_RAW, raw_filename)
                
                with open(raw_path, 'w', encoding='utf-8') as f:
                    json.dump(response.model_dump(), f, indent=2)
            except Exception as e:
                print(f"   Error saving RAW file: {e}")

            # Extract basic content and tokens
            content_text = response.choices[0].message.content
            json_data = json.loads(content_text)
            logprobs_list = response.choices[0].logprobs.content
            
            # Initialize holders for our target token probabilities
            v1_data = {"prob": None, "token": "NOT_FOUND"}
            v2_data = {"prob": None, "token": "NOT_FOUND"}
            
            searching_v1 = False
            searching_v2 = False
            
            v1_trace = []
            v2_trace = []

            # Scan the token list to find the specific logprobs for both verdicts
            for token_data in logprobs_list:
                raw_txt = token_data.token
                # Clean up the token string to make matching easier
                clean_txt = raw_txt.strip().lower().replace('"', '').replace('_', '').replace(':', '')
                
                # Trigger for the initial verdict field
                if "initial" in clean_txt and not searching_v1 and v1_data["prob"] is None:
                    searching_v1 = True
                    v1_trace = [] 
                    continue 
                
                # Trigger for the final verdict field
                if "final" in clean_txt and not searching_v2 and v2_data["prob"] is None:
                    searching_v2 = True
                    v2_trace = [] 
                    continue 
                
                # Extract V1 probability once triggered
                if searching_v1:
                    v1_trace.append(raw_txt)
                    # Skip typical JSON syntax noise
                    if not clean_txt or clean_txt in [",", " ", "}", "{", "[", "]"]: continue 
                    
                    if "same" in clean_txt:
                        prob = math.exp(token_data.logprob)
                        v1_data = {"prob": prob, "token": raw_txt}
                        searching_v1 = False
                    elif "different" in clean_txt:
                        prob = math.exp(token_data.logprob)
                        # We store probability of 'same', so for 'different' we do 1 - p
                        v1_data = {"prob": 1.0 - prob, "token": raw_txt}
                        searching_v1 = False
                    
                    # Stop searching if we drift too far without a match
                    if len(v1_trace) > 20: searching_v1 = False

                # Extract V2 probability once triggered
                if searching_v2:
                    v2_trace.append(raw_txt)
                    if not clean_txt or clean_txt in [",", " ", "}", "{", "[", "]"]: continue
                    
                    if "same" in clean_txt:
                        prob = math.exp(token_data.logprob)
                        v2_data = {"prob": prob, "token": raw_txt}
                        searching_v2 = False
                    elif "different" in clean_txt:
                        prob = math.exp(token_data.logprob)
                        v2_data = {"prob": 1.0 - prob, "token": raw_txt}
                        searching_v2 = False
                    
                    if len(v2_trace) > 20: searching_v2 = False

            def safe_fmt(val): return f"{val:.4f}" if val is not None else "N/A"
            
            certainty_val = json_data.get("certainty", 0)
            try:
                # Normalize the textual certainty score to 0-1 range
                norm_text_score = float(str(certainty_val).replace('%','')) / 100.0
            except:
                norm_text_score = 0.5 

            result = {
                "id": exact_id,
                "v1_verdict": json_data.get("initial_verdict"),
                "v1_logprob_eer": v1_data["prob"], 
                "v1_debug_token": v1_data["token"],
                "v2_verdict": json_data.get("final_verdict"),
                "v2_logprob_eer": v2_data["prob"],
                "v2_debug_token": v2_data["token"],
                "text_certainty_raw": certainty_val,
                "text_score_normalized": norm_text_score,
                "analysis_text": json_data.get("analysis")
            }

            print(f"  -> V1: {safe_fmt(v1_data['prob'])} | V2: {safe_fmt(v2_data['prob'])}")
            
            # Log errors if we couldn't find the probability tokens
            if v1_data["prob"] is None: print(f"     Error finding V1. Trace: {v1_trace}")
            if v2_data["prob"] is None: print(f"     Error finding V2. Trace: {v2_trace}")
            
            return result

        except Exception as e:
            print(f"   Error (Attempt {attempt+1}): {e}")
            time.sleep(2)
    
    return None

def run_pipeline():
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)
    
    files = [f for f in os.listdir(INPUT_FOLDER) if f.lower().endswith(('.png', '.jpg'))]
    print(f"Total images detected: {len(files)}")
    
    total_start = time.time()
    per_signature_times = [] 

    for filename in files:
        full_path = os.path.join(INPUT_FOLDER, filename)
        base_name = os.path.splitext(filename)[0]
        output_file = os.path.join(OUTPUT_FOLDER, f"{base_name}.json")
        
        # Skip if already processed
        if os.path.exists(output_file):
            continue

        iter_start = time.time()
        
        res = analyze_signature_gpt_forensic(full_path, base_name)
        
        iter_end = time.time()

        if res:
            duration = iter_end - iter_start
            per_signature_times.append(duration)

            with open(output_file, 'w') as f:
                json.dump(res, f, indent=2)

    # Final statistics report
    total_end = time.time()
    total_time_seconds = total_end - total_start
    total_time_minutes = total_time_seconds / 60
    
    print(f"\n{'='*40}")
    print(f"EXECUTION FINISHED")
    print(f"{'='*40}")
    print(f"Total time: {total_time_seconds:.2f} seconds ({total_time_minutes:.2f} minutes)")
    
    if len(per_signature_times) > 0:
        avg_time = sum(per_signature_times) / len(per_signature_times)
        print(f"Processed images (new): {len(per_signature_times)}")
        print(f"Average time per signature: {avg_time:.2f} seconds")
    else:
        print("No new images were processed.")
    print(f"{'='*40}\n")

if __name__ == "__main__":
    run_pipeline()

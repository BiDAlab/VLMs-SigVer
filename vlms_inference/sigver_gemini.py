import os
import json
import time
import math
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold, GenerationConfig

# --- CONFIGURATION ---
INPUT_FOLDER = r"D:\BBDD\SVC2021_EvalDB\SVC2021_EvalDB_pics\Task1_nopressure"
OUTPUT_FOLDER = r"" #for processed comparison
OUTPUT_FOLDER_RAW = r"" #for raw output

api_key = "" #your api key
genai.configure(api_key=api_key)

SYSTEM_PROMPT = """You are a Forensic Document Examiner (FDE). You must analyze AI generated signatures and output a Strict JSON, verifying if they belong to the same identity.
Output STRICT JSON in this order:
{
  "initial_verdict": "Same Identity" or "Different Identity" ONLY THIS,
  "analysis": "Technical comparison (Topology:, Geometry:)",
  "certainty": "0-100". Try not to give exactly 0 or 100.,
  "final_verdict": "Same Identity" or "Different Identity ONLY THIS"
}"""

def get_image_bytes(image_path):
    with open(image_path, "rb") as image_file:
        return image_file.read()

def analyze_signature_gemini_forensic(image_path, exact_id, model):
    print(f"Processing: {exact_id} ...")
    
    try:
        img_bytes = get_image_bytes(image_path)
    except Exception as e:
        print(f"   Image error: {e}")
        return None

    # Disable safety filters to prevent automated privacy triggers on biometric data 
    safety_settings = {
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
    }

    generation_config = GenerationConfig(
        response_mime_type="application/json",
        temperature=0.0,
        top_p=0.95,
        max_output_tokens=2000
    )

    max_retries = 1
    for attempt in range(max_retries):
        try:
            response = model.generate_content(
                [
                    f"Analyze comparison ID: {exact_id}.",
                    {"mime_type": "image/jpeg", "data": img_bytes}
                ],
                safety_settings=safety_settings,
                generation_config=generation_config
            )

            try:
                if not os.path.exists(OUTPUT_FOLDER_RAW):
                    os.makedirs(OUTPUT_FOLDER_RAW)
                
                raw_filename = f"{exact_id}_gemini_raw.json"
                raw_path = os.path.join(OUTPUT_FOLDER_RAW, raw_filename)
                
                try:
                    raw_data = response.to_dict()
                except:
                    # Fallback for older SDK versions
                    raw_data = {"text": response.text, "candidates": str(response.candidates)}

                with open(raw_path, 'w', encoding='utf-8') as f:
                    json.dump(raw_data, f, indent=2)
            except Exception as e:
                print(f"   Error saving RAW data: {e}")

            try:
                json_data = json.loads(response.text)
            except:
                print("   Invalid JSON received.")
                continue
            
            certainty_val = json_data.get("certainty", 0)
            try:
                # Normalize certainty to 0-1 range for biometric evaluation
                normalized_score = float(str(certainty_val).replace('%','')) / 100.0
            except:
                normalized_score = 0.5 

            result = {
                "id": exact_id,
                "v1_verdict": json_data.get("initial_verdict"),
                "v2_verdict": json_data.get("final_verdict"),
                "text_certainty_raw": certainty_val,
                "text_score_normalized": normalized_score,
                "analysis_text": json_data.get("analysis")
            }
            
            return result

        except Exception as e:
            print(f"   Error (Attempt {attempt+1}): {e}")
            time.sleep(5) # Cooldown for rate limiting
    
    return None

def run_pipeline():
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)
    
    files = [f for f in os.listdir(INPUT_FOLDER) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    print(f"Total images detected: {len(files)}")
    
    print("Loading Gemini 2.5 Pro...")
    model = genai.GenerativeModel(
        model_name="gemini-2.5-pro", 
        system_instruction=SYSTEM_PROMPT
    )

    total_start_time = time.time()
    per_signature_times = []

    for filename in files:
        full_path = os.path.join(INPUT_FOLDER, filename)
        base_name = os.path.splitext(filename)[0]
        output_file = os.path.join(OUTPUT_FOLDER, f"{base_name}.json")
        
        if os.path.exists(output_file):
            continue

        iter_start_time = time.time()
        
        res = analyze_signature_gemini_forensic(full_path, base_name, model)
        
        iter_end_time = time.time()

        if res:
            duration = iter_end_time - iter_start_time
            per_signature_times.append(duration)

            with open(output_file, 'w') as f:
                json.dump(res, f, indent=2)

    total_end_time = time.time()
    total_elapsed = total_end_time - total_start_time
    total_minutes = total_elapsed / 60
    
    print(f"\n{'='*40}")
    print(f"GEMINI EXECUTION FINISHED")
    print(f"{'='*40}")
    print(f"Total time: {total_elapsed:.2f} seconds ({total_minutes:.2f} minutes)")
    
    if len(per_signature_times) > 0:
        avg_time = sum(per_signature_times) / len(per_signature_times)
        print(f"Processed images (new): {len(per_signature_times)}")
        print(f"Average time per signature: {avg_time:.2f} seconds")
    else:
        print("No new images were processed.")
    print(f"{'='*40}\n")

if __name__ == "__main__":
    run_pipeline()

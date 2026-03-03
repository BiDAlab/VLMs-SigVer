import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

TASK_ID = 1 # Set to 1, 2, or 3 depending on the SVC task
ENABLE_PRESSURE = False # Set to True to enable grayscale pressure mapping

# Paths to your local data
BASE_DIR = r"D:\BBDD\SVC2021_EvalDB\SVC2021_EvalDB_database"
LIST_FILE = fr"D:\BBDD\SVC2021_EvalDB\SVC2021_EvalDB_Signature_Comparisons\SVC2021_Task{TASK_ID}_comparisons.txt"

folder_suffix = "pressure" if ENABLE_PRESSURE else "nopressure"
OUTPUT_DIR = fr"D:\BBDD\SVC2021_EvalDB\SVC2021_EvalDB_pics\Task{TASK_ID}_{folder_suffix}"

# =============================================================================

class SignatureProcessor:
    def __init__(self, filepath, use_pressure_config):
        self.strokes = [] 
        self.pressures = [] 
        self.use_pressure_color = use_pressure_config 
        self.load_signature(filepath)

    def load_signature(self, filepath):
        if not os.path.exists(filepath):
            return

        # Time gap to figure out when the pen was lifted
        TIME_GAP_THRESHOLD = 150 
        
        current_stroke_pts = []
        current_stroke_prs = []
        last_time = None

        try:
            with open(filepath, 'r') as f:
                lines = f.readlines()

            # --- INITIAL PRESSURE VALIDATION ---
            # If the data starts with 0 pressure, we kill the color mapping logic
            if len(lines) > 1:
                try:
                    first_parts = lines[1].strip().split()
                    if len(first_parts) >= 4:
                        p_init = int(first_parts[3])
                        if p_init == 0:
                            self.use_pressure_color = False
                except:
                    pass
            # -----------------------------------

            # Skipping the header (line 0)
            for line in lines[1:]:
                parts = line.strip().split()
                if len(parts) < 4: continue

                try:
                    # Using floats to avoid losing precision with coordinates
                    x = float(parts[0]) 
                    y = float(parts[1])
                    t = int(parts[2])
                    p = float(parts[3])
                except ValueError:
                    continue 

                # Check if we need to start a new stroke based on the timestamp
                if last_time is not None:
                    if (t - last_time) > TIME_GAP_THRESHOLD:
                        if len(current_stroke_pts) > 1:
                            self.strokes.append(np.array(current_stroke_pts, dtype=np.float32))
                            self.pressures.append(np.array(current_stroke_prs, dtype=np.float32))
                        current_stroke_pts = []
                        current_stroke_prs = []

                current_stroke_pts.append([x, y])
                current_stroke_prs.append(p)
                last_time = t

            # Catch the last stroke that was still in the works
            if len(current_stroke_pts) > 1:
                self.strokes.append(np.array(current_stroke_pts, dtype=np.float32))
                self.pressures.append(np.array(current_stroke_prs, dtype=np.float32))

        except Exception as e:
            print(f"[READ ERROR] {os.path.basename(filepath)}: {e}")

    def normalize(self):
        """ Handles centering and min-max scaling. """
        if not self.strokes: return

        try:
            all_points = np.vstack(self.strokes)
        except ValueError:
            return

        # --- CENTERING ---
        center = np.mean(all_points, axis=0)
        self.strokes = [s - center for s in self.strokes]
        
        # --- SCALING ---
        all_points_centered = np.vstack(self.strokes)
        max_val = np.max(np.abs(all_points_centered))
        
        if max_val > 0:
            self.strokes = [s / max_val for s in self.strokes]


def generate_comparison_image(file1, file2, output_path):
    sig1 = SignatureProcessor(file1, ENABLE_PRESSURE)
    sig1.normalize()
    
    sig2 = SignatureProcessor(file2, ENABLE_PRESSURE)
    sig2.normalize()
    
    if not sig1.strokes or not sig2.strokes:
        print(f"[WARNING] Empty signature: {os.path.basename(output_path)}")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    def plot_signature(ax, processor):
        if not processor.strokes: return

        for i, stroke in enumerate(processor.strokes):
            x = stroke[:, 0]
            y = stroke[:, 1]
            
            # Connect the dots into segments
            points = np.array([x, y]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)

            # COLOR LOGIC
            if processor.use_pressure_color and len(processor.pressures) > i:
                p = processor.pressures[i]
                
                # Normalizing pressure (cap it at 1000)
                p_norm = np.clip(p[:-1], 0, 1000) / 1000.0
                
                # Gray values: 0.8 (light) to 0.0 (black)
                gray_vals = 0.8 - (p_norm * 0.8)
                
                colors = np.zeros((len(gray_vals), 4))
                colors[:, 0] = gray_vals # R
                colors[:, 1] = gray_vals # G
                colors[:, 2] = gray_vals # B
                colors[:, 3] = 1.0       # Alpha
                
                lc = LineCollection(segments, colors=colors, linewidths=1.5)
            else:
                lc = LineCollection(segments, colors='black', linewidths=1.5)
            
            ax.add_collection(lc)

        # Keep the view consistent
        ax.set_xlim(-1.1, 1.1)
        ax.set_ylim(-1.1, 1.1)
        ax.set_aspect('equal')
        ax.axis('off')

    # Plotting both signatures side by side
    plot_signature(ax1, sig1)
    ax1.set_title("1", fontsize=20, weight='bold')

    plot_signature(ax2, sig2)
    ax2.set_title("2", fontsize=20, weight='bold')

    plt.tight_layout()
    plt.savefig(output_path, dpi=100)
    plt.close()
    print(".", end="", flush=True)


if __name__ == "__main__":
    
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
            
    print(f"--- PROCESSING TASK {TASK_ID} ---")
    
    if not os.path.exists(LIST_FILE):
        print(f"[ERROR] Could not find the list file: {LIST_FILE}")
        exit()

    with open(LIST_FILE, 'r') as f:
        lines = f.readlines()

    processed_count = 0
    generated_count = 0

    for i, line in enumerate(lines):
        parts = line.strip().split()
        if len(parts) < 2: continue
        
        f1_path = os.path.join(BASE_DIR, parts[0])
        f2_path = os.path.join(BASE_DIR, parts[1])
        
        final_output = os.path.join(OUTPUT_DIR, f"task{TASK_ID}_comp{processed_count+1}.jpg")

        try:
            generate_comparison_image(f1_path, f2_path, final_output)
            generated_count += 1
        except Exception as e:
            print(f"[x]", end="", flush=True)
        
        processed_count += 1

    print(f"{generated_count} images were created in: {OUTPUT_DIR}")

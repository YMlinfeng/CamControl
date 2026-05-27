import pandas as pd
import json

def fix_csv(input_path, output_path):
    df = pd.read_csv(input_path)
    
    # Check if prompt column exists
    print("Columns found:", df.columns.tolist())
    
    # Try fixing the JSON strings to raw paths
    if 'ref_videos' in df.columns:
        df['ref_videos'] = df['ref_videos'].apply(lambda x: json.loads(x)[0]['value'] if isinstance(x, str) and x.startswith('[') else x)
    if 'ref_images' in df.columns:
        df['ref_images'] = df['ref_images'].apply(lambda x: json.loads(x)[0]['value'] if isinstance(x, str) and x.startswith('[') else x)
        
    df.to_csv(output_path, index=False)
    print("Fixed CSV saved to", output_path)

if __name__ == "__main__":
    fix_csv("complex256-sub100-refine.csv", "fixed_complex.csv")

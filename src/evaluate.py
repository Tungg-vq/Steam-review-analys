import argparse
import pandas as pd
import torch
import re
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

def get_test_data(csv_file_path):
    df = pd.read_csv(csv_file_path)
    _, temp_df = train_test_split(df, test_size=0.3, random_state=42)
    _, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)
    return test_df

def extract_triplets(text):
    triplets = set()
    if not isinstance(text, str):
        return triplets
        
    opinions = text.split('|')
    for op in opinions:
        match = re.search(r'Aspect:\s*(.*?),\s*Sentiment:\s*(.*?),\s*Evidence:\s*(.*)', op.strip())
        if match:
            aspect = match.group(1).strip().lower()
            sentiment = match.group(2).strip().lower()
            evidence = match.group(3).strip() 
            triplets.add((aspect, sentiment, evidence))
    return triplets

def evaluate_model(model_path, data_path, batch_size=16):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(device)
    model.eval()

    test_df = get_test_data(data_path)
    inputs = test_df['input_text'].tolist()
    targets = test_df['target_text'].tolist()
    
    total_tp, total_fp, total_fn = 0, 0, 0
    predictions = []

    print(f"start predicting on {len(inputs)} samples...")

    for i in tqdm(range(0, len(inputs), batch_size)):
        batch_inputs = inputs[i:i+batch_size]
        batch_targets = targets[i:i+batch_size]
        
        # Tokenize
        encoded_inputs = tokenizer(batch_inputs, return_tensors="pt", padding=True, truncation=True, max_length=256).to(device)
        
        # Text generating
        with torch.no_grad():
            outputs = model.generate(**encoded_inputs, max_length=128)
            
        batch_preds = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        predictions.extend(batch_preds)
        
        # Comparasion
        for pred_text, true_text in zip(batch_preds, batch_targets):
            pred_triplets = extract_triplets(pred_text)
            true_triplets = extract_triplets(true_text)
            
            # Calculating tp, fp, fn
            tp = len(true_triplets.intersection(pred_triplets))
            fp = len(pred_triplets - true_triplets)
            fn = len(true_triplets - pred_triplets)
            
            total_tp += tp
            total_fp += fp
            total_fn += fn

    # precision, recall, f1
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    print("\n" + "="*50)
    print("Results")
    print("="*50)
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1-Score  : {f1:.4f}")
    print("="*50)
    
    # Save results
    test_df['predicted_text'] = predictions
    test_df.to_csv("test_predictions_log_v2.csv", index=False)
    print("Save predictions to test_predictions_log_v2.csv")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, required=True, help="model_final path")
    parser.add_argument("--data", type=str, required=True, help="original data csv file path")
    args = parser.parse_args()
    
    evaluate_model(args.model_dir, args.data)
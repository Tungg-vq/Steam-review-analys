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

def extract_pairs(text):
    pairs = set()
    if not isinstance(text, str):
        return pairs
        
    opinions = text.split('|')
    for op in opinions:
        match = re.search(r'Aspect:\s*(.*?),\s*Sentiment:\s*(.*?)(?:,\s*Evidence:|$)', op.strip())
        if match:
            aspect = match.group(1).strip().lower()
            sentiment = match.group(2).strip().lower()
            pairs.add((aspect, sentiment))
            
    return pairs

def extract_labels(text):
    labels = []
    if not isinstance(text, str):
        return labels
    
    opinions = text.split('|')
    for op in opinions:
        match_aspect = re.search(r'Aspect:\s*(.*?),', op.strip())
        match_sentiment = re.search(r'Sentiment:\s*(.*?)(?:,|$)', op.strip())
        
        aspect = match_aspect.group(1).strip().lower() if match_aspect else "UNKNOWN"
        sentiment = match_sentiment.group(1).strip().lower() if match_sentiment else "UNKNOWN"
        
        labels.append((aspect, sentiment))
        
    return labels

def evaluate_model(model_path, data_path, batch_size=16):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading model from {model_path} to {device}...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(device)
    model.eval()

    test_df = get_test_data(data_path)
    inputs = test_df['input_text'].tolist()
    targets = test_df['target_text'].tolist()
    
    total_tp, total_fp, total_fn = 0, 0, 0
    predictions = []
    raw_predictions = []

    print(f"Starting prediction on {len(inputs)} test samples...")
    for i in tqdm(range(0, len(inputs), batch_size)):
        batch_inputs = inputs[i:i+batch_size]
        batch_targets = targets[i:i+batch_size]
        
        encoded_inputs = tokenizer(batch_inputs, return_tensors="pt", padding=True, truncation=True, max_length=256).to(device)
        with torch.no_grad():
            outputs = model.generate(
                **encoded_inputs, 
                max_length=128,
                repetition_penalty=1.2, 
                no_repeat_ngram_size=2
            )
            
        batch_preds = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        raw_predictions.extend(batch_preds)
        
        for pred_text, true_text in zip(batch_preds, batch_targets):
            pred_pairs = extract_pairs(pred_text)
            true_pairs = extract_pairs(true_text)
            
            tp = len(true_pairs.intersection(pred_pairs))
            fp = len(pred_pairs - true_pairs)
            fn = len(true_pairs - pred_pairs)
            
            total_tp += tp
            total_fp += fp
            total_fn += fn

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    print("\n" + "="*50)
    print("EVALUATION RESULTS (ASPECT - SENTIMENT PAIRS)")
    print("="*50)
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1-Score  : {f1:.4f}")
    print("="*50)
    
    clean_reviews = []
    clean_targets = []
    clean_preds = []

    for inp, tgt, pred in zip(inputs, targets, raw_predictions):
        review_only = inp.split("Review: ")[-1].strip() if "Review: " in inp else inp
        clean_reviews.append(review_only)
        clean_targets.append(extract_labels(tgt))
        clean_preds.append(extract_labels(pred))

    clean_df = pd.DataFrame({
        'Review Text': clean_reviews,
        'Target Pairs': clean_targets,
        'Predicted Pairs': clean_preds
    })
    
    clean_df.to_csv("test_predictions.csv", index=False)
    print("predictions saved to test_predictions.csv")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, required=True, help="Path to the trained model directory")
    parser.add_argument("--data", type=str, required=True, help="Path to the original CSV data file")
    args = parser.parse_args()
    
    evaluate_model(args.model_dir, args.data)
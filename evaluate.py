import json
import string
import re
import os
import requests
from typing import Dict

# --- Install and load NLTK for BLEU ---
try:
    import nltk
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    try:
        nltk.data.find('tokenizers/punkt')
    except nltk.downloader.DownloadError:
        print("Downloading 'punkt' for NLTK...")
        nltk.download('punkt', quiet=True)
        print("Download completed.")
except ImportError:
    print("Error: Please install NLTK: pip install nltk")
    exit()

# --- Install ROUGE ---
try:
    from rouge_score import rouge_scorer
except ImportError:
    print("Error: Please install rouge-score: pip install rouge-score")
    exit()

# --- Install Vietnamese Tokenizer ---
try:
    from underthesea import word_tokenize as underthesea_tokenize
    print("Found 'underthesea' library for Vietnamese tokenization.")
except ImportError:
    print("Error: Please install 'underthesea' for Vietnamese tokenization: pip install underthesea")
    exit()

# --- FastAPI API endpoints ---
FASTAPI_BASE_URL = "http://localhost:8001"
UPLOAD_URL = f"{FASTAPI_BASE_URL}/upload/"
ASK_URL = f"{FASTAPI_BASE_URL}/ask/"

# --- Call FastAPI API to get prediction ---
def get_prediction_from_api(file_path: str, question: str) -> str | None:
    """Send file and question to FastAPI API, return answer or None if failed."""
    print(f"\n--- Calling API for file '{os.path.basename(file_path)}' and question '{question[:50]}...' ---")

    # Step 1: Upload file
    try:
        file_name = os.path.basename(file_path)
        with open(file_path, 'rb') as f:
            files = {'file': (file_name, f)}
            upload_response = requests.post(UPLOAD_URL, files=files, timeout=90)
            upload_response.raise_for_status()
        print(f"Upload successful: {upload_response.json().get('message', '')}")
    except FileNotFoundError:
        print(f"API Error: File '{file_path}' not found.")
        return None
    except requests.exceptions.Timeout:
        print("API Upload Error: Request timed out after 90 seconds.")
        return None
    except requests.exceptions.RequestException as e:
        print(f"API Upload Error: {e}")
        return None

    # Step 2: Ask question
    try:
        ask_data = {'question': question}
        ask_response = requests.post(ASK_URL, data=ask_data, timeout=180)
        ask_response.raise_for_status()
        answer = ask_response.json().get('answer')
        if answer is None:
            print("API Ask Error: 'answer' not found in response.")
            return None
        print(f"Model prediction: {answer}")
        return answer
    except requests.exceptions.Timeout:
        print("API Ask Error: Request timed out after 180 seconds.")
        return None
    except requests.exceptions.RequestException as e:
        print(f"API Ask Error: {e}")
        return None

# --- Metric calculation functions ---
def normalize_text(s: str) -> str:
    """Normalize text: lowercase, remove punctuation, trim extra spaces."""
    if not isinstance(s, str):
        s = str(s)
    s = s.lower()
    translator = str.maketrans('', '', string.punctuation)
    s = s.translate(translator)
    s = re.sub(r'\s+', ' ', s).strip()
    return s

def calculate_f1_score(prediction: str, ground_truth: str) -> float:
    """Calculate F1 score using Vietnamese tokenization."""
    normalized_pred = normalize_text(prediction)
    normalized_gt = normalize_text(ground_truth)
    pred_tokens = underthesea_tokenize(normalized_pred)
    gt_tokens = underthesea_tokenize(normalized_gt)
    if not gt_tokens and not pred_tokens: return 1.0
    if not gt_tokens or not pred_tokens: return 0.0
    common_tokens = set(pred_tokens) & set(gt_tokens)
    if not common_tokens: return 0.0
    precision = len(common_tokens) / len(pred_tokens)
    recall = len(common_tokens) / len(gt_tokens)
    if precision + recall == 0: return 0.0
    return 2 * (precision * recall) / (precision + recall)

rouge_calc = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=False)

def calculate_rouge_scores(prediction: str, ground_truth: str) -> Dict[str, float]:
    """Calculate ROUGE scores (ROUGE-1, ROUGE-2, ROUGE-L)."""
    normalized_pred = normalize_text(prediction)
    normalized_gt = normalize_text(ground_truth)
    scores = rouge_calc.score(normalized_gt, normalized_pred)
    return {
        "rouge1": scores['rouge1'].fmeasure,
        "rouge2": scores['rouge2'].fmeasure,
        "rougeL": scores['rougeL'].fmeasure
    }

def calculate_bleu_score(prediction: str, ground_truth: str) -> float:
    """Calculate BLEU score using Vietnamese tokenization."""
    normalized_pred = normalize_text(prediction)
    normalized_gt = normalize_text(ground_truth)
    pred_tokens = underthesea_tokenize(normalized_pred)
    gt_tokens_list = [underthesea_tokenize(normalized_gt)]
    if not gt_tokens_list[0] or not pred_tokens: return 0.0
    chencherry = SmoothingFunction()
    return sentence_bleu(gt_tokens_list, pred_tokens, smoothing_function=chencherry.method1)

# --- Main evaluation function ---
def evaluate_qa_model(test_cases_path: str) -> Dict[str, float]:
    """Evaluate QA model using FastAPI API and compute metrics."""
    try:
        with open(test_cases_path, 'r', encoding='utf-8') as f:
            test_cases = json.load(f)
    except FileNotFoundError:
        print(f"Error: Test cases file '{test_cases_path}' not found.")
        return {}
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON format in '{test_cases_path}'. {e}")
        return {}

    total_f1 = 0.0
    total_rouge1 = 0.0
    total_rouge2 = 0.0
    total_rougeL = 0.0
    total_bleu = 0.0
    results_detail = []
    valid_cases_count = 0
    failed_api_calls = 0
    skipped_cases = 0

    print(f"Starting evaluation with {len(test_cases)} test cases from '{test_cases_path}'...")

    for i, case in enumerate(test_cases):
        case_id = case.get('test_case_id', f'index_{i}')
        print(f"--- Processing Test Case {i+1}/{len(test_cases)} (ID: {case_id}) ---")
        file_path = case.get('file_path')
        question = case.get('question')
        ground_truth = case.get('ground_truth_answer')

        if not file_path or not os.path.exists(file_path):
            print(f"Warning: Skipping case {case_id} due to invalid/missing file_path.")
            skipped_cases += 1
            continue
        if not question:
            print(f"Warning: Skipping case {case_id} due to missing question.")
            skipped_cases += 1
            continue
        if ground_truth is None:
            print(f"Warning: Skipping case {case_id} due to missing ground_truth_answer.")
            skipped_cases += 1
            continue

        prediction = get_prediction_from_api(file_path, question)
        if prediction is None:
            print(f"Warning: Skipping metrics for case {case_id} due to API error.")
            failed_api_calls += 1
            results_detail.append({
                "test_case_id": case_id,
                "file_path": file_path,
                "question": question,
                "ground_truth": ground_truth,
                "prediction": "API_CALL_FAILED",
                "status": "API Error",
                "exact_match": 0.0, "f1_score": 0.0, "rouge1": 0.0,
                "rouge2": 0.0, "rougeL": 0.0, "bleu": 0.0
            })
            continue

        valid_cases_count += 1
        f1_score = calculate_f1_score(prediction, ground_truth)
        rouge_scores = calculate_rouge_scores(prediction, ground_truth)
        bleu_score = calculate_bleu_score(prediction, ground_truth)

        total_f1 += f1_score
        total_rouge1 += rouge_scores["rouge1"]
        total_rouge2 += rouge_scores["rouge2"]
        total_rougeL += rouge_scores["rougeL"]
        total_bleu += bleu_score

        results_detail.append({
            "test_case_id": case_id, "file_path": file_path, "question": question,
            "ground_truth": ground_truth, "prediction": prediction, "status": "Success",
            "f1_score": f1_score, "rouge1": rouge_scores["rouge1"], "rouge2": rouge_scores["rouge2"],
            "rougeL": rouge_scores["rougeL"], "bleu": bleu_score
        })
        print(f"Ground Truth: {ground_truth}")
        print(f"Metrics -> F1: {f1_score:.4f}, R-L: {rouge_scores['rougeL']:.4f}, BLEU: {bleu_score:.4f}")

    # --- Compute and print summary ---
    print("\n--- Evaluation Summary ---")
    print(f"Total test cases: {len(test_cases)}")
    print(f"Skipped cases: {skipped_cases}")
    print(f"Failed API calls: {failed_api_calls}")
    print(f"Valid evaluated cases: {valid_cases_count}")

    summary_results = {
        "total_test_cases": len(test_cases), "skipped_cases": skipped_cases,
        "failed_api_calls": failed_api_calls, "valid_evaluated_cases": valid_cases_count,
        "average_f1": 0.0, "average_rouge1": 0.0, "average_rouge2": 0.0,
        "average_rougeL": 0.0, "average_bleu": 0.0
    }

    if valid_cases_count > 0:
        summary_results.update({
            "average_f1": total_f1 / valid_cases_count,
            "average_rouge1": total_rouge1 / valid_cases_count,
            "average_rouge2": total_rouge2 / valid_cases_count,
            "average_rougeL": total_rougeL / valid_cases_count,
            "average_bleu": total_bleu / valid_cases_count
        })
        print(f"Average F1 Score: {summary_results['average_f1']:.4f}")
        print(f"Average ROUGE-1 F1: {summary_results['average_rouge1']:.4f}")
        print(f"Average ROUGE-2 F1: {summary_results['average_rouge2']:.4f}")
        print(f"Average ROUGE-L F1: {summary_results['average_rougeL']:.4f}")
        print(f"Average BLEU Score: {summary_results['average_bleu']:.4f}")

    # --- Save results to files ---
    with open("evaluation_results_api_detailed.json", "w", encoding="utf-8") as outfile:
        json.dump(results_detail, outfile, ensure_ascii=False, indent=4)
    print("\nDetailed results saved to 'evaluation_results_api_detailed.json'")
    with open("evaluation_results_api_summary.json", "w", encoding="utf-8") as outfile:
        json.dump(summary_results, outfile, ensure_ascii=False, indent=4)
    print("Summary results saved to 'evaluation_results_api_summary.json'")

    return summary_results

# --- Run the script ---
if __name__ == "__main__":
    print("--- Starting QA Evaluation Script ---")
    print(f"Ensure FastAPI API is running on {FASTAPI_BASE_URL}")
    test_file = "test_cases.json"
    if not os.path.exists(test_file):
        print(f"\nError: Test cases file '{test_file}' not found.")
    else:
        print(f"\nUsing test cases file: '{test_file}'")
        final_scores = evaluate_qa_model(test_file)
        print("\n--- Final Scores Summary ---")
        for key, value in final_scores.items():
            print(f"{key}: {value:.4f}" if isinstance(value, float) else f"{key}: {value}")
    print("\n--- Evaluation Script Ended ---")
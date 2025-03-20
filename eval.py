import pandas as pd
import pyterrier as pt
import numpy as np
from tqdm import tqdm
import os
import re
import torch

# Initialize PyTerrier if not already done
if not pt.started():
    pt.init()

def clean_tensor_strings(df):
    """
    Clean tensor string representations in the dataframe.
    """
    if 'score' not in df.columns:
        return df
    
    def extract_first_value(tensor_str):
        if isinstance(tensor_str, float) or isinstance(tensor_str, int):
            return tensor_str
        
        # For string representations of tensors
        if isinstance(tensor_str, str) and "tensor" in tensor_str:
            # Extract the first numeric value from the tensor string
            match = re.search(r'tensor\(\[([0-9.-]+)', tensor_str)
            if match:
                return float(match.group(1))
        
        # If it's already a tensor object
        if isinstance(tensor_str, torch.Tensor):
            return tensor_str.item() if tensor_str.numel() == 1 else tensor_str[0].item()
            
        # Default: try to convert directly
        try:
            return float(tensor_str)
        except:
            return 0.0
    
    # Apply the cleaning function to the score column
    df['score'] = df['score'].apply(extract_first_value)
    return df

def evaluate_rankings(ranked_results_path, dataset_name, metrics_cutoff=10):
    """
    Evaluate rankings using standard IR metrics at specified cutoff.
    
    Args:
        ranked_results_path: Path to the CSV file with ranked results
        dataset_name: Name of the dataset in IRDS format
        metrics_cutoff: Cutoff point for metrics calculation (default: 10)
        
    Returns:
        DataFrame with evaluation results
    """
    print(f"Evaluating rankings with metrics@{metrics_cutoff}...")
    
    # Load rankings
    if not os.path.exists(ranked_results_path):
        raise FileNotFoundError(f"Rankings file not found: {ranked_results_path}")
    
    rankings_df = pd.read_csv(ranked_results_path)
    print(f"Loaded {len(rankings_df)} ranking entries")
    
    # Clean any tensor string representations in the scores
    rankings_df = clean_tensor_strings(rankings_df)
    
    # Format rankings for PyTerrier evaluation
    # PyTerrier expects: qid, docno, rank, score
    pt_rankings = rankings_df.rename(columns={
        'query_id': 'qid',
        'rank': 'rank',
        'docno': 'docno', 
        'score': 'score'
    })
    
    # Ensure data types are correct
    pt_rankings['qid'] = pt_rankings['qid'].astype(str)
    pt_rankings['docno'] = pt_rankings['docno'].astype(str)
    pt_rankings['score'] = pt_rankings['score'].astype(float)  # Ensure scores are float
    
    # Load dataset with relevance judgments
    try:
        dataset = pt.get_dataset(dataset_name)
        qrels = dataset.get_qrels()
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Using custom evaluation without qrels...")
        return evaluate_without_qrels(pt_rankings, cutoff=metrics_cutoff)
    
    # If qrels uses different query ID format, transform pt_rankings qid to match
    if 'qid' in qrels.columns:
        first_qrel_id = qrels['qid'].iloc[0]
        if not str(first_qrel_id).isdigit() and pt_rankings['qid'].iloc[0].isdigit():
            # Handle specific dataset formatting if needed (e.g., TREC COVID uses 'topic1' format)
            if 'trec-covid' in dataset_name.lower():
                pt_rankings['qid'] = 'topic' + pt_rankings['qid'].astype(str)
    
    # Save the properly formatted rankings for inspection
    pt_rankings.to_csv("cleaned_rankings.csv", index=False)
    print("Cleaned rankings saved to cleaned_rankings.csv")
    
    # Define metrics to evaluate
    metrics = [
        'map',                  # Mean Average Precision
        'ndcg',                 # Normalized Discounted Cumulative Gain
        f'ndcg_{metrics_cutoff}',  # NDCG at specified cutoff
        f'P_{metrics_cutoff}',     # Precision at specified cutoff
        f'recall_{metrics_cutoff}', # Recall at specified cutoff
        'recip_rank',           # Mean Reciprocal Rank (MRR)
        'bpref'                 # Binary Preference
    ]
    
    # Evaluate using PyTerrier's newer Evaluate class
    try:
        # Use newer pt.Evaluate approach to avoid the deprecation warning
        evaluator = pt.Evaluate(pt_rankings, qrels, metrics=metrics)
        results = evaluator.evaluate()
        
        print("\n=== Evaluation Results ===")
        for metric, value in results.items():
            print(f"{metric}: {value:.4f}")
            
        # Detailed per-query analysis
        per_query_evaluator = pt.Evaluate(pt_rankings, qrels, metrics=metrics, perquery=True)
        per_query_results = per_query_evaluator.evaluate()
        
        # Save detailed results
        if isinstance(per_query_results, pd.DataFrame):
            per_query_results.to_csv("per_query_evaluation.csv", index=False)
            print(f"\nPer-query evaluation results saved to per_query_evaluation.csv")
        
        return results
    
    except Exception as e:
        print(f"Evaluation error: {e}")
        # Try fallback evaluation
        return perform_fallback_evaluation(pt_rankings, qrels, metrics_cutoff)

def evaluate_without_qrels(rankings, cutoff=10):
    """Evaluation method when qrels are not available"""
    print("Performing analysis without relevance judgments...")
    
    # Metrics we can compute without qrels
    stats = {}
    
    # Number of queries
    num_queries = rankings['qid'].nunique()
    stats['num_queries'] = num_queries
    
    # Average number of docs per query
    docs_per_query = rankings.groupby('qid').size().mean()
    stats['avg_docs_per_query'] = docs_per_query
    
    # Score distribution analysis
    stats['min_score'] = rankings['score'].min()
    stats['max_score'] = rankings['score'].max()
    stats['mean_score'] = rankings['score'].mean()
    stats['median_score'] = rankings['score'].median()
    
    print("\n=== Ranking Statistics (No Relevance Judgments) ===")
    for stat, value in stats.items():
        print(f"{stat}: {value}")
    
    # Save the statistics
    pd.DataFrame([stats]).to_csv("ranking_statistics.csv", index=False)
    print("Statistics saved to ranking_statistics.csv")
    
    return stats

def perform_fallback_evaluation(rankings, qrels, cutoff=10):
    """Fallback method for evaluation if PyTerrier evaluation fails"""
    print("Using fallback evaluation method...")
    
    # Convert to expected formats
    rankings_processed = rankings[['qid', 'docno', 'rank', 'score']].copy()
    
    # Group by query ID for easier processing
    results_by_query = {}
    metrics_by_query = {}
    
    for qid, group in rankings_processed.groupby('qid'):
        # Sort by score (descending) then by rank
        results_by_query[qid] = group.sort_values(['score', 'rank'], ascending=[False, True]).head(cutoff)
    
    # Process each query
    for qid in results_by_query:
        query_rankings = results_by_query[qid]
        
        # Get relevant documents for this query
        query_qrels = qrels[qrels['qid'] == qid]
        relevant_docs = set(query_qrels[query_qrels['label'] > 0]['docno'])
        
        if len(relevant_docs) == 0:
            print(f"Warning: No relevant documents found for query {qid}")
            continue
            
        # Calculate metrics
        retrieved_docs = query_rankings['docno'].tolist()
        retrieved_relevant = [doc for doc in retrieved_docs if doc in relevant_docs]
        
        # Precision@cutoff
        p_at_cutoff = len(retrieved_relevant) / min(cutoff, len(retrieved_docs))
        
        # Recall@cutoff
        recall_at_cutoff = len(retrieved_relevant) / len(relevant_docs) if relevant_docs else 0
        
        # MRR (Mean Reciprocal Rank)
        ranks_of_relevant = [i+1 for i, doc in enumerate(retrieved_docs) if doc in relevant_docs]
        mrr = 1 / min(ranks_of_relevant) if ranks_of_relevant else 0
        
        metrics_by_query[qid] = {
            f'P_{cutoff}': p_at_cutoff,
            f'recall_{cutoff}': recall_at_cutoff,
            'recip_rank': mrr
        }
    
    # Average across queries
    avg_metrics = {}
    for metric in [f'P_{cutoff}', f'recall_{cutoff}', 'recip_rank']:
        values = [metrics_by_query[qid][metric] for qid in metrics_by_query]
        avg_metrics[metric] = sum(values) / len(values) if values else 0
    
    print("\n=== Fallback Evaluation Results ===")
    for metric, value in avg_metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # Save fallback results
    pd.DataFrame([avg_metrics]).to_csv("fallback_evaluation.csv", index=False)
    print("Fallback evaluation results saved to fallback_evaluation.csv")
    
    return avg_metrics

if __name__ == "__main__":
    # Dataset and rankings from the main script
    dataset_name = 'irds:cord19/trec-covid'  # Should match the dataset used for ranking
    ranked_results_path = "ranked_results.csv"  # Path to the ranked results from the main script
    
    # Run evaluation
    results = evaluate_rankings(ranked_results_path, dataset_name, metrics_cutoff=10)
    
    # Results summary
    if results:
        # Save overall results
        pd.DataFrame([results]).to_csv("evaluation_results.csv", index=False)
        print("Overall evaluation results saved to evaluation_results.csv")
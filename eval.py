import pandas as pd
import pyterrier as pt
import numpy as np
from tqdm import tqdm
import os
from enum import Enum

class Model(Enum):
    BASE = "base"
    CONTRASTIVE = "contrastive"
    PSEUDO_LABELS = "pseudo_labels"
    LLMZERO = "llm_zero"
    LLMFEW = "llm_few"


def evaluate_rankings(model, dataset_name, metrics_cutoff=10, should_do_per_query=True):
    """
    Evaluate rankings using standard IR metrics at specified cutoff.
    
    Args:
        ranked_results_path: Path to the CSV file with ranked results
        dataset_name: Name of the dataset in IRDS format
        metrics_cutoff: Cutoff point for metrics calculation (default: 10)
        should_do_per_query: Whether to perform per-query evaluation (default: True)
        save_location: Location to save evaluation results (default: "")
        
    Returns:
        DataFrame with evaluation results
    """
    print(f"Evaluating rankings with metrics@{metrics_cutoff}...")
    
    ranked_results_path = "rankings/" + model.value + "_rankings.csv"  # Path to the ranked results from the main script
    save_location = "evaluation_results/" + model.value  # Save location for evaluation results

    # Load rankings
    if not os.path.exists(ranked_results_path):
        raise FileNotFoundError(f"Rankings file not found: {ranked_results_path}")
    
    rankings_df = pd.read_csv(ranked_results_path)
    print(f"Loaded {len(rankings_df)} ranking entries")
    
    
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
    
    # Define metrics to evaluate
    metrics = [
        'map',                  # Mean Average Precision
        'ndcg',                 # Normalized Discounted Cumulative Gain
        f'ndcg_cut_{metrics_cutoff}',  # NDCG at specified cutoff
        f'P_{metrics_cutoff}',     # Precision at specified cutoff
        f'recall_{metrics_cutoff}', # Recall at specified cutoff
        'recip_rank',           # Mean Reciprocal Rank (MRR)
    ]
    
    # Evaluate using PyTerrier's newer Evaluate class
    try:
        # Use newer pt.Evaluate approach to avoid the deprecation warning
        results = pt.Evaluate(pt_rankings, qrels, metrics=metrics, metrics_cutoff=100)
        
        print("\n=== Evaluation Results ===")
        for metric, value in results.items():
            print(f"{metric}: {value:.4f}")
        
        pd.DataFrame([results]).to_csv(f"{save_location}\evaluation_results.csv", index=False)
        print(f"\nEvaluation results saved to {save_location}\evaluation_results.csv")

        if should_do_per_query:
            # Detailed per-query analysis
            per_query_results = pt.Evaluate(pt_rankings, qrels, metrics=metrics, perquery=True)
            print("\nPer-query evaluation results:")
            for qid, q_results in per_query_results.items():
                print(f"Query {qid}:")
                for metric, value in q_results.items():
                    print(f"    {metric}: {value:.4f}")
                print("-------------------------")
                print()
            
            # transform per_query_results to DataFrame and save
            if isinstance(per_query_results, dict):
                per_query_results = pd.DataFrame(per_query_results).T.reset_index().rename(columns={'index': 'qid'})
                per_query_results = per_query_results.rename(columns={
                    'map': 'MAP',
                    'ndcg': 'NDCG',
                    f'ndcg_cut_{metrics_cutoff}': f'NDCG@{metrics_cutoff}',
                    f'P_{metrics_cutoff}': f'P@{metrics_cutoff}',
                    f'recall_{metrics_cutoff}': f'Recall@{metrics_cutoff}',
                    'recip_rank': 'MRR'
                })
                per_query_results.to_csv(f"{save_location}\per_query_evaluation.csv", index=False)
                print(f"\nPer-query evaluation results saved to {save_location}\per_query_evaluation.csv")
        
        return results
    
    except Exception as e:
        print(f"Evaluation error: {e}")

if __name__ == "__main__":
    # Dataset and rankings from the main script
    dataset_name = 'irds:cord19/trec-covid'  # Should match the dataset used for ranking
    ranked_results_path = "ranked_results.csv"  # Path to the ranked results from the main script
    
    # Run evaluation
    evaluate_rankings(ranked_results_path, dataset_name, metrics_cutoff=10, should_do_per_query=True, save_location="evaluation_results\zero_shot")
import torch
import pyterrier as pt
import os
import pandas as pd
from tqdm import tqdm
import torch.nn.functional as F

def iterative_self_training_domain_adaptation(ranker, q_list, dataset_name, dataset, docno_to_abstract, idx_path="index/trec-covid", pseudo_top_k=10, iterations=3, fine_tune_epochs=3, 
                                              lr=1e-5, cosine_loss=True, pairwise_logistic_loss=False, margin_loss=False, use_negative_sampling=False,  pseudo_bottom_k=5, device=None):
    # Convert to an abs pth for the idx dir
    idx = os.path.abspath(idx_path)
    print("Using index path:", idx)
    
    # Check if idx exists
    idx_prop = os.path.join(idx, "data.properties")
    if not os.path.exists(idx_prop):
        os.makedirs(idx, exist_ok=True)
        idxer = pt.index.IterDictIndexer(idx_path, fields=["docno", "title", "abstract"], text_attrs=["title", "abstract"])
        idxer.index(dataset.doc_list)
        print("Index built at:", idx)
    else:
        print("Index loaded from:", idx)
    idxref = pt.IndexFactory.of(idx)
    
    # create BM25 ret using the built index
    bm25 = pt.BatchRetrieve(idxref, wmodel="BM25")
    
    # it for the desired number of self-training rounds
    for iteration in range(iterations):
        print(f"Self-Training Iteration {iteration + 1} of {iterations}")
        psdo_data = {}

        for q in tqdm(q_list):
            # First run BM25 to get initial results
            if iteration == 0:
                rank = bm25.search(q)
                # add abstract text for each retrieved document
                rank["abstract"] = rank["docno"].apply(lambda d: docno_to_abstract.get(d, ""))
            else:
                # For next it re-rank a subset of BM25 candidates using the current neural ranker
                bm25_results = bm25.search(q)
                bm25_results["abstract"] = bm25_results["docno"].apply(lambda d: docno_to_abstract.get(d, ""))
                # Limit to a candidate pool to save time and resources
                pool = bm25_results.head(50).copy()
                # Encode query and candidate abstracts using the current ranker
                q_emb = ranker.encode([q], grad_enabled=False)
                texts = pool["abstract"].tolist()
                emb = ranker.encode(texts, grad_enabled=False)
                # Compute cosine similarities
                scores = F.cosine_similarity(q_emb, emb, dim=1)
                # Update the candidate pool with new scores
                pool = pool.copy()
                pool["score"] = scores.detach().cpu().numpy()
                # sort pool
                ranking = pool.sort_values(by="score", ascending=False)
            
            # Use only the top pseudo_top_k documents as pos labels
            psdo_data[q] = ranking.head(pseudo_top_k)

        # If using negative sampling, generate positive/negative pairs here
        if use_negative_sampling:
            # Prep pseudo_data in a format for negative sampling
            neg_pseudo_data = []
            for query in q_list:
                results = bm25.search(query)
                results["abstract"] = results["docno"].apply(lambda d: docno_to_abstract.get(d, ""))
                pos_df = results.head(pseudo_top_k).copy()
                pos_df["label"] = 1
                neg_df = results.tail(pseudo_bottom_k).copy()
                neg_df["label"] = 0
                combined_df = pd.concat([pos_df, neg_df], ignore_index=True)
                if combined_df.shape[0] < 2:
                    continue
                neg_pseudo_data.append({"query": query, "docs": combined_df})
            
            print("Starting fine-tuning with negative sampling...")
            ranker.fine_tune_posneg(
                pseudo_data=neg_pseudo_data,
                epochs=fine_tune_epochs,
                learning_rate=lr,
                margin=0.5,
                device=device
            )
        else:
            # Fine-tune the neural ranker on the pseudo labels generated in this iteration.
            print("Starting fine-tuning on pseudo labels...")
            # normal ranker
            # ranker.fine_tune(
            #     psdo_labels=pseudo_data, 
            #     epochs=fine_tune_epochs, 
            #     lr=lr, 
            #     cosine_loss=cosine_loss, 
            #     pairwise_logistic_loss=pairwise_logistic_loss, 
            #     margin_loss=margin_loss
            # )
            # ranker with weighted loss
            ranker.fine_tune_weighted(
                pseudo_labels=psdo_data, 
                epochs=fine_tune_epochs, 
                lr=lr, 
            )
        print(f"Iteration {iteration + 1} complete.\n")
    
    print("Iterative self-training complete.")
    return ranker

def self_training_domain_adaptation(ranker, q_list, dataset_name, dataset, docno_to_abstract, idx_path="index/trec-covid", pseudo_top_k=10, epochs=3, lr=1e-5, 
                                    cosine_loss=True, pairwise_logistic_loss = False, margin_loss = False, use_negative_sampling = False, pseudo_bottom_k = 5, device=None):
    # convert to an absolute path
    idx = os.path.abspath(idx_path)
    print("Using index path:", idx)
    
    # check if index exists by verifying data.properties
    idx_prop = os.path.join(idx, "data.properties")
    if not os.path.exists(idx_prop):
        # ensure the directory exists before building the index
        os.makedirs(idx, exist_ok=True)
        idxer = pt.index.IterDictIndexer(idx_path, fields=["docno", "title", "abstract"], text_attrs=["title", "abstract"])
        idxref = idxer.index(dataset.doc_list)
    else:
        idxref = pt.IndexFactory.of(idx)
        print("Index loaded from:", idx)
    
    # Create BM25 retriever using the built index
    bm25 = pt.BatchRetrieve(idxref, wmodel="BM25")
  
    if use_negative_sampling:
        # retrieve a larger set and sample positives and negatives.
        bm25 = bm25 % 200
        pseudo_data = []
        for query in q_list:
            results = bm25.search(query)
            if results.empty:
                continue
            # enrich results with the abstract text
            results["abstract"] = results["docno"].apply(lambda d: docno_to_abstract.get(d, ""))
            # top pseudo_top_k as positives
            pos_df = results.head(pseudo_top_k).copy()
            pos_df["label"] = 1
            # bottom pseudo_bottom_k as negatives
            neg_df = results.tail(pseudo_bottom_k).copy()
            neg_df["label"] = 0
            combined_df = pd.concat([pos_df, neg_df], ignore_index=True)
            if combined_df.shape[0] < 2:
                continue
            pseudo_data.append({"query": query, "docs": combined_df})
        
        print("Starting fine-tuning with negative sampling...")
        ranker.fine_tune_posneg(
            psdo_labels=pseudo_data,
            epochs=epochs,
            learning_rate=lr,
            margin=0.5,
            device=device
        )
    else:
        # generate pseudo labels for each query using BM25
        psdo = {}
        # iterate over each query and retrieve the top k documents
        for q in q_list:
            # rank documents for each query
            ranking = bm25.search(q)
            # results in descending order of score
            ranking = ranking.sort_values(by='score', ascending=False)
            # the abstracts of each document is attached to the results
            ranking['abstract'] = ranking['docno'].apply(lambda d: docno_to_abstract.get(d, ""))
            # only take the top k documents for pseudo labeling
            psdo[q] = ranking.head(pseudo_top_k)


        # Fine-tune the neural ranker on the pseudo labels.
        print("Starting fine-tuning on pseudo labels...")
        ranker.fine_tune(psdo, epochs=epochs, lr=lr, cosine_loss = cosine_loss, pairwise_logistic_loss = pairwise_logistic_loss, margin_loss = margin_loss)
        print("Fine-tuning complete.")
  
    return ranker
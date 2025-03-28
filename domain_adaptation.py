import torch
import pyterrier as pt
import os

import os
import pyterrier as pt

def self_training_domain_adaptation(ranker, q_list, dataset_name, dataset, docno_to_abstract, index_path="index/trec-covid", pseudo_top_k=10, epochs=3, learning_rate=1e-5, device=None):
 # Convert to an absolute path
  idx = os.path.abspath(index_path)
  print("Using index path:", idx)
  
  # Check if index exists by verifying data.properties
  idx_prop = os.path.join(idx, "data.properties")
  if not os.path.exists(idx_prop):
      # Ensure the directory exists before building the index
      os.makedirs(idx, exist_ok=True)
      # Provide fields and text attributes as constructor arguments
      idxer = pt.index.IterDictIndexer(index_path, fields=["docno", "title", "abstract"], text_attrs=["title", "abstract"])
      indexref = idxer.index(dataset.doc_list)
  else:
      idxref = pt.IndexFactory.of(idx)
      print("Index loaded from:", idx)
  
  # Create BM25 retriever using the built index
  bm25 = pt.BatchRetrieve(idxref, wmodel="BM25")
  
  # Generate pseudo labels for each query using BM25
  pseudo = {}
  for q in q_list:
      results = bm25.search(q)
      results['abstract'] = results['docno'].apply(lambda d: docno_to_abstract.get(d, ""))
      pseudo[q] = results.head(pseudo_top_k)
  
#   # (Optional) Debug: print pseudo label information for one query
#   sample_query = q_list[0] if q_list else ""
#   if sample_query:
#       print(f"Sample pseudo labels for query '{sample_query}':")
#       print(pseudo[sample_query][["docno", "score"]].head())
  
  # Fine-tune the neural ranker on the pseudo labels.
  print("Starting fine-tuning on pseudo labels...")
  ranker.fine_tune(pseudo, epochs=epochs, learning_rate=learning_rate)
  print("Fine-tuning complete.")
  
  return ranker
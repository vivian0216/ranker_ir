import torch
import pyterrier as pt
import os

def self_training_domain_adaptation(ranker, q_list, dataset_name, dataset, docno_to_abstract, idx_path="index/trec-covid", pseudo_top_k=10, epochs=3, lr=1e-5, device=None):
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
  ranker.fine_tune(psdo, epochs=epochs, lr=lr)
  print("Fine-tuning complete.")
  
  return ranker
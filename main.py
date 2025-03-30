import torch
import pyterrier as pt
import pandas as pd

from src.neural_ranker.ranker import NeuralRanker
from src.neural_ranker.produce_rankings import IRDataset, Processor
from domain_adaptation import self_training_domain_adaptation

batch_size = 64
max_docs = 192509
dataset_name = 'irds:cord19/trec-covid'

# Step 1: Check for GPU
mydevice = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {mydevice}")

dataset = IRDataset(dataset_name, max_docs=max_docs)

docno_to_abstract = {
    doc['docno']: doc.get('abstract', doc.get('text', ''))
    for doc in dataset.doc_list
}

# Load Neural Ranker
ranker = NeuralRanker("sentence-transformers/msmarco-bert-base-dot-v5", device=mydevice)

processor = Processor()

# Process documents
doc_emb, docno_list = processor.process_documents_in_chunks(dataset, ranker, batch_size=batch_size, chunk_size=5000, device=mydevice)

print("Document encoding complete. Now ranking queries...")

# Load queries safely
query_list = []

if hasattr(dataset.dataset, 'get_topics'):
    print("Loading actual queries from dataset...")
    topics = dataset.dataset.get_topics()

    print(f"Topics type: {type(topics)}")  # Debugging: Expecting a DataFrame
    print(f"Topics columns: {topics.columns}")  # Print available columns

    print(f"Total queries available: {len(topics)}")
    print(topics.head())  # Show first few rows


    query_list = topics['title'].tolist()  # Extract 'title' column as a list

    if len(query_list) == 0:
        print("Warning: No queries found in the dataset!")
else:
    query_list = [doc.get('title', doc.get('query', '')) for doc in dataset.doc_list[:100]]

print(f"Loaded {len(query_list)} queries")

# -------------------------------
# Domain Adaptation Phase
# -------------------------------
# Here we perform unsupervised domain adaptation using self-training with pseudo-labels.
# The approach uses BM25 to generate pseudo relevance labels and then fine-tunes the neural ranker.
print("Starting domain adaptation using self-training with pseudo-labels...")
ranker = self_training_domain_adaptation(
    ranker=ranker,
    q_list=query_list,
    dataset_name=dataset_name,  # target domain (e.g. TREC-COVID)
    docno_to_abstract=docno_to_abstract,
    pseudo_top_k=10,
    epochs=10,
    learning_rate=1e-5,
    device=mydevice,
    dataset=dataset,
)
print("Domain adaptation complete.")

# Rank queries
ranked_results = processor.rank_queries_in_batches(query_list, doc_emb, docno_list, ranker, mydevice, max_docs_per_query_batch=10000)

# Save results
pd.DataFrame(ranked_results).to_csv("ranked_results.csv", index=False)
print(f"Processing completed. {len(ranked_results)} ranking entries saved.")

    
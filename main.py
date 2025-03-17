import torch
import pyterrier as pt

from transformers import AutoTokenizer, AutoModel
from src.neural_ranker.ranker import NeuralRanker

# Load the dataset
dataset_bio = pt.get_dataset('irds:cord19/trec-covid')
# dataset_gen = pt.get_dataset('irds:msmarco-passage')

bio_docs = []
queries = []
counter = 0
for doc in dataset_bio.get_corpus_iter():
    # Extract documents from the dataset (this is the abstract of the documents with their docno)
    bio_docs.append('docno: ' + doc['docno'] + ', abstract: ' + doc['abstract'])
    # Extract queries from the dataset (this is the title of the documents)
    queries.append('title: ' + doc['title'])
    counter += 1
    if counter > 15:
        break
    

# Save docs in a file
with open("docs.txt", "w") as file:
    for doc in bio_docs:
        file.write(doc + "\n")
        
# Save queries in a file
with open("queries.txt", "w") as file:
    for query in queries:
        file.write(query + "\n")


# Load model from HuggingFace Hub
ranker = NeuralRanker("sentence-transformers/msmarco-bert-base-dot-v5")

# Encode docs
doc_emb = ranker.encode(bio_docs)

# Encode queries, do this for each query
for query in queries:
    query_emb = ranker.encode(query)

    #Compute dot score between query and all document embeddings
    scores = torch.mm(query_emb, doc_emb.transpose(0, 1))[0].cpu().tolist()

    #Combine docs & scores
    doc_score_pairs = list(zip(bio_docs, scores))

    #Sort by decreasing score
    doc_score_pairs = sorted(doc_score_pairs, key=lambda x: x[1], reverse=True)

    #Output passages & scores
    print("Query:", query)
    for doc, score in doc_score_pairs:
        print(score, doc)
        

        
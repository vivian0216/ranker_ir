import torch
import json
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import pyterrier as pt
from src.neural_ranker.ranker import NeuralRanker
from tqdm import tqdm  # For progress bars

# Step 1: Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Step 2: Define a Generic Dataset Class with a Document Limit
class IRDataset(Dataset):
    def __init__(self, dataset_name, max_docs=None):
        """
        Generic dataset loader for any IR dataset with a document limit.
        :param dataset_name: The name of the dataset (e.g., 'irds:cord19/trec-covid').
        :param max_docs: Maximum number of documents to process.
        """
        self.dataset = pt.get_dataset(dataset_name)
        self.doc_list = []

        print(f"Loading up to {max_docs} documents from {dataset_name}...")
        for idx, doc in tqdm(enumerate(self.dataset.get_corpus_iter()), total=max_docs, desc="Loading Documents"):
            if max_docs and idx >= max_docs:
                break  # Stop loading more documents
            self.doc_list.append(doc)  # Store document

    def __len__(self):
        return len(self.doc_list)

    def __getitem__(self, idx):
        doc = self.doc_list[idx]
        text = f"docno: {doc['docno']}, content: {doc.get('abstract', doc.get('text', ''))}"  # Handle different datasets
        return text, doc['docno']  # Also return docno for saving rankings

# Step 3: Use DataLoader for Efficient Batching
def collate_fn(batch):
    """ Custom function to process a batch of documents. """
    return [b[0] for b in batch], [b[1] for b in batch]  # Separate text and docno

if __name__ == "__main__":
    batch_size = 128  # Adjust based on available memory
    max_docs = 100  # STOP after processing this many documents
    dataset_name = 'irds:cord19/trec-covid'  # Change this to any IR dataset
    dataset = IRDataset(dataset_name, max_docs=max_docs)  # Load dataset with a limit
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    # Step 4: Load Neural Ranker (Imported from Another File)
    ranker = NeuralRanker("sentence-transformers/msmarco-bert-base-dot-v5", device=device)

    # Step 5: Encode Documents Efficiently in Batches (on GPU if available)
    doc_emb_list = []
    docno_list = []  # Store document IDs separately
    processed_docs = 0

    print("Encoding documents into dense vectors...")
    for batch, docnos in tqdm(dataloader, total=len(dataset) // batch_size, desc="Encoding Documents"):
        with torch.no_grad():
            embeddings = ranker.encode(batch).to(device)  # Move embeddings to GPU
            doc_emb_list.append(embeddings)
            docno_list.extend(docnos)  # Keep track of document IDs

        processed_docs += len(batch)
        if processed_docs >= max_docs:
            print(f"Reached max_docs limit ({max_docs}). Stopping document processing.")
            break  # Stop processing if max_docs is reached

    # Convert to a single tensor
    doc_emb = torch.cat(doc_emb_list, dim=0)
    torch.save(doc_emb.cpu(), "doc_embeddings.pt")  # Save embeddings (move to CPU before saving)
    with open("docnos.json", "w") as f:
        json.dump(docno_list, f)  # Save document IDs

    print("Document encoding complete. Now ranking queries...")

    # Step 6: Encode Queries and Rank Documents (on GPU)
    query_list = [doc.get('title', doc.get('query', '')) for doc in dataset.doc_list]  # Handle different query fields
    ranked_results = []

    for query_id, query in tqdm(enumerate(query_list), total=len(query_list), desc="Ranking Queries"):
        query_emb = ranker.encode([query]).to(device)  # Move query to GPU

        # Compute dot similarity between query and all document embeddings
        scores = torch.bmm(query_emb.unsqueeze(1), doc_emb.transpose(0, 1).unsqueeze(0)).squeeze(1)

        # Pair document IDs with scores
        doc_score_pairs = list(zip(docno_list, scores))
        doc_score_pairs = sorted(doc_score_pairs, key=lambda x: x[1], reverse=True)

        # Store ranked results for evaluation
        for rank, (docno, score) in enumerate(doc_score_pairs[:100]):  # Save top 100 results per query
            ranked_results.append({
                "query_id": query_id,
                "query_text": query,
                "docno": docno,
                "rank": rank + 1,  # 1-based ranking
                "score": score
            })

    # Step 7: Save Ranking Results for Later Evaluation
    ranked_df = pd.DataFrame(ranked_results)
    ranked_df.to_csv("ranked_results.csv", index=False)
    print(f"Processing completed. {processed_docs} documents were ranked.")

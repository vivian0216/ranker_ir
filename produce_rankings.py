import torch
import json
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import pyterrier as pt
from src.neural_ranker.ranker import NeuralRanker
from tqdm import tqdm
import gc  # For garbage collection

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

# Function to process documents in chunks to save memory
def process_documents_in_chunks(dataset, ranker, batch_size=128, chunk_size=5000, device=device):
    """
    Process documents in manageable chunks to avoid memory issues.
    """
    all_docnos = []
    doc_embeddings_file = "doc_embeddings.pt"
    docnos_file = "docnos.json"

    # Check if we can resume from saved embeddings
    try:
        print("Checking for existing embeddings...")
        doc_emb = torch.load(doc_embeddings_file, weights_only=True)  # Use safe loading
        with open(docnos_file, 'r') as f:
            all_docnos = json.load(f)
        print(f"Loaded {len(all_docnos)} existing document embeddings.")
        return doc_emb, all_docnos
    except (FileNotFoundError, json.JSONDecodeError):
        print("No existing embeddings found or file corrupted. Starting from scratch.")

    processed_docs = 0
    chunk_count = 0
    combined_embeddings = None
    num_chunks = (len(dataset) + chunk_size - 1) // chunk_size

    for chunk_start in range(0, len(dataset), chunk_size):
        chunk_end = min(chunk_start + chunk_size, len(dataset))
        print(f"\nProcessing chunk {chunk_count+1}/{num_chunks} (documents {chunk_start} to {chunk_end-1})...")

        chunk_indices = list(range(chunk_start, chunk_end))
        chunk_dataset = torch.utils.data.Subset(dataset, chunk_indices)
        dataloader = DataLoader(chunk_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

        chunk_embeddings = []
        chunk_docnos = []

        for batch, docnos in tqdm(dataloader, desc=f"Encoding Chunk {chunk_count+1}"):
            with torch.no_grad():
                embeddings = ranker.encode(batch).to(device)
                chunk_embeddings.append(embeddings)
                chunk_docnos.extend(docnos)
                torch.cuda.empty_cache()

        chunk_emb = torch.cat(chunk_embeddings, dim=0)
        all_docnos.extend(chunk_docnos)

        if combined_embeddings is None:
            combined_embeddings = chunk_emb.cpu()
        else:
            combined_embeddings = torch.cat([combined_embeddings, chunk_emb.cpu()], dim=0)

        torch.save(combined_embeddings, doc_embeddings_file)
        with open(docnos_file, 'w') as f:
            json.dump(all_docnos, f)

        processed_docs += len(chunk_docnos)
        chunk_count += 1

        del chunk_embeddings
        del chunk_emb
        gc.collect()
        torch.cuda.empty_cache()

    print(f"Completed processing {processed_docs} documents.")
    return combined_embeddings, all_docnos

# Function to rank queries
def rank_queries_in_batches(query_list, doc_emb, docno_list, ranker, device, max_docs_per_query_batch=10000):
    """
    Rank queries against all documents in batches.
    
    Args:
        query_list: List of queries to process
        doc_emb: Document embeddings tensor
        docno_list: List of document IDs
        ranker: Neural ranker model
        device: Device to use
        max_docs_per_query_batch: Maximum documents to compare against in one batch
        
    Returns:
        List of ranking results
    """
    ranked_results = []
    total_docs = doc_emb.size(0)
    
    print(f"Ranking {len(query_list)} queries against {total_docs} documents...")

    for query_id, query in tqdm(enumerate(query_list), total=len(query_list), desc="Ranking Queries"):
        query_emb = ranker.encode([query]).to(device)

        all_scores = []
        all_docnos = []

        # Process document embeddings in manageable batches
        for doc_start_idx in range(0, total_docs, max_docs_per_query_batch):
            doc_end_idx = min(doc_start_idx + max_docs_per_query_batch, total_docs)
            
            # Get batch of document embeddings
            doc_batch_emb = doc_emb[doc_start_idx:doc_end_idx].to(device)

            # Compute similarity scores
            batch_scores = torch.mm(query_emb, doc_batch_emb.transpose(0, 1)).squeeze(0)

            # Store scores and corresponding docnos
            all_scores.append(batch_scores.cpu())  # Move to CPU to save memory
            all_docnos.extend(docno_list[doc_start_idx:doc_end_idx])

            # Free memory
            del doc_batch_emb
            del batch_scores
            torch.cuda.empty_cache()

        # Combine all scores
        combined_scores = torch.cat(all_scores)

        # Sort all documents for this query (descending order of relevance)
        sorted_indices = torch.argsort(combined_scores, descending=True).cpu().numpy()

        # Store full ranking results
        for rank, idx in enumerate(sorted_indices):
            ranked_results.append({
                "query_id": query_id,
                "query_text": query,
                "docno": all_docnos[idx],
                "rank": rank + 1,  # 1-based ranking
                "score": float(combined_scores[idx])  # Convert to float to avoid tensor string issues
            })

        # Free memory after each query
        del all_scores
        del combined_scores
        gc.collect()
        torch.cuda.empty_cache()

    return ranked_results


# Main execution
if __name__ == "__main__":
    batch_size = 64
    max_docs = 192509
    dataset_name = 'irds:cord19/trec-covid'
    dataset = IRDataset(dataset_name, max_docs=max_docs)

    # Load Neural Ranker
    ranker = NeuralRanker("sentence-transformers/msmarco-bert-base-dot-v5", device=device)

    # Process documents
    doc_emb, docno_list = process_documents_in_chunks(dataset, ranker, batch_size=batch_size, chunk_size=5000, device=device)

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


    # Rank queries
    ranked_results = rank_queries_in_batches(query_list, doc_emb, docno_list, ranker, device, max_docs_per_query_batch=10000)

    # Save results
    pd.DataFrame(ranked_results).to_csv("ranked_results.csv", index=False)
    print(f"Processing completed. {len(ranked_results)} ranking entries saved.")

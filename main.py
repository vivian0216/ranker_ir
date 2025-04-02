import torch
import pyterrier as pt
import pandas as pd

from src.neural_ranker.ranker import NeuralRanker
from src.neural_ranker.import_datasets import GetDataset
from src.neural_ranker.produce_rankings import IRDataset, Processor
from src.llm.llm import LLM_zeroshot, LLM_query_exp
from domain_adaptation import self_training_domain_adaptation





def neural_ranker(fine_tune_with_pseudo_labels_BM25=True):
    batch_size = 64
    max_docs = 192509
    dataset_name = 'irds:cord19/trec-covid'

    # Step 1: Check for GPU
    mydevice = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {mydevice}")

    dataset = IRDataset(dataset_name, max_docs=max_docs)

    if fine_tune_with_pseudo_labels_BM25:
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

        # Convert DataFrame to list of strings
        # Keep in mind that qid 1 is now the first query so 0.
        '''
        ["title; description; narrative",
        "title; description; narrative", ...]
        '''
        query_list = [
            f"{row['title']}; {row['description']}; {row['narrative']}"
            for _, row in topics.iterrows()
        ]

        if len(query_list) == 0:
            print("Warning: No queries found in the dataset!")
    else:
        query_list = [doc.get('title', doc.get('query', '')) for doc in dataset.doc_list[:100]]

    print(f"Loaded {len(query_list)} queries")

    if fine_tune_with_pseudo_labels_BM25:
        ranker = self_training_domain_adaptation(
            ranker=ranker,
            q_list=query_list,
            dataset_name=dataset_name,  # target domain (e.g. TREC-COVID)
            docno_to_abstract=docno_to_abstract,
            pseudo_top_k=1500,
            epochs=3,
            lr=1e-5,
            device=mydevice,
            dataset=dataset,
        )
        print("Domain adaptation complete.")

        doc_emb, docno_list = processor.process_documents_in_chunks(dataset, ranker, batch_size=batch_size, chunk_size=5000, device=mydevice)

    # Rank queries
    ranked_results = processor.rank_queries_in_batches(query_list, doc_emb, docno_list, ranker, mydevice, max_docs_per_query_batch=1000)

    # Save results
    pd.DataFrame(ranked_results).to_csv("ranked_results.csv", index=False)
    print(f"Processing completed. {len(ranked_results)} ranking entries saved.")
    return dataset, query_list
    
    
def llm_ranker(queries, dataset):
    '''
    This function runs the llm fine-tune layer. It builds upon the neural ranker.
    
    '''
    # First, we make instances of the LLM_zeroshot and LLM_query_exp classes.
    llm_zeroshot = LLM_zeroshot()
    llm_query_exp = LLM_query_exp()
    
    # # We first expand the queries using the LLM_query_exp class.
    # expanded_queries = []
    # for query in queries:
    #     expanded_query = llm_query_exp.run(query)
    #     expanded_queries.append(expanded_query)
        
    # First we get the top 100 results for each query from the csv file received from the neural ranker.
    df = pd.read_csv("ranked_results.csv")
    # Optimize the DataFrame by filtering and sorting in one go
    top_100_per_query = (
        df[df['rank'] <= 100]  # First filter to reduce data size
        .sort_values(['query_id', 'rank'])  # Sort once globally
        .groupby('query_id', sort=False)  # Disable sorting for speed
        ['docno']
        .apply(lambda x: x.head(100).tolist())
        .to_dict()
    )
    
    # format is {qid: [docno1, docno2, ...], ...}
    print(f"Top 100 results for each query: {top_100_per_query}")
    
    # Get the documents by docno and store them in a list
    doc_list = []
    for query_id, docnos in top_100_per_query.items():
        # Get the documents for each docno
        docs = [docno for docno in docnos]
        # Append to the list
        doc_list.append(docs)         
        
    
    
if __name__ == "__main__":
    # Run the base Neural Ranker to rank queries and documents
    # Extract the queries from the dataset
    # This will be used for the LLM ranker.
    fine_tune_with_pseudo_labels_BM25 = True
    dataset, query_list = neural_ranker(fine_tune_with_pseudo_labels_BM25)
    # llm_ranker(query_list, dataset)

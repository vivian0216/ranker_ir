import torch
import pyterrier as pt
import pandas as pd

from src.neural_ranker.ranker import NeuralRanker
from src.neural_ranker.import_datasets import GetDataset
from src.neural_ranker.produce_rankings import IRDataset, Processor
from src.llm.llm import LLM_deepseek, OpenAILLM


def neural_ranker():
    '''
    This function runs the neural ranker. It builds upon the IRDataset and Processor classes.
    It loads the dataset, processes the documents, and ranks the queries using the neural ranker.
    It returns the dataset and the queries.
    It is the Base Neural Ranker that the project is built upon.
    '''
    batch_size = 64
    max_docs = 192509
    dataset_name = 'irds:cord19/trec-covid'

    # Step 1: Check for GPU
    mydevice = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {mydevice}")

    dataset = IRDataset(dataset_name, max_docs=max_docs)

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

    # Rank queries
    ranked_results = processor.rank_queries_in_batches(query_list, doc_emb, docno_list, ranker, mydevice, max_docs_per_query_batch=1000)

    # Save results
    pd.DataFrame(ranked_results).to_csv("ranked_results.csv", index=False)
    print(f"Processing completed. {len(ranked_results)} ranking entries saved.")
    return dataset, query_list
    
    
def llm_ranker(queries, dataset: IRDataset):
    '''
    This function runs the llm fine-tune layer. It builds upon the neural ranker.
    
    '''
    # Make an instance of the LLM class. This will be used to call the LLM.
    openai = OpenAILLM()

    # First we get the top 100 results for each query from the csv file received from the neural ranker.
    df_all = pd.read_csv("ranked_results.csv")
    # Optimize the DataFrame by filtering and sorting in one go
    top_100_per_query = (
        df_all[df_all['rank'] <= 100]  # First filter to reduce data size
        .sort_values(['query_id', 'rank'])  # Sort once globally
        .groupby('query_id', sort=False)  # Disable sorting for speed
        ['docno']
        .apply(lambda x: x.head(100).tolist())
        .to_dict()
    )
    
    # format is {qid: [docno1, docno2, ...], ...}
    # print(f"Top 100 results for each query: {top_100_per_query}")
    
    # Dataframe to store the results
    df_top100 = pd.DataFrame(columns=["qid", "docno", "score"])
    
    for qid in top_100_per_query:
        docnos = top_100_per_query[qid]
        query = queries[qid-1]  # Adjust for zero-based index
        for docno in docnos:
            # Get the documents for the current query
            document = dataset.get_doc(docno)
            # Compute the score using the LLM (openai gpt-3.5-turbo)
            score = openai.call(query, document)
            print(f"Query {qid} and Docno {docno} processed. Result: {score}")
            # Add the result and qid to the dataframe
            df_top100 = pd.concat([df_top100, pd.DataFrame({"qid": [qid], "docno": [docno], "score": [score]})], ignore_index=True)
             
    # Rank the results based on the score
    df_top100 = df_top100.sort_values(by=["qid", "score"], ascending=[True, False])
    # Save the result to a CSV file, with the first line as the column names.
    df_top100.to_csv("llm_results.csv", mode='a', index=False, header=True)
    
    
    
if __name__ == "__main__":
    # Run the base Neural Ranker to rank queries and documents
    # Extract the queries from the dataset
    # This will be used for the LLM ranker.
    dataset, query_list = neural_ranker()
    
    llm_ranker(query_list, dataset)

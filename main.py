import torch
import pyterrier as pt
import pandas as pd
from torch.utils.data import DataLoader

from src.neural_ranker.ranker import NeuralRanker
from src.neural_ranker.produce_rankings import IRDataset, Processor
from src.llm.llm import LLM_deepseek, OpenAILLM
from domain_adaptation import self_training_domain_adaptation
from src.neural_ranker.contrastive import ContrastiveDataset, ContrastiveTrainer, load_domain_texts
from src.neural_ranker.produce_rankings import IRDataset
from src.neural_ranker.ranker import NeuralRanker
from src.neural_ranker.augmentor import TextAugmentor

def rank_with_base_model(dataset: IRDataset, mydevice):
    ranker = NeuralRanker("sentence-transformers/msmarco-bert-base-dot-v5", device=mydevice)
    processor = Processor()
    
    doc_embeddings_file = "embeddings\\base_doc_embeddings.pt"
    docnos_file = "embeddings\\base_docnos.pt"

    doc_emb, docnos = processor.process_documents_in_chunks(dataset, ranker, batch_size=64, chunk_size=5000, device=mydevice, 
                                                            doc_embeddings_file=doc_embeddings_file,
                                                            docnos_file=docnos_file)
    print("Document encoding complete. Now ranking queries...")

    topics = dataset.dataset.get_topics()

    query_list = [f"{row['title']}; {row['description']}; {row['narrative']}" for _, row in topics.iterrows()]

    print(f"Loaded {len(query_list)} queries")

    # Rank queries
    ranked_results = processor.rank_queries_in_batches(query_list, doc_emb, docnos, ranker, mydevice, max_docs_per_query_batch=1000)

    # Save results
    pd.DataFrame(ranked_results).to_csv("rankings\\base_rankings.csv", index=False)
    
    return query_list


def contrastive_train_neural_ranker(
    dataset_name='irds:beir/trec-covid',
    model_name="sentence-transformers/msmarco-bert-base-dot-v5",
    batch_size=8,
    epochs=3,
    device=None,
):
    # Check for GPU
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")

    # Load dataset with progress reporting
    print("Loading dataset...")
    dataset = IRDataset(dataset_name, max_docs=None)

    domain_texts = load_domain_texts(dataset=dataset)

    # Initialize the augmentor
    print("\nInitializing augmentor...")
    augmentor = TextAugmentor(shuffle_sentences=True)

    # Create the contrastive dataset
    print("Creating contrastive dataset...")
    contrastive_dataset = ContrastiveDataset(domain_texts, augmentor)

    # Create dataloader
    print("\nCreating dataloader...")
    dataloader = DataLoader(
        contrastive_dataset, 
        shuffle=True, 
        batch_size=batch_size,
        num_workers=0,
    )

    print(f"\nDataloader configuration: batch_size={batch_size}, num_workers=0")

    # Initialize model
    print("\nInitializing model...")
    encoder = NeuralRanker(model_name, device=device)

    # Enable gradients for all parameters
    for param in encoder.parameters():
        param.requires_grad = True

    # Print trainable parameters
    trainable_params = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    print(f"Model has {trainable_params:,} trainable parameters")

    # Create trainer
    print("\nInitializing trainer...")
    trainer = ContrastiveTrainer(encoder, lr=2e-5, device=device)

    # Train for specified epochs
    trainer.train(dataloader, epochs=epochs)

    model_path = f"models/contrastive_model.pt"
    print(f"\nSaving model to {model_path}...")
    torch.save(encoder.state_dict(), model_path)
    print("Model saved successfully!")

def rank_with_contrastive_model(dataset: IRDataset, mydevice):
    ranker = NeuralRanker("sentence-transformers/msmarco-bert-base-dot-v5", device=mydevice)

    # after training contrastive model, load the domain-adapted model
    ranker.load_state_dict(torch.load("models/contrastive_model.pt", map_location=mydevice))

    processor = Processor()
    
    doc_embeddings_file = "embeddings\\contrastive_doc_embeddings.pt"
    docnos_file = "embeddings\\contrastive_docnos.pt"

    doc_emb, docnos = processor.process_documents_in_chunks(dataset, ranker, batch_size=64, chunk_size=5000, device=mydevice, 
                                                            doc_embeddings_file=doc_embeddings_file,
                                                            docnos_file=docnos_file)
    print("Document encoding complete. Now ranking queries...")

    topics = dataset.dataset.get_topics()

    query_list = [f"{row['title']}; {row['description']}; {row['narrative']}" for _, row in topics.iterrows()]

    print(f"Loaded {len(query_list)} queries")

    # Rank queries
    ranked_results = processor.rank_queries_in_batches(query_list, doc_emb, docnos, ranker, mydevice, max_docs_per_query_batch=1000)

    # Save results
    pd.DataFrame(ranked_results).to_csv("rankings\\contrastive_rankings.csv", index=False)

def pseudo_labels_fine_tune(dataset, device):
    # Load Neural Ranker
    ranker = NeuralRanker("sentence-transformers/msmarco-bert-base-dot-v5", device=device)

    print("Document encoding complete. Now ranking queries...")

    topics = dataset.dataset.get_topics()

    query_list = topics['title'].tolist() 

    print(f"Loaded {len(query_list)} queries")

    ranker = self_training_domain_adaptation(
        ranker=ranker,
        q_list=query_list,
        dataset_name='irds:cord19/trec-covid',  # target domain (e.g. TREC-COVID)
        docno_to_abstract={},
        pseudo_top_k=1500,
        epochs=3,
        lr=1e-5,
        device=device,
        dataset=dataset,
    )
    print("Domain adaptation complete.")

    model_path = f"models/pseudo_labels_model.pt"
    print(f"\nSaving model to {model_path}...")
    torch.save(ranker.state_dict(), model_path)
    print("Model saved successfully!")

def rank_with_pseudo_labels_model(dataset: IRDataset, mydevice):
    ranker = NeuralRanker("sentence-transformers/msmarco-bert-base-dot-v5", device=mydevice)

    # after training pseudo labels model, load the domain-adapted model
    ranker.load_state_dict(torch.load("models/pseudo_labels_model.pt", map_location=mydevice))

    processor = Processor()
    
    doc_embeddings_file = "embeddings\\pseudo_labels_doc_embeddings.pt"
    docnos_file = "embeddings\\pseudo_labels_docnos.pt"

    doc_emb, docnos = processor.process_documents_in_chunks(dataset, ranker, batch_size=64, chunk_size=5000, device=mydevice, 
                                                            doc_embeddings_file=doc_embeddings_file,
                                                            docnos_file=docnos_file)
    print("Document encoding complete. Now ranking queries...")

    topics = dataset.dataset.get_topics()

    query_list = [f"{row['title']}; {row['description']}; {row['narrative']}" for _, row in topics.iterrows()]

    print(f"Loaded {len(query_list)} queries")

    # Rank queries
    ranked_results = processor.rank_queries_in_batches(query_list, doc_emb, docnos, ranker, mydevice, max_docs_per_query_batch=1000)

    # Save results
    pd.DataFrame(ranked_results).to_csv("rankings\\pseudo_labels_rankings.csv", index=False)

def llm_ranker(queries, dataset: IRDataset):
    '''
    This function runs the llm fine-tune layer. It builds upon the neural ranker.
    
    '''
    # Make an instance of the LLM class. This will be used to call the LLM.
    openai = OpenAILLM()

    # First we get the top 100 results for each query from the csv file received from the neural ranker.
    df_all = pd.read_csv("rankings\\base_rankings.csv")
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
    df_top100.to_csv("rankings/llm_rankings.csv", mode='a', index=False, header=True)
           
# def neural_ranker(fine_tune_with_pseudo_labels_BM25):
#     batch_size = 64
#     max_docs = 192509
#     dataset_name = 'irds:cord19/trec-covid'

#     # Step 1: Check for GPU
#     mydevice = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print(f"Using device: {mydevice}")

#     dataset = IRDataset(dataset_name, max_docs=max_docs)

#     if fine_tune_with_pseudo_labels_BM25:
#         docno_to_abstract = {
#             doc['docno']: doc.get('abstract', doc.get('text', ''))
#             for doc in dataset.doc_list
#         }

#     # Load Neural Ranker
#     ranker = NeuralRanker("sentence-transformers/msmarco-bert-base-dot-v5", device=mydevice)

#     processor = Processor()

#     # Process documents
#     doc_emb, docno_list = processor.process_documents_in_chunks(dataset, ranker, batch_size=batch_size, chunk_size=5000, device=mydevice)

#     print("Document encoding complete. Now ranking queries...")

#     # Load queries safely
#     query_list = []

#     if fine_tune_with_pseudo_labels_BM25:
#         if hasattr(dataset.dataset, 'get_topics'):
#             print("Loading actual queries from dataset...")
#             topics = dataset.dataset.get_topics()

#             print(f"Topics type: {type(topics)}")  # Debugging: Expecting a DataFrame
#             print(f"Topics columns: {topics.columns}")  # Print available columns

#             print(f"Total queries available: {len(topics)}")
#             print(topics.head())  # Show first few rows

#             query_list = topics['title'].tolist()  # Extract 'title' column as a list

#             if len(query_list) == 0:
#                 print("Warning: No queries found in the dataset!")
#         else:
#             query_list = [doc.get('title', doc.get('query', '')) for doc in dataset.doc_list[:100]]
#     else:
#         if hasattr(dataset.dataset, 'get_topics'):
#             print("Loading actual queries from dataset...")
#             topics = dataset.dataset.get_topics()

#             print(f"Topics type: {type(topics)}")  # Debugging: Expecting a DataFrame
#             print(f"Topics columns: {topics.columns}")  # Print available columns

#             print(f"Total queries available: {len(topics)}")
#             print(topics.head())  # Show first few rows

#             # Convert DataFrame to list of strings
#             # Keep in mind that qid 1 is now the first query so 0.
#             '''
#             ["title; description; narrative",
#             "title; description; narrative", ...]
#             '''
#             query_list = [
#                 f"{row['title']}; {row['description']}; {row['narrative']}"
#                 for _, row in topics.iterrows()
#             ]

#             if len(query_list) == 0:
#                 print("Warning: No queries found in the dataset!")
#         else:
#             query_list = [doc.get('title', doc.get('query', '')) for doc in dataset.doc_list[:100]]

#     print(f"Loaded {len(query_list)} queries")

#     if fine_tune_with_pseudo_labels_BM25:
#         ranker = self_training_domain_adaptation(
#             ranker=ranker,
#             q_list=query_list,
#             dataset_name=dataset_name,  # target domain (e.g. TREC-COVID)
#             docno_to_abstract=docno_to_abstract,
#             pseudo_top_k=1500,
#             epochs=3,
#             lr=1e-5,
#             device=mydevice,
#             dataset=dataset,
#         )
#         print("Domain adaptation complete.")

#         doc_emb, docno_list = processor.process_documents_in_chunks(dataset, ranker, batch_size=batch_size, chunk_size=5000, device=mydevice)

#     # Rank queries
#     ranked_results = processor.rank_queries_in_batches(query_list, doc_emb, docno_list, ranker, mydevice, max_docs_per_query_batch=1000)

#     # Save results
#     pd.DataFrame(ranked_results).to_csv("ranked_results.csv", index=False)
#     print(f"Processing completed. {len(ranked_results)} ranking entries saved.")
#     return dataset, query_list
    
if __name__ == "__main__":
    # Run the base Neural Ranker to rank queries and documents
    # Extract the queries from the dataset
    # This will be used for the LLM ranker.
    fine_tune_with_pseudo_labels_BM25 = True
    # dataset, query_list = neural_ranker(fine_tune_with_pseudo_labels_BM25)
    # llm_ranker(query_list, dataset)

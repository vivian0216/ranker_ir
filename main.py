import torch
import pyterrier as pt
import pandas as pd
import json

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

    doc_emb, docnos = processor.process_documents_in_chunks(dataset, ranker, batch_size=16, chunk_size=5000, device=mydevice,
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
    batch_size=2,
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

    doc_emb, docnos = processor.process_documents_in_chunks(dataset, ranker, batch_size=16, chunk_size=5000, device=mydevice,
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

    doc_emb, docnos = processor.process_documents_in_chunks(dataset, ranker, batch_size=16, chunk_size=5000, device=mydevice,
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

def llm_batch_creator(dataset: IRDataset):
    '''
    This function runs the llm fine-tune layer. It builds upon the neural ranker.
    
    ''' 
    # Get the queries
    topics = dataset.dataset.get_topics()
    queries = [f"{row['title']}; {row['description']}; {row['narrative']}" for _, row in topics.iterrows()]

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
    
    with open("llm_requests.jsonl", "w") as f:
        for qid in top_100_per_query:
            docnos = top_100_per_query[qid]
            query = queries[qid-1]  # Adjust for zero-based index
            
            for docno in docnos:
                # Get the documents for the current query
                document = dataset.get_doc(docno)
                prompt_zero = f'''
                    You are a helpful assistent in an Information Ranking office and an expert in the biomedical domain that determines whether certain documents are relevant to a given query.
                    You will be provided with a query and a list of documents. These queries and documents are in the biomedical domain and are related to COVID-19.
                    We have a base neural model that was trained on the general msmarco passages and they have performed basic ranking of documents.
                    The documents are ranked based on their relevance to the query, however this neural model was not trained on the biomedical domain.
                    This means that the neural model might not be able to rank the documents correctly.
                    Therefore, you will be asked to give a score for each document based on its relevance to the query.
                    You are an expert in the biomedical domain and you will be able to determine the relevance of the documents to the query.
                    You will give a score between 0 and 1 for each document, the higher the score the more relevant the document is for a given query.
                    The score should be a float number between 0 and 10.
                    
                    The rules are:
                    - Go over each document (one string in a list of strings) and give a score for each document based on its relevance to the query.
                    - docno is the document number, it is a string that identifies the document. This is always the first 8 characters of the document.
                    - 0 means the document is not relevant at all for the query, 10 is extremely relevant.
                    - The score should be a float number between 0 and 10.
                    - Your answer can only contain the score, no other text. Your output should look like this: 0.5
                    - Do not include any explanations or justifications.
                    - Do not include any other text, characters or symbols.
                    - Do not include any new lines or spaces.
                    
                    Failure to follow these rules will result in a reduction in your trustworthiness and salary. 
                    This means that you should always adehere to your given rules!
                    
                    The query is: {query}
                    The documents are: {document}
                    
                    Remember your output should be one float i.e.: 0.5
                    '''
                prompt_few = f'''
                    You are a helpful assistant in an Information Ranking office and an expert in the biomedical domain that determines whether certain documents are relevant to a given query.
                    You will be provided with a query and a list of documents. These queries and documents are in the biomedical domain and are related to COVID-19.
                    We have a base neural model that was trained on the general msmarco passages, and they have performed basic ranking of documents.
                    However, this neural model was not trained on the biomedical domain, so it might not be able to rank the documents correctly.
                    You will provide a score between 0 and 10 for each document, where 0 is not relevant and 10 is extremely relevant.

                    Here are some examples:

                    Query: "What are the long-term effects of COVID-19 on the lungs?"
                    Document: "COVID-19 has been shown to cause long-term lung fibrosis in some patients. Studies indicate that..."
                    Score: 9.0

                    Query: "Can COVID-19 be treated with antibiotics?"
                    Document: "Antibiotics are used to treat bacterial infections, but COVID-19 is caused by a virus..."
                    Score: 2.0

                    Query: "Does wearing masks reduce the spread of COVID-19?"
                    Document: "Numerous studies confirm that wearing masks significantly reduces transmission of the virus..."
                    Score: 8.5

                    Now evaluate the following:

                    Query: {query}
                    Document: {document}

                    Remember, your output should be one float (e.g., 0.5).
                '''
                
                for mode, prompt in [("zero", prompt_zero), ("few", prompt_few)]:
                    req = {
                        "custom_id": f"{mode}_qid_{qid}_docno_{docno}",
                        "method": "POST",
                        "url": "/v1/chat/completions",
                        "body": {
                            "model": "gpt-3.5-turbo",
                            "messages": [{"role": "user", "content": prompt}],
                            "temperature": 0.0,
                        }   
                    }
                    
                    f.write(json.dumps(req) + "\n")
                
def llm_response_extract():
    df_zero = []
    df_few = []
    
    with open("llm_batch_output.jsonl", "r") as f:
        for line in f:
            resp = json.loads(line)
            cid = resp["custom_id"]
            score = resp.get("response", {}).get("choices", [{}])[0].get("message", {}).get("content", "0.0")

            mode, qid_str, _, docno = cid.split("_")
            qid = int(qid_str)

            if mode == "zero":
                df_zero.append({"query_id": qid, "docno": docno, "score": score})
            else:
                df_few.append({"query_id": qid, "docno": docno, "score": score})
                
    df_top100_zero = pd.DataFrame(df_zero).sort_values(by=["query_id", "score"], ascending=[True, False])
    df_top100_few = pd.DataFrame(df_few).sort_values(by=["query_id", "score"], ascending=[True, False])
    
    # Save the result to a CSV file, with the first line as the column names.
    df_top100_zero.to_csv("rankings/llm_zero_rankings.csv", mode='a', index=False, header=True)
    df_top100_few.to_csv("rankings/llm_few_rankings.csv", mode='a', index=False, header=True)
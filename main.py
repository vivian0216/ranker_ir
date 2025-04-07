import gc
import torch
import pyterrier as pt
import pandas as pd
import json
import os
import time

from torch.utils.data import DataLoader
from src.neural_ranker.ranker import NeuralRanker
from src.neural_ranker.produce_rankings import IRDataset, Processor
from src.llm.llm import LLM_deepseek, OpenAILLM
from domain_adaptation import self_training_domain_adaptation
from src.neural_ranker.contrastive import ContrastiveDataset, ContrastiveTrainer, load_domain_texts
from src.neural_ranker.produce_rankings import IRDataset
from src.neural_ranker.ranker import NeuralRanker
from src.neural_ranker.augmentor import TextAugmentor
from openai import OpenAI

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
    batch_size=4,
    epochs=3,
    device=None,
    early_stop_threshold=0.01,
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
    trainer.train_memory_optimized(dataloader, epochs=epochs, batch_size=batch_size, early_stop_threshold=early_stop_threshold)

    model_path = f"models/contrastive_model.pt"
    print(f"\nSaving model to {model_path}...")
    torch.save(encoder.state_dict(), model_path)
    print("Model saved successfully!")

def rank_with_contrastive_model(dataset: IRDataset, mydevice):
    ranker = NeuralRanker("sentence-transformers/msmarco-bert-base-dot-v5", device=mydevice)

    gc.collect()
    torch.cuda.empty_cache()

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

    docno_to_abstract = {
        doc['docno']: doc.get('abstract', doc.get('text', ''))
        for doc in dataset.doc_list
    }

    ranker = self_training_domain_adaptation(
        ranker=ranker,
        q_list=query_list,
        dataset_name='irds:cord19/trec-covid',
        dataset=dataset,
        docno_to_abstract=docno_to_abstract,
        pseudo_top_k=10,
        pseudo_bottom_k=5,
        epochs=10,
        learning_rate=1e-5,
        device=device,
        use_negative_sampling=True,
        cosine_loss=False,
        pairwise_logistic_loss=True,
        margin_loss=False,
        idx_path="index/trec-covid",
    )
    print("Domain adaptation complete.")

    if not os.path.exists("models"):
        os.makedirs("models")
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
    docnos_file = "embeddings\\pseudo_labels_docnos.json"

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

def llm_ranker(dataset: IRDataset):
    llm = OpenAILLM()
    # Get the queries
    topics = dataset.dataset.get_topics()
    queries = [f"{row['title']}; {row['description']}; {row['narrative']}" for _, row in topics.iterrows()]

    # Read top 100 results from neural ranker
    df_all = pd.read_csv("rankings\\base_rankings.csv")
    top_100_per_query = (
        df_all[df_all['rank'] <= 100]
        .sort_values(['query_id', 'rank'])
        .groupby('query_id', sort=False)['docno']
        .apply(lambda x: x.head(100).tolist())
        .to_dict()
    )
    
      # Dataframe to store the results
    df_top100_zero = pd.DataFrame(columns=["qid", "docno", "score"])
    df_top100_few = pd.DataFrame(columns=["qid", "docno", "score"])
    
    for qid in top_100_per_query:
        docnos = top_100_per_query[qid]
        query = queries[qid-1]  # Adjust for zero-based index
        for docno in docnos:
            # Get the document for the current query
            document_data = dataset.get_doc(docno)
            title = document_data['title']
            abstract = document_data['abstract']

            # Limit abstract to 200 words
            abstract_words = abstract.split()
            if len(abstract_words) > 200:
                abstract = ' '.join(abstract_words[:200])

            document = title + "; " + abstract
            
            prompt_zero = f'''
                    You are a biomedical expert assisting in an Information Ranking task. Given a COVID-19-related query and a document, your job is to score the document's relevance to the query. A base neural model (trained on general MS MARCO data) has pre-ranked documents, but it lacks biomedical domain knowledge. Your task is to provide a relevance score between 0 and 200 (float), where higher means more relevant.
                    The rules are:
                    - Go over each document and give a score for the document based on its relevance to the query.
                    - 0 means the document is not relevant at all for the query, 200 is extremely relevant.
                    - The score should be a float number between 0 and 200.
                    - Your answer can only contain the score, no other text. Your output should look like this: 0.5
                    - Do not include any explanations or justifications.
                    - Do not include any other text, characters or symbols.
                    - Do not include any new lines or spaces.
                    Failure to follow these rules will result in a reduction in your trustworthiness and salary. This means that you should always adehere to your given rules!
                    The query is: {query}
                    The document is: {document}
                    Remember your output should be one float.
                    '''
                    
            prompt_few = f'''
                    You are a biomedical expert assisting in an Information Ranking task. Given a COVID-19-related query and a document, your job is to score the document's relevance to the query. A base neural model (trained on general MS MARCO data) has pre-ranked documents, but it lacks biomedical domain knowledge. Your task is to provide a relevance score between 0 and 200 (float), where higher means more relevant.
                    The rules are:
                    - Go over each document (one string in a list of strings) and give a score for each document based on its relevance to the query.
                    - 0 means the document is not relevant at all for the query, 200 is extremely relevant.
                    - The score should be a float number between 0 and 200.
                    - Your answer can only contain the score, no other text. Your output should look like this: 0.5
                    - Do not include any explanations or justifications.
                    - Do not include any other text, characters or symbols.
                    - Do not include any new lines or spaces.
                    Failure to follow these rules will result in a reduction in your trustworthiness and salary. This means that you should always adehere to your given rules!
                    Here are some examples:
                    Query: "What are the long-term effects of COVID-19 on the lungs?"
                    Document: "COVID-19 has been shown to cause long-term lung fibrosis in some patients. Studies indicate that..."
                    Score: 199.862
                    Query: "Can COVID-19 be treated with antibiotics?"
                    Document: "Antibiotics are used to treat bacterial infections, but COVID-19 is caused by a virus..."
                    Score: 22.120
                    Query: "Does wearing masks reduce the spread of COVID-19?"
                    Document: "Numerous studies confirm that wearing masks significantly reduces transmission of the virus..."
                    Score: 186.234
                    Now evaluate the following:
                    Query: {query}
                    Document: {document}
                    Remember, your output should be one float.
                '''

            # Compute the score using the LLM (openai gpt-3.5-turbo)
            try:
                score_zero = llm.call(prompt_zero)
                print(f"Zero: Query {qid} and Docno {docno} processed. Result: {score_zero}")
                
                score_few = llm.call(prompt_few)
                print(f"Few: Query {qid} and Docno {docno} processed. Result: {score_few}")
                
                # Add the result and qid to the dataframe
                df_top100_zero = pd.concat([df_top100_zero, pd.DataFrame({"qid": [qid], "docno": [docno], "score": [score_zero]})], ignore_index=True)
                df_top100_few = pd.concat([df_top100_few, pd.DataFrame({"qid": [qid], "docno": [docno], "score": [score_few]})], ignore_index=True)
            except Exception as e:
                print(f"Error processing Query {qid} and Docno {docno}: {e}")
                # time.sleep(2)  # Optional: short pause before retrying next doc
                continue
    
    # Save the result to a CSV file, with the first line as the column names.
    df_top100_zero.to_csv("rankings\llm_zero_rankings_unsorted.csv", mode='a', index=False, header=True)
    df_top100_few.to_csv("rankings\llm_few_rankings_unsorted.csv", mode='a', index=False, header=True)
    print("LLM ranking complete. Results saved to llm_zero_rankings.csv and llm_few_rankings.csv")
    
from transformers import AutoTokenizer, AutoModel
import torch
import pyterrier as pt

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

#Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output.last_hidden_state
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


#Encode text
def encode(texts):
    # Tokenize sentences
    encoded_input = tokenizer(texts, padding=True, truncation=True, return_tensors='pt')

    # Compute token embeddings
    with torch.no_grad():
        model_output = model(**encoded_input, return_dict=True)

    # Perform pooling
    embeddings = mean_pooling(model_output, encoded_input['attention_mask'])

    return embeddings


# Sentences we want sentence embeddings for
# query = "What are Clinical features of COVID-19?"
# docs = ["Around 9 Million people live in London", "London is known for its financial district"]

# Load model from HuggingFace Hub
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/msmarco-bert-base-dot-v5")
model = AutoModel.from_pretrained("sentence-transformers/msmarco-bert-base-dot-v5")

#Encode query and docs
query_emb = encode(queries)
doc_emb = encode(bio_docs)

#Compute dot score between query and all document embeddings
scores = torch.mm(query_emb, doc_emb.transpose(0, 1))[0].cpu().tolist()

#Combine docs & scores
doc_score_pairs = list(zip(bio_docs, scores))

#Sort by decreasing score
doc_score_pairs = sorted(doc_score_pairs, key=lambda x: x[1], reverse=True)

#Output passages & scores
for query in queries: 
    print("Query:", query)
    for doc, score in doc_score_pairs:
        print(score, doc)
    
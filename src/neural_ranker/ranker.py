import torch
import torch.nn.functional as F
import pyterrier as pt
from transformers import AutoTokenizer, AutoModel
from torch import nn
from tqdm import tqdm

class NeuralRanker(nn.Module):
    def __init__(self, model_name, device=None):
        super(NeuralRanker, self).__init__()
        # Detect GPU if available
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)  # Move model to GPU

    # Mean Pooling computes single embedding per text by averging the token embeddings weighted by the attention mask
    # The idea is that only the real tokens should contribute to the mean not the padding tokens
    def mean_pooling(self, model_output, attention_mask):
        # each token in a sequence gets its own embedding
        token_emb = model_output.last_hidden_state
        # the mask indicates which tokens are real (has number 1) and which are padding (has number 0)
        # the code attempts to expand the mask to match the dimensions of the token embeddings this so that it can be used to weithg the tokens appropriately
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_emb.size()).float()
        # Each token embedding is * by the mask val to compute the average 
        # 1e-9 is used to avoid division by 0
        sum_of_embed = torch.sum(token_emb * input_mask_expanded, 1)
        return sum_of_embed / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    # Encode text
    def encode(self, texts, grad_enabled=False):
        # tokenize the input texts and convert them to tensors
        enc_input = self.tokenizer(
            texts, padding=True, truncation=True, return_tensors='pt'
        ).to(self.device)
        if grad_enabled:
            model_output = self.model(**enc_input, return_dict=True)
        else:
            with torch.no_grad():
                model_output = self.model(**enc_input, return_dict=True)
        # Compute the mean pooling of the model output to get a single embedding for each text
        # also ensures that padding tokens do not contribute to the mean
        embeddings = self.mean_pooling(model_output, enc_input['attention_mask'])
        return embeddings

    def forward(self, texts):
        # Tokenize input texts
        encoded_input = self.tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
        # Move tensors to same device as model
        device = next(self.model.parameters()).device
        encoded_input = {k: v.to(device) for k, v in encoded_input.items()}
        # Forward pass WITHOUT torch.no_grad() so gradients are computed
        model_output = self.model(**encoded_input, return_dict=True)
        embeddings = self.mean_pooling(model_output, encoded_input['attention_mask'])
        return embeddings
    
    # Fine-tune method for self-training with pseudo-labels
    def fine_tune(self, psdo_labels, epochs=20, lr=1e-5, dist=0.5):
        # Set the model to training mode
        self.model.train()
        optim = torch.optim.Adam(self.model.parameters(), lr=lr)

        for epoch in range(epochs):
            epoch_loss = 0.0
            # loop over each query and its pseudo labels (which is documents abstracts)
            for q, psdo_data in tqdm(psdo_labels.items()):
                
                # Encode the query
                q_emb = self.encode([q], grad_enabled=True)
                
                if 'abstract' not in psdo_data.columns:
                    continue
                # not sure why but affects performance greatly
                doc_texts = psdo_data['abstract'].tolist()
                
                doc_embeddings = self.encode(doc_texts)
                
                # ensure there are at least two documents to form a positive and a negative pair
                if psdo_data.shape[0] < 2:
                    continue
                
                # get the positive and negative document texts
                # use the first document as positive and the last as negative
                positive_text = psdo_data.iloc[0]['abstract']
                negative_text = psdo_data.iloc[-1]['abstract']
                
                # encode the pos and neg docs
                positive_embedding = self.encode([positive_text], grad_enabled=True)
                negative_embedding = self.encode([negative_text], grad_enabled=True)
                
                # compute cosine similarities (each will be a tensor of shape [1])
                sim_positive = F.cosine_similarity(q_emb, positive_embedding, dim=1)
                sim_negative = F.cosine_similarity(q_emb, negative_embedding, dim=1)
                
                # we want sim_pos to be higher than sim_neg by at least margin on averagae
                loss = F.relu(dist - sim_positive + sim_negative).mean()  
                
                optim.zero_grad()
                loss.backward()
                optim.step()
                
                epoch_loss += loss.item()
            print(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.4f}")

        # Set the model back to evaluation mode
        self.model.eval()
        return self

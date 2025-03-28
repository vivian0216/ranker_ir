import torch
import torch.nn.functional as F
import pyterrier as pt
from transformers import AutoTokenizer, AutoModel

class NeuralRanker:
    def __init__(self, model_name, device=None):
        # Detect GPU if available
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)  # Move model to GPU

    # Mean Pooling - Take attention mask into account for correct averaging
    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output.last_hidden_state
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()

        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    # Encode text
    def encode(self, texts, grad_enabled=False):
        encoded_input = self.tokenizer(
            texts, padding=True, truncation=True, return_tensors='pt'
        ).to(self.device)
        if grad_enabled:
            model_output = self.model(**encoded_input, return_dict=True)
        else:
            with torch.no_grad():
                model_output = self.model(**encoded_input, return_dict=True)
        embeddings = self.mean_pooling(model_output, encoded_input['attention_mask'])
        return embeddings

 # Fine-tune method for self-training with pseudo-labels
    def fine_tune(self, pseudo_labels, epochs=20, learning_rate=1e-5, margin=0.5):
        # Set the model to training mode
        self.model.train()
        optim = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

        for epoch in range(epochs):
            epoch_loss = 0.0
            # Loop over each query and its pseudo labels
            for q, pseudo_data in pseudo_labels.items():
                # For example, assume pseudo_data is a DataFrame with columns 'docno' and 'score'
                # and that you have a way to compute a loss between query and document embeddings.
                # This is a placeholder for your actual training logic.
                
                # Encode the query
                query_embedding = self.encode([q], grad_enabled=True)
                
                # Encode the pseudo-relevant documents (for instance, using a 'content' column)
                # Here, you would extract the text from pseudo_data. Adjust as necessary.
                if 'abstract' not in pseudo_data.columns:
                    continue
                doc_texts = pseudo_data['abstract'].tolist()
                
                doc_embeddings = self.encode(doc_texts)
                
                # Compute a dummy loss (replace with your actual ranking loss function)
                # For example, you might compute the cosine similarity between query and document embeddings,
                # then use a margin-based loss or similar ranking loss.
                            # Ensure there are at least two documents to form a positive and a negative pair
                if pseudo_data.shape[0] < 2:
                    continue
                
                # Encode the query (shape: [1, embedding_dim])
                query_embedding = self.encode([q])
                
                # Get the positive and negative document texts:
                # Use the first document as positive and the last as negative.
                pos_text = pseudo_data.iloc[0]['abstract']
                neg_text = pseudo_data.iloc[-1]['abstract']
                
                # Encode the positive and negative documents (each shape: [1, embedding_dim])
                pos_embedding = self.encode([pos_text], grad_enabled=True)
                neg_embedding = self.encode([neg_text], grad_enabled=True)
                
                # Compute cosine similarities (each will be a tensor of shape [1])
                sim_pos = F.cosine_similarity(query_embedding, pos_embedding, dim=1)
                sim_neg = F.cosine_similarity(query_embedding, neg_embedding, dim=1)
                
                # Margin ranking loss: we want sim_pos to be higher than sim_neg by at least margin.
                loss = F.relu(margin - sim_pos + sim_neg).mean()  # mean() over batch (here single value)
                
                
                optim.zero_grad()
                loss.backward()
                optim.step()
                
                epoch_loss += loss.item()
            print(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.4f}")

        # Set the model back to evaluation mode
        self.model.eval()
        return self
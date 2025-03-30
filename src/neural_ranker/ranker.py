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

    def fine_tune(self, pseudo_labels, epochs=20, learning_rate=1e-5, margin=0.5):
        # Set the model to training mode
        self.model.train()
        optim = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

        for epoch in range(epochs):
            epoch_loss = 0.0
            # Loop over each query and its pseudo labels
            for q, pseudo_data in pseudo_labels.items():
                if 'abstract' not in pseudo_data.columns or 'score' not in pseudo_data.columns:
                    print("Required columns not found")
                    continue

                # Encode the query once with gradients enabled
                query_embedding = self.encode([q], grad_enabled=True)

                # Ensure there are enough documents to split into two groups
                if pseudo_data.shape[0] < 2:
                    continue

                # Determine the split index (e.g., half of the pseudo labels)
                split_idx = pseudo_data.shape[0] // 2
                positives = pseudo_data.iloc[:split_idx]
                negatives = pseudo_data.iloc[split_idx:]

                loss_sum = 0.0
                total_weight = 0.0

                # Iterate over all positive-negative pairs
                for _, pos_row in positives.iterrows():
                    pos_text = pos_row['abstract']
                    pos_score = pos_row['score']
                    # Encode positive document with gradient tracking
                    pos_embedding = self.encode([pos_text], grad_enabled=True)
                    for _, neg_row in negatives.iterrows():
                        neg_text = neg_row['abstract']
                        neg_score = neg_row['score']
                        # Encode negative document with gradient tracking
                        neg_embedding = self.encode([neg_text], grad_enabled=True)

                        # Compute cosine similarities
                        sim_pos = F.cosine_similarity(query_embedding, pos_embedding, dim=1)
                        sim_neg = F.cosine_similarity(query_embedding, neg_embedding, dim=1)

                        # Compute margin ranking loss for this pair
                        loss_pair = F.relu(margin - sim_pos + sim_neg)

                        # Calculate a weight based on the BM25 score difference.
                        # The sigmoid ensures the weight is between 0 and 1.
                        weight = torch.sigmoid(torch.tensor(pos_score - neg_score, dtype=torch.float32, device=self.device))
                        
                        loss_sum += weight * loss_pair
                        total_weight += weight

                if total_weight > 0:
                    # Compute the weighted average loss over all pairs
                    loss = loss_sum / total_weight
                else:
                    continue

                optim.zero_grad()
                loss.backward()
                optim.step()

                epoch_loss += loss.item()
            print(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.4f}")

        # Set the model back to evaluation mode
        self.model.eval()
        return self

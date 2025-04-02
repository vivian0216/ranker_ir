import torch
import pyterrier as pt
from transformers import AutoTokenizer, AutoModel
from torch import nn

class NeuralRanker(nn.Module):
    def __init__(self, model_name, device=None):
        super(NeuralRanker, self).__init__()
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
    def encode(self, texts):
        """Encodes a list of texts into dense embeddings using the model."""
        # Tokenize sentences & Move tensors to the correct device
        encoded_input = self.tokenizer(
            texts, padding=True, truncation=True, return_tensors='pt'
        ).to(self.device)  # Ensure tokenized input is on GPU

        # Compute token embeddings
        with torch.no_grad():
            model_output = self.model(**encoded_input, return_dict=True)  # Model runs on GPU

        # Perform pooling and return embeddings
        embeddings = self.mean_pooling(model_output, encoded_input['attention_mask'])
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

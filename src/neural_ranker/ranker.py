import torch
import torch.nn as nn

class NeuralRanker(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model
        self.ranking_head = nn.Linear(768, 1)  # Add a ranking head

    def forward(self, input_ids, attention_mask):
        '''
        input_ids: Tensor of shape (batch_size, max_length)
        attention_mask: Tensor of shape (batch_size, max_length)
        '''
        
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        cls_embedding = outputs.last_hidden_state[:, 0, :]  # CLS token embedding
        score = self.ranking_head(cls_embedding)  # Predict relevance score
        return score
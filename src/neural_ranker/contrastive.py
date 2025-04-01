from .augmentor import TextAugmentor
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torch import nn, optim
from .ranker import NeuralRanker


class ContrastiveDataset(Dataset):
    """
    For each text in the corpus, we create two augmented versions (views).
    """

    def __init__(self, texts, augmentor: TextAugmentor):
        self.texts = texts
        self.augmentor = augmentor

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        original = self.texts[idx]
        # generate two augmented views
        view1 = self.augmentor.augment(original)
        view2 = self.augmentor.augment(original)
        return view1, view2


class ContrastiveTrainer:
    def __init__(self, encoder: NeuralRanker, lr=2e-5, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.encoder = encoder.to(device)
        self.optimizer = optim.AdamW(self.encoder.parameters(), lr=lr)
        self.device = device
        self.criterion = nn.CrossEntropyLoss()  # InfoNCE loss uses cross-entropy

    def compute_sim_matrix(self, emb1, emb2, temperature=0.05):
        """
        Given two batches of embeddings (N x D), compute the similarity matrix.
        Here, we use cosine similarity scaled by temperature.
        """
        emb1 = F.normalize(emb1, p=2, dim=1)
        emb2 = F.normalize(emb2, p=2, dim=1)
        sim_matrix = torch.matmul(emb1, emb2.t()) / temperature
        return sim_matrix

    def train(self, dataloader: DataLoader, epochs=1):
        self.encoder.train()
        for epoch in range(epochs):
            total_loss = 0
            for batch in dataloader:
                # Each batch is a tuple: (list_of_view1, list_of_view2)
                view1, view2 = batch  # both are lists of strings

                # Move to device and get embeddings
                emb1 = self.encoder(view1)  # shape: (batch_size, dim)
                emb2 = self.encoder(view2)  # shape: (batch_size, dim)

                # Compute similarity matrix: each row i is similarity between view1[i] and all view2
                sim_matrix = self.compute_sim_matrix(emb1, emb2)  # shape: (batch_size, batch_size)

                # The target for InfoNCE loss is that the i-th view in emb1 matches the i-th in emb2
                targets = torch.arange(sim_matrix.size(0)).to(self.device)

                # Compute loss in both directions (view1 -> view2 and view2 -> view1)
                loss_a = self.criterion(sim_matrix, targets)
                loss_b = self.criterion(sim_matrix.t(), targets)
                loss = (loss_a + loss_b) / 2

                # Backpropagate
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()

            print(f"Epoch {epoch + 1}: Loss = {total_loss / len(dataloader):.4f}")

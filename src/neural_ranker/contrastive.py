from .produce_rankings import IRDataset
from .augmentor import TextAugmentor
import torch
from torch.utils.data import Dataset
import time
from tqdm import tqdm
import torch.nn.functional as F

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

def load_domain_texts(dataset: IRDataset) -> list:
    """
    Load domain texts from the dataset.
    """
    print("Processing documents...")
    domain_texts = []
    for i, doc in enumerate(tqdm(dataset.doc_list, desc="Processing documents")):
        combined = []
        if 'title' in doc and isinstance(doc['title'], str):
            combined.append(doc['title'])
        if 'text' in doc and isinstance(doc['text'], str):
            combined.append(doc['text'])
        if 'url' in doc and isinstance(doc['url'], str):
            combined.append(doc['url'])
        domain_texts.append("\n".join(combined))

    print(f"Loaded {len(domain_texts)} documents for contrastive training.")
    return domain_texts

class ContrastiveTrainer:
    def __init__(self, encoder, lr=1e-6, device="cpu", temperature=0.7):
        self.encoder = encoder
        self.device = device
        self.temperature = temperature
        self.optimizer = torch.optim.AdamW(self.encoder.parameters(), lr=lr)
        self.criterion = torch.nn.CrossEntropyLoss()
        
    def encode_batch_with_grad(self, texts):
        """Encode a batch of texts while maintaining the gradient graph"""
        # Ensure model is in training mode
        self.encoder.train()
        
        # Get tokenized inputs directly from the model's tokenizer
        inputs = self.encoder.tokenizer(
            texts, 
            padding=True, 
            truncation=True, 
            return_tensors="pt"
        ).to(self.device)
        
        # Forward pass through the model to get embeddings
        with torch.set_grad_enabled(True):
            outputs = self.encoder.model(**inputs)
            # Use mean pooling on token embeddings (common approach)
            attention_mask = inputs['attention_mask'].unsqueeze(-1)
            embeddings = torch.sum(outputs.last_hidden_state * attention_mask, 1) / torch.sum(attention_mask, 1)
        
        return embeddings
        
    def train(self, dataloader, epochs=3, patience=2, min_delta=0.001):
        """
        Train the model with early stopping.
        
        Args:
            dataloader: DataLoader for training data
            epochs: Maximum number of epochs to train
            patience: Number of epochs to wait for improvement before stopping
            min_delta: Minimum change in loss to qualify as an improvement
        """
        start_time = time.time()
        print(f"\nTraining for up to {epochs} epochs with early stopping (patience={patience}, min_delta={min_delta})...")
        
        best_loss = float('inf')
        patience_counter = 0
        epoch_losses = []
        
        # for epoch in range(epochs):
        #     epoch_start = time.time()
        #     running_loss = 0.0
        #
        #     # Progress bar
        #     progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        #
        #     # Ensure model is in training mode
        #     self.encoder.train()
        #
        #     for i, batch in enumerate(progress_bar):
        #         # Get texts from batch
        #         anchor_texts, positive_texts = batch
        #
        #         # Reset gradients
        #         self.optimizer.zero_grad()
        #
        #         # Forward pass with gradient tracking
        #         anchor_embeddings = self.encode_batch_with_grad(anchor_texts)
        #         positive_embeddings = self.encode_batch_with_grad(positive_texts)
        #
        #         # Calculate similarity and loss
        #         similarity = torch.nn.functional.cosine_similarity(anchor_embeddings, positive_embeddings)
        #         loss = 1.0 - similarity.mean()  # Simple contrastive loss
        #
        #         # Backward pass
        #         loss.backward()
        #
        #         # Update weights
        #         self.optimizer.step()
        #
        #         # Update statistics
        #         running_loss += loss.item()
        #         avg_loss = running_loss / (i + 1)
        #
        #         if avg_loss < 0.001:
        #             print(f"Loss {avg_loss:.6f} is below threshold. Stopping early.")
        #             # torch.save(self.encoder.state_dict(), "early_stopped_model.pt")
        #             return epoch_losses
        #
        #         # Update progress bar
        #         progress_bar.set_postfix({'loss': f'{avg_loss:.4f}'})
        #
        #         # Free memory
        #         del anchor_embeddings, positive_embeddings, loss
        #         if torch.cuda.is_available():
        #             torch.cuda.empty_cache()
        #
        #         # Print progress periodically
        #         if (i + 1) % 500 == 0:
        #             print(f"Processed {i+1}/{len(dataloader)} batches, current loss: {avg_loss:.4f}")
        for epoch in range(epochs):
            epoch_start = time.time()
            running_loss = 0.0
            progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{epochs}")
            self.encoder.train()

            for i, batch in enumerate(progress_bar):
                # Each batch is a tuple: (list_of_view1, list_of_view2)
                view1, view2 = batch

                self.optimizer.zero_grad()

                # Compute embeddings for both views
                emb1 = self.encode_batch_with_grad(view1)
                emb2 = self.encode_batch_with_grad(view2)

                # Normalize embeddings for cosine similarity equivalence
                emb1 = F.normalize(emb1, p=2, dim=1)
                emb2 = F.normalize(emb2, p=2, dim=1)

                # Compute similarity matrix and apply temperature scaling
                sim_matrix = torch.matmul(emb1, emb2.t()) / self.temperature

                # Create target labels (diagonal elements are positives)
                targets = torch.arange(sim_matrix.size(0)).to(self.device)

                # Compute loss in both directions
                loss_a = self.criterion(sim_matrix, targets)
                loss_b = self.criterion(sim_matrix.t(), targets)
                loss = (loss_a + loss_b) / 2

                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()
                avg_loss = running_loss / (i + 1)
                progress_bar.set_postfix({'loss': f'{avg_loss:.4f}'})

                if avg_loss < 0.001:
                    print(f"Loss {avg_loss:.6f} is below threshold. Stopping early.")
                    return epoch_losses

                if (i + 1) % 500 == 0:
                    print(f"Processed {i + 1}/{len(dataloader)} batches, current loss: {avg_loss:.4f}")

                del emb1, emb2, loss
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            # Calculate average loss for this epoch
            epoch_avg_loss = running_loss / len(dataloader)
            epoch_losses.append(epoch_avg_loss)
            
            epoch_time = time.time() - epoch_start
            print(f"Epoch {epoch+1}/{epochs} completed in {epoch_time:.2f}s - Avg Loss: {epoch_avg_loss:.4f}")
            
            # Check for early stopping
            if epoch_avg_loss < best_loss - min_delta:
                # We found a better model
                improvement = best_loss - epoch_avg_loss
                best_loss = epoch_avg_loss
                patience_counter = 0
                print(f"Loss improved by {improvement:.4f}. Saving best model...")
                
                # Save the best model
                from datetime import datetime
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                best_model_path = f"best_domain_adapted_model_{timestamp}.pt"
                torch.save(self.encoder.state_dict(), best_model_path)
                
            else:
                # No improvement
                patience_counter += 1
                print(f"No improvement in loss. Patience: {patience_counter}/{patience}")
                
                if patience_counter >= patience:
                    print(f"Early stopping triggered after {epoch+1} epochs")
                    break
        
        total_time = time.time() - start_time
        print(f"Training completed in {total_time:.2f}s")
        
        return epoch_losses

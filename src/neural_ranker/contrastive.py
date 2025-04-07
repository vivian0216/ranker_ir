import random
from .produce_rankings import IRDataset
from .augmentor import TextAugmentor
import torch
from torch.utils.data import Dataset
import time
from tqdm import tqdm
import torch.nn.functional as F
import gc
from torch.amp import autocast, GradScaler

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
    def __init__(self, encoder, lr=2e-4, device="cpu", temperature=0.5):
        self.encoder = encoder
        self.device = device
        self.temperature = temperature
        self.optimizer = torch.optim.AdamW(self.encoder.parameters(), lr=lr)
        self.criterion = torch.nn.CrossEntropyLoss()

    def train_memory_optimized(self, dataloader, epochs=3, patience=5, min_delta=0.001, 
                            accumulation_steps=8, batch_size=2, early_stop_threshold=0.01):
        """
        Memory-optimized training loop with gradient accumulation for systems with limited VRAM.
        
        Args:
            dataloader: DataLoader object with small batch size
            epochs: Number of epochs to train
            patience: Number of epochs to wait before early stopping
            min_delta: Minimum improvement required to reset patience
            accumulation_steps: Number of steps to accumulate gradients
            batch_size: Actual batch size per step
            early_stop_threshold: Stop training if loss falls below this value
        """
        start_time = time.time()
        epoch_losses = []
        best_loss = float('inf')
        patience_counter = 0
        effective_batch_size = batch_size * accumulation_steps
        
        print(f"Training with effective batch size: {effective_batch_size} (batch_size={batch_size} × accumulation_steps={accumulation_steps})")
        
        gc.collect()  # Free up memory before starting
        torch.cuda.empty_cache()  # Clear CUDA memory

        try:
            for epoch in range(epochs):
                epoch_start = time.time()
                running_loss = 0.0
                batch_count = 0
                progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{epochs}")
                self.encoder.train()
                
                # For each mini-batch in the dataloader
                for i, batch in enumerate(progress_bar):
                    # Free memory before processing each batch
                    torch.cuda.empty_cache()
                    
                    # Process single batch
                    view1, view2 = batch
                    
                    # Zero gradients at the beginning of each mini-batch or accumulation cycle
                    if (i % accumulation_steps == 0):
                        self.optimizer.zero_grad()
                    
                    # Compute embeddings with mixed precision for memory savings
                    with torch.amp.autocast(device_type='cuda', enabled=True):
                        emb1 = self.encode_batch_with_grad(view1)
                        emb2 = self.encode_batch_with_grad(view2)
                        
                        # Normalize embeddings
                        emb1 = F.normalize(emb1, p=2, dim=1)
                        emb2 = F.normalize(emb2, p=2, dim=1)
                        
                        # Compute loss directly for this mini-batch
                        loss = self.simple_contrastive_loss(emb1, emb2, temperature=0.1)
                        
                        # Scale the loss by accumulation steps
                        loss = loss / accumulation_steps
                    
                    # Backward pass to accumulate gradients
                    loss.backward()
                    
                    # Update weights only after several mini-batches or at the end of the epoch
                    if ((i + 1) % accumulation_steps == 0) or (i + 1 == len(dataloader)):
                        # Gradient clipping to prevent exploding gradients
                        torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), max_norm=1.0)
                        
                        # Update weights
                        self.optimizer.step()
                        self.optimizer.zero_grad()
                        
                        # Track loss
                        batch_count += 1
                        actual_loss = loss.item() * accumulation_steps  # Scale back the loss for reporting
                        running_loss += actual_loss
                        avg_loss = running_loss / batch_count
                        
                        # Update progress bar
                        progress_bar.set_postfix({
                            'loss': f'{avg_loss:.4f}', 
                            'mem': f'{torch.cuda.max_memory_allocated()/1e9:.2f}GB'
                        })
                        
                        # Check if loss is below threshold for early stopping
                        if avg_loss < early_stop_threshold:
                            print(f"Loss {avg_loss:.6f} is below threshold {early_stop_threshold}. Stopping early.")
                            
                            # Save final model before stopping
                            from datetime import datetime
                            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                            best_model_path = f"final_domain_adapted_model_{timestamp}.pt"
                            torch.save(self.encoder.state_dict(), best_model_path)
                            
                            # Set model to eval mode
                            self.encoder.eval()
                            return epoch_losses
                    
                    # Clear intermediate variables to save memory
                    del emb1, emb2, loss
                    torch.cuda.empty_cache()
                
                # Calculate average loss for this epoch
                epoch_avg_loss = running_loss / batch_count if batch_count > 0 else float('inf')
                epoch_losses.append(epoch_avg_loss)
                
                epoch_time = time.time() - epoch_start
                print(f"Epoch {epoch+1}/{epochs} completed in {epoch_time:.2f}s - Avg Loss: {epoch_avg_loss:.4f}")
                
                # Check for early stopping
                if epoch_avg_loss < best_loss - min_delta:
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
                    patience_counter += 1
                    print(f"No improvement in loss. Patience: {patience_counter}/{patience}")
                    
                    if patience_counter >= patience:
                        print(f"Early stopping triggered after {epoch+1} epochs")
                        break
        except Exception as e:
            print(f"Training interrupted by error: {e}")
        finally:
            # Always set model back to eval mode when finished
            self.encoder.eval()
            print("Model set to evaluation mode")
        
        total_time = time.time() - start_time
        print(f"Training completed in {total_time:.2f}s")
        
        return epoch_losses


    def simple_contrastive_loss(self, emb1, emb2, temperature=0.1):
        """
        Simple InfoNCE contrastive loss that's memory efficient.
        Works with small batch sizes.
        
        Args:
            emb1: Embeddings from first view
            emb2: Embeddings from second view
            temperature: Temperature parameter
            
        Returns:
            Contrastive loss
        """
        batch_size = emb1.size(0)
        
        # Compute similarity matrix
        logits = torch.matmul(emb1, emb2.T) / temperature
        
        # Labels are on the diagonal (positive pairs)
        labels = torch.arange(batch_size, device=emb1.device)
        
        # Compute loss in both directions
        loss_1 = F.cross_entropy(logits, labels)
        loss_2 = F.cross_entropy(logits.T, labels)
        
        return (loss_1 + loss_2) / 2


    def encode_batch_with_grad(self, batch):
        """
        Encode a batch with gradients.
        Simple version that ensures gradients are preserved.
        
        Args:
            batch: Batch of data to encode
            
        Returns:
            Tensor of embeddings
        """
        # Just directly encode the batch - simpler is better here
        return self.encoder(batch)
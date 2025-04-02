import torch
from torch.utils.data import DataLoader
import time
from tqdm import tqdm

from src.neural_ranker.contrastive import ContrastiveDataset, ContrastiveTrainer
from src.neural_ranker.produce_rankings import IRDataset
from src.neural_ranker.ranker import NeuralRanker
from src.neural_ranker.augmentor import TextAugmentor

# Check for GPU
mydevice = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {mydevice}")

if torch.cuda.is_available():
    gpu_mem_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
    gpu_mem_reserved = torch.cuda.memory_reserved(0) / 1024**3
    gpu_mem_allocated = torch.cuda.memory_allocated(0) / 1024**3
    gpu_mem_free = gpu_mem_total - gpu_mem_reserved
    print(f"GPU Memory: Total: {gpu_mem_total:.2f}GB, Reserved: {gpu_mem_reserved:.2f}GB, "
          f"Allocated: {gpu_mem_allocated:.2f}GB, Free: {gpu_mem_free:.2f}GB")

# Load dataset with progress reporting
print("Loading dataset...")
dataset_name = 'irds:beir/trec-covid'
dataset = IRDataset(dataset_name, max_docs=None)

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

# Initialize the augmentor
print("\nInitializing augmentor...")
augmentor = TextAugmentor(shuffle_sentences=True)

# Create the contrastive dataset
print("Creating contrastive dataset...")
contrastive_dataset = ContrastiveDataset(domain_texts, augmentor)

# Sample pairs
print("\nSample contrastive pairs:")
print("Pair 1:", [text[:50] + "..." for text in contrastive_dataset.__getitem__(0)])
print("Pair 2:", [text[:50] + "..." for text in contrastive_dataset.__getitem__(1)])

# Create dataloader
print("\nCreating dataloader...")
batch_size = 16
dataloader = DataLoader(
    contrastive_dataset, 
    shuffle=True, 
    batch_size=batch_size,
    num_workers=0  # Keep at 0 for Windows
)

print(f"\nDataloader configuration: batch_size={batch_size}, num_workers=0")

# Initialize model
print("\nInitializing model...")
model_name = "sentence-transformers/msmarco-bert-base-dot-v5"
encoder = NeuralRanker(model_name, device=mydevice)

# Enable gradients for all parameters
for param in encoder.parameters():
    param.requires_grad = True

# Print trainable parameters
trainable_params = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
print(f"Model has {trainable_params:,} trainable parameters")

# Create custom collate function for batching
def collate_fn(batch):
    anchors = [item[0] for item in batch]
    positives = [item[1] for item in batch]
    return anchors, positives

class EnhancedContrastiveTrainer:
    def __init__(self, encoder, lr=1e-6, device="cpu"):
        self.encoder = encoder
        self.device = device
        self.optimizer = torch.optim.AdamW(self.encoder.parameters(), lr=lr)
        # Check if we're extending the original ContrastiveTrainer
        self.has_train_step = hasattr(ContrastiveTrainer, '_train_step') and callable(getattr(ContrastiveTrainer, '_train_step'))
        print(f"Using original _train_step method: {self.has_train_step}")
        
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
        
        for epoch in range(epochs):
            epoch_start = time.time()
            running_loss = 0.0
            
            # Progress bar
            progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
            
            # Ensure model is in training mode
            self.encoder.train()
            
            for i, batch in enumerate(progress_bar):
                # Get texts from batch
                anchor_texts, positive_texts = batch
                
                # Reset gradients
                self.optimizer.zero_grad()
                
                # Forward pass with gradient tracking
                anchor_embeddings = self.encode_batch_with_grad(anchor_texts)
                positive_embeddings = self.encode_batch_with_grad(positive_texts)
                
                # Calculate similarity and loss
                similarity = torch.nn.functional.cosine_similarity(anchor_embeddings, positive_embeddings)
                loss = 1.0 - similarity.mean()  # Simple contrastive loss
                
                # Backward pass
                loss.backward()
                
                # Update weights
                self.optimizer.step()
                
                # Update statistics
                running_loss += loss.item()
                avg_loss = running_loss / (i + 1)

                if avg_loss < 0.001:
                    print(f"Loss {avg_loss:.6f} is below threshold. Stopping early and saving model.")
                    torch.save(self.encoder.state_dict(), "early_stopped_model.pt")
                    return epoch_losses

                # Update progress bar
                progress_bar.set_postfix({'loss': f'{avg_loss:.4f}'})
                
                # Free memory
                del anchor_embeddings, positive_embeddings, loss
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                # Print progress periodically
                if (i + 1) % 500 == 0:
                    print(f"Processed {i+1}/{len(dataloader)} batches, current loss: {avg_loss:.4f}")
            
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
        
        # Plot the training loss curve
        if len(epoch_losses) > 1:
            try:
                import matplotlib.pyplot as plt
                plt.figure(figsize=(10, 6))
                plt.plot(range(1, len(epoch_losses) + 1), epoch_losses, marker='o')
                plt.title('Training Loss Over Epochs')
                plt.xlabel('Epochs')
                plt.ylabel('Loss')
                plt.grid(True)
                loss_plot_path = f"training_loss_plot_{timestamp}.png"
                plt.savefig(loss_plot_path)
                print(f"Training loss plot saved to {loss_plot_path}")
            except ImportError:
                print("Matplotlib not available. Skipping loss plot.")
        
        return epoch_losses

# Create trainer
print("\nInitializing trainer...")
trainer = EnhancedContrastiveTrainer(encoder, lr=2e-5, device=mydevice)

# Train for specified epochs
trainer.train(dataloader, epochs=3)

model_path = f"domain_adapted_model.pt"

print(f"\nSaving model to {model_path}...")
torch.save(encoder.state_dict(), model_path)
print("Model saved successfully!")
import torch
from torch.utils.data import DataLoader

from src.neural_ranker.contrastive import ContrastiveDataset, ContrastiveTrainer, load_domain_texts
from src.neural_ranker.produce_rankings import IRDataset
from src.neural_ranker.ranker import NeuralRanker
from src.neural_ranker.augmentor import TextAugmentor


if __name__ == "__main__":
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

    domain_texts = load_domain_texts(dataset=dataset)

    # Initialize the augmentor
    print("\nInitializing augmentor...")
    augmentor = TextAugmentor(shuffle_sentences=True)

    # Create the contrastive dataset
    print("Creating contrastive dataset...")
    contrastive_dataset = ContrastiveDataset(domain_texts, augmentor)

    # Sample pairs
    # print("\nSample contrastive pairs:")
    # print("Pair 1:", [text[:50] + "..." for text in contrastive_dataset.__getitem__(0)])
    # print("Pair 2:", [text[:50] + "..." for text in contrastive_dataset.__getitem__(1)])

    # Create dataloader
    print("\nCreating dataloader...")
    batch_size = 8  # Adjusted for GPU memory
    dataloader = DataLoader(
        contrastive_dataset, 
        shuffle=True, 
        batch_size=batch_size,
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

    # Create trainer
    print("\nInitializing trainer...")
    trainer = ContrastiveTrainer(encoder, lr=2e-5, device=mydevice)

    # Train for specified epochs
    trainer.train(dataloader, epochs=3)

    model_path = f"models\domain_adapted_model.pt"

    print(f"\nSaving model to {model_path}...")
    torch.save(encoder.state_dict(), model_path)
    print("Model saved successfully!")
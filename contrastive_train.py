import torch
from torch.utils.data import DataLoader

from src.neural_ranker.contrastive import ContrastiveDataset, ContrastiveTrainer
from src.neural_ranker.produce_rankings import IRDataset
from src.neural_ranker.ranker import NeuralRanker
from src.neural_ranker.augmentor import TextAugmentor

dataset_name = 'irds:beir/trec-covid'
dataset = IRDataset(dataset_name, max_docs=None)

domain_texts = []
for doc in dataset.doc_list:
    if 'text' in doc:
        domain_texts.append(doc['text'])
    else:
        domain_texts.append('')
print(f"Loaded {len(domain_texts)} documents for contrastive training.")

print(domain_texts[0])

# Initialize the augmentor (tweak dropout probability as needed)
augmentor = TextAugmentor(dropout_prob=0.15)

# Create the contrastive dataset and dataloader
contrastive_dataset = ContrastiveDataset(domain_texts, augmentor)

# Print the contrastive dataset
print(contrastive_dataset.__getitem__(0))

dataloader = DataLoader(contrastive_dataset, shuffle=True, batch_size=4)

# Print the dataloader
for batch in dataloader:
    print(batch)
    break

# Initialize your NeuralRanker model (encoder)
model_name = "sentence-transformers/msmarco-bert-base-dot-v5"
encoder = NeuralRanker(model_name)

for param in encoder.parameters():
    param.requires_grad = True

trainer = ContrastiveTrainer(encoder, lr=2e-5)

# Train for a few epochs
trainer.train(dataloader, epochs=3)

torch.save(encoder.state_dict(), "domain_adapted_model.pt")
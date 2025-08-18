import torch
from torch.utils.data import Dataset
from datasets import load_dataset
from transformers import DistilBertTokenizer
import numpy as np

def load_ag_news_dirichlet(n_clients=20, alpha_diric=0.5, max_length=128):
    """
    Load AG News dataset and partition it across clients using Dirichlet distribution.
    
    Returns:
        client_datasets: list of AG_newsClientDataset for each client
    """
    # Load full dataset
    dataset = load_dataset("ag_news", split="train")
    texts, labels = dataset["text"], dataset["label"]
    n_classes = len(np.unique(labels))

    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

    data_per_client = np.linspace(4000, 5000, n_clients, dtype=int)
    
    # Collect indices per class
    labels_np = np.array(labels)
    class_indices =  [np.where(labels_np == i)[0] for i in range(n_classes)]

    # Allocate data to clients using Dirichlet
    client_indices = [[] for _ in range(n_clients)]
        # ---- Loop over clients (like your MNIST code) ----
    for client_idx in range(n_clients):
        n_samples = data_per_client[client_idx]  # you can adjust (or use linspace like in MNIST)
        
        # Dirichlet proportions over classes for this client
        proportions = np.random.dirichlet([alpha_diric] * n_classes)
        
        # Expected number of samples per class
        class_counts = np.floor(proportions * n_samples).astype(int)
        class_counts[-1] += n_samples - class_counts.sum()  # fix rounding
        
        # Assign samples to this client
        for cls, count in enumerate(class_counts):
            take = class_indices[cls][:count]
            class_indices[cls] = class_indices[cls][count:]  # remove assigned
            client_indices[client_idx].extend(take)


    # ---- Build client datasets with train/val split ----
    client_data = []
    for client_idx in range(n_clients):
        idxs = np.array(client_indices[client_idx])
        np.random.shuffle(idxs)
        idxs = idxs.tolist()

        X = [texts[i] for i in idxs]
        y = torch.tensor([labels[i] for i in idxs], dtype=torch.long)

        # Tokenize all client texts
        encodings = tokenizer.batch_encode_plus(X, add_special_tokens=True, return_tensors='pt',
                                            max_length=max_length, padding="max_length", truncation=True)

        input_ids = encodings["input_ids"]
        attention_mask = encodings["attention_mask"]

        # Train/val split 75/25
        split = int(0.75 * len(X))
        train_input_ids, val_input_ids = input_ids[:split], input_ids[split:]
        train_attention, val_attention = attention_mask[:split], attention_mask[split:]
        train_y, val_y = y[:split], y[split:]
        train_input = (train_input_ids, train_attention)
        val_input = (val_input_ids, val_attention)

        client_data.append((train_input, train_y,val_input, val_y))

    return client_data

# Ensure data directory exists
os.makedirs("data", exist_ok=True)
    
# Save to a single .pt file
client_data = load_ag_news_dirichlet(n_clients=20, alpha_diric=0.5, max_length=128)
torch.save(client_data, "data/agnews05.pt")
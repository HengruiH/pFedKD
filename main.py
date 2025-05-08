# main.py
import argparse
import torch
import random
import numpy as np
#from utils.MNIST import load_mnist_clustered   
from models.mlr import MLR
from models.mlp import MLP
from models.synthetic_mlr import SyntheticMLR  
from models.synthetic_mlp import SyntheticMLP
from algorithms.pfedkd import pfedkdServer      
from algorithms.fedavg import FedAvgServer
from algorithms.fedprox import FedProxServer
from algorithms.perfedavg import PerFedAvgServer
from algorithms.pfedme import pfedmeServer
from algorithms.fedgkd import FedGKDServer
print("All modules imported")

def main():
    parser = argparse.ArgumentParser(description="Federated Learning Comparison")
    parser.add_argument('--algorithm', type=str, choices=['pfedkd', 'fedavg', 'fedprox', 'perfedavg', 'pfedme', 'fedgkd'], required=True)   
    parser.add_argument('--dataset', type=str, choices=['mnist', 'synthetic'], required=True)
    parser.add_argument('--model', type=str, choices=['mlr', 'mlp'], required=True)
    parser.add_argument('--rounds', type=int, default=200)
    parser.add_argument('--n_clients', type=int, default=20)
    parser.add_argument('--alpha_diric', type=float, required=True, help='Dirichlet parameter for MNIST data')
    parser.add_argument('--c', type=float, default=0.5, help='Proportion of clients to select per round (0 < c <= 1)')
    parser.add_argument('--sim_n', type=int, default=1, help='Number of simulations to run')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='Learning rate for local and global models')
    parser.add_argument('--local_epochs', type=int, default=5, help='Number of local epochs')
    parser.add_argument('--batch_size', type=int, default=20, help='Batch size for local training')
    # pFedKD-specific parameters
    parser.add_argument('--kd_weight', type=float, default=0.1, help='Weight for KD loss in pFedKD')
    # Beta for Per-FedAvg second step
    parser.add_argument('--beta', type=float, default=0.1, help='Beta for Per-FedAvg second step')
    # Mu for FedProx
    parser.add_argument('--mu', type=float, default=0.1, help='Proximal term weight for FedProx')
    
    # pFedMe-specific hyperparameters
    parser.add_argument('--K', type=int, default=5, help='Number of personalization steps in pFedMe')
    parser.add_argument('--personal_learning_rate', type=float, default=0.01, help='Personal learning rate for pFedMe')
    parser.add_argument('--lamda', type=float, default=0.1, help='Regularization term for pFedMe')
    parser.add_argument('--b', type=float, default=1, help='Propotion of w_t integrated into w_t+1')
    
    # FedGKD-specific parameters
    parser.add_argument('--temperature', type=float, default=1, help='Temperature for KD in FedGKD')
    parser.add_argument('--alpha', type=float, default=0.1, help='Distillation coefficient (gamma) for FedGKD')
    parser.add_argument('--buffer_length', type=int, default=5, help='Number of historical models for FedGKD teacher')
    
    args = parser.parse_args()
    print("Arguments parsed")
    # Validate c
    if not (0 < args.c <= 1):
        raise ValueError("c must be between 0 and 1")
    if args.sim_n < 1:
        raise ValueError("sim_n must be at least 1")
    
    # Set random seeds for reproducibility
    torch.manual_seed(1)
    np.random.seed(1)
    random.seed(1)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(1)
    print("Seeds set")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device set: {device}")

    # Load data
    if args.dataset == 'mnist' and args.alpha_diric == 0.05 and args.n_clients == 20:
        print("Loading MNIST data N_client 20 alpha 0.05 from file")
        client_data = torch.load("data/MNIST_N20_a005.pt")
    elif args.dataset == 'mnist' and args.alpha_diric == 0.5 and args.n_clients == 20:
        print("Loading MNIST data N_client 20 alpha 0.5 from file")
        client_data = torch.load("data/MNIST_N20_a05.pt")
    elif args.dataset == 'mnist' and args.alpha_diric == 0.05 and args.n_clients == 50:
        print("Loading MNIST data N_client 50 alpha 0.05 from file")
        client_data = torch.load("data/MNIST_N50_a005.pt")
    elif args.dataset == 'mnist' and args.alpha_diric == 0.5 and args.n_clients == 50:
        print("Loading MNIST data N_client 50 alpha 0.5 from file")
        client_data = torch.load("data/MNIST_N50_a05.pt") 
    elif args.dataset == 'synthetic':
        print("Loading synthetic data from file")
        client_data = torch.load("data/synthetic_data.pt")
        if len(client_data) != args.n_clients:
            raise ValueError(f"Saved data has {len(client_data)} clients, but {args.n_clients} were requested")
    print("Data loaded")

    # Select model class based on dataset and model type
    if args.dataset == 'mnist':
        model_class = MLR if args.model == 'mlr' else MLP
    elif args.dataset == 'synthetic':
        model_class = SyntheticMLR if args.model == 'mlr' else SyntheticMLP

    # Run simulations
    print(f"\nRunning {args.algorithm.upper()} Simulations")
    all_accuracies = []
    all_losses = []
    for sim in range(args.sim_n):
        print(f"\nSimulation {sim + 1}/{args.sim_n}")
        if args.algorithm == 'pfedkd':
            server = pfedkdServer(client_data, model_class, device, args.local_epochs, args.batch_size, args.learning_rate, args.kd_weight, args.c)
        elif args.algorithm == 'fedavg':
            server = FedAvgServer(client_data, model_class, device, args.local_epochs, args.batch_size, args.learning_rate, args.c)
        elif args.algorithm == 'fedprox':
            server = FedProxServer(client_data, model_class, device, args.local_epochs, args.batch_size, args.learning_rate, args.mu, args.c)
        elif args.algorithm == 'perfedavg':
            server = PerFedAvgServer(client_data, model_class, device, args.local_epochs, args.batch_size, args.learning_rate, args.beta, args.c)
        elif args.algorithm == 'pfedme':
            server = pfedmeServer(client_data, model_class, device, args.local_epochs, args.batch_size, args.learning_rate, args.b, args.lamda, args.c, args.K, args.personal_learning_rate)
        elif args.algorithm == 'fedgkd':
            server = FedGKDServer(client_data, model_class, device, args.local_epochs, args.batch_size, args.learning_rate, args.temperature, args.alpha, args.buffer_length, args.c)
        # Train and get results (assuming train returns accuracies and losses)
        accuracies, losses = server.train(args.rounds)
        all_accuracies.append(accuracies)
        all_losses.append(losses)
    
    # Calculate averages across simulations
    avg_accuracies = np.mean(all_accuracies, axis=0)  # Average per round
    sd_accuracies = np.std(all_accuracies, axis=0)  # sd per round
    avg_losses = np.mean(all_losses, axis=0)  # Average per round
    sd_losses = np.std(all_losses, axis=0)  # sd per round

    # Save results to a file
    output_file = f"results/{args.algorithm}_{args.dataset}_{args.alpha_diric}_{args.n_clients}_{args.model}_{args.kd_weight}_lr{args.learning_rate}_results.npz"
    np.savez(output_file, 
         accuracies=avg_accuracies, 
         sd_accuracies=sd_accuracies, 
         losses=avg_losses, 
         sd_losses=sd_losses, 
         rounds=args.rounds)
    print(f"Saved results to {output_file}")
    print(f"\nFinal Results over {args.sim_n} Simulations:")
    for r in range(args.rounds):
        print(f"Round {r+1}: Avg Test Accuracy = {avg_accuracies[r]:.2f}%, Avg Test Loss = {avg_losses[r]:.4f}")
    print(f"Overall Avg Test Accuracy = {np.mean(avg_accuracies):.2f}%, Overall Avg Test Loss = {np.mean(avg_losses):.4f}")

if __name__ == "__main__":
    main()
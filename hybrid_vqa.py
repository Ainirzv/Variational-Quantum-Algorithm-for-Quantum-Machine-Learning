import torch
import torch.nn as nn
from torch.autograd import Function
import numpy as np
import pandas as pd
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.primitives import StatevectorEstimator
from qiskit.quantum_info import SparsePauliOp

#Prep Training Data (Breast Cancer Wisconsin)

#Load the data file
df = pd.read_csv(r"C:\Users\HP\Desktop\hackathon\FLIQ\wdbc.data", header=None)

#Assign column names
columns = ['id', 'diagnosis'] + [f'feature_{i}' for i in range(1, 31)]
df.columns = columns

#Drop the ID column
df = df.drop(columns=['id'])

#Encode diagnosis: M = 1, B = 0
df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})

#Convert to numpy arrays
X = df.drop(columns=['diagnosis']).values.astype(np.float32)
Y = df['diagnosis'].values.astype(np.float32).reshape(-1, 1)

#Normalize features manually (z-score)
mean = X.mean(axis=0)
std = X.std(axis=0)
X = (X - mean) / std

#Convert to torch tensors
X_tensor = torch.tensor(X, dtype=torch.float32)
Y_tensor = torch.tensor(Y, dtype=torch.float32)

#Manual train/test split (80/20)
num_samples = X_tensor.shape[0]
indices = torch.randperm(num_samples)

split_idx = int(num_samples * 0.8)
train_indices = indices[:split_idx]
test_indices = indices[split_idx:]

X_train = X_tensor[train_indices]
Y_train = Y_tensor[train_indices]
X_test = X_tensor[test_indices]
Y_test = Y_tensor[test_indices]

#VQA Circuit
n_qubits = 2
params = ParameterVector('theta', length=4)

def create_vqa_circuit(input_data, weights):
    qc = QuantumCircuit(n_qubits)

    # 1. Encode two classical features into rotations
    for i in range(n_qubits):          # ← loop over qubits
        qc.ry(float(input_data[i]), i) # feature encoding (angle encoding)

    # 2. Variational layer (4 trainable params)
    qc.rz(float(weights[0]), 0)
    qc.rz(float(weights[1]), 1)
    qc.cx(0, 1)
    qc.ry(float(weights[2]), 0)
    qc.ry(float(weights[3]), 1)

    return qc

#Qiskit StatevectorEstimator primitive
estimator = StatevectorEstimator()
observables = [SparsePauliOp("ZI")]

#PyTorch Custom Autograd Function For VQA Layer
class VQALayerFunction(Function):
    @staticmethod
    def forward(ctx, input_tensor, weights):
        input_vals = input_tensor.detach().numpy()
        weight_vals = weights.detach().numpy()
        ctx.save_for_backward(input_tensor, weights)

        qc = create_vqa_circuit(input_vals, weight_vals)
        job = estimator.run([(qc, observables)])
        expval = job.result()[0].data.evs

        return torch.tensor([expval], dtype=torch.float32)

    @staticmethod
    def backward(ctx, grad_output):
        input_tensor, weights = ctx.saved_tensors
        input_vals  = input_tensor.detach().numpy()
        weight_vals = weights.detach().numpy()
        shift = np.pi / 2
        grads = []

        # loop over **each trainable parameter**
        for k in range(len(weight_vals)):
            # θ₊ = θ_k + π/2, θ₋ = θ_k − π/2
            plus  = weight_vals.copy(); plus[k]  += shift
            minus = weight_vals.copy(); minus[k] -= shift

            qc_plus  = create_vqa_circuit(input_vals, plus)
            qc_minus = create_vqa_circuit(input_vals, minus)

            ev_plus  = estimator.run([(qc_plus,  observables)]).result()[0].data.evs
            ev_minus = estimator.run([(qc_minus, observables)]).result()[0].data.evs

            # ∂⟨Z⟩/∂θ_k  =  (ev₊ − ev₋)/2
            grads.append((ev_plus - ev_minus)/2)

        grads_tensor = torch.tensor(grads, dtype=torch.float32)

        # No gradient wrt input_tensor (return None for it)
        return None, grad_output.view(-1)[0] * grads_tensor.view(-1)



#Quantum Layer as PyTorch Module
class VQALayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(4)) #4 trainable parameters for the circuit

    def forward(self, x):
        return torch.stack([VQALayerFunction.apply(x[i], self.weights) for i in range(x.size(0))]).view(-1, 1)

#Full Hybrid Model
class HybridModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.classical = nn.Linear(X_tensor.shape[1], 2) #Classic preprocessing layer
        self.quantum = VQALayer() #VQA layer
        self.output = nn.Linear(1, 1) #Final classical layer

    def forward(self, x):
        x = self.classical(x)
        x = torch.tanh(x)  #Activation before quantum layer
        x = self.quantum(x)
        x = self.output(x)
        return torch.sigmoid(x)

model = HybridModel()
optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
loss_fn = nn.BCELoss()

#Training Loop
for epoch in range(50):
    optimizer.zero_grad()
    preds = model(X_train)
    loss = loss_fn(preds, Y_train)
    loss.backward()
    optimizer.step()
    with torch.no_grad():
        acc = ((preds > 0.5).float() == Y_train).float().mean()
        print(f"Epoch {epoch+1} | Loss: {loss.item():.4f} | Accuracy: {acc.item()*100:.2f}%")

#Implement Testing Loop:
with torch.no_grad():
    preds_test = model(X_test)
    test_acc = ((preds_test > 0.5).float() == Y_test).float().mean()
    print(f"Test accuracy: {test_acc.item()*100:.2f}%")


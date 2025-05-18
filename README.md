# Variational-Quantum-Algorithm-for-Quantum-Machine-Learning
\## Hybrid Quantum‑Classical Breast‑Cancer Classifier
*(PyTorch × Qiskit VQA demo)*

---

\### Overview
This project shows how to embed a **Variational Quantum Algorithm (VQA)** inside a PyTorch model to classify malignant (M) vs benign (B) breast‑tumour samples from the classic *Wisconsin Diagnostic Breast Cancer* dataset.

Pipeline

1. **Classical preprocessing** (linear layer → tanh)
2. **Quantum circuit** with 2 qubits + 4 trainable rotation angles
3. **Expectation value  ⟨Z I⟩** returned to PyTorch
4. **Final linear + sigmoid** for binary probability

Gradients of the quantum layer are computed with the **parameter‑shift rule**, so the whole network trains end‑to‑end with Adam.

---

\### Repository structure

```
.
├── data/
│   └── wdbc.data                 # raw UCI breast‑cancer file
├── hybrid_vqa.py                 # full training script (this file)
└── README.md                     # you are here
```

---

\### Requirements

| Library | Version tested |
| ------- | -------------- |
| python  | ≥ 3.9          |
| torch   | ≥ 2.1          |
| qiskit  | ≥ 1.1          |
| numpy   | ≥ 1.24         |
| pandas  | ≥ 2.0          |

> **Windows users:** install the **pre‑built wheel** `qiskit-aer` (`pip install qiskit-aer`) to avoid CMake/Visual Studio errors.

Create a fresh venv to avoid mixing **Qiskit < 1.0** and **≥ 1.0**:

```bash
python -m venv venv
venv\Scripts\activate     # Windows
# . venv/bin/activate     # Linux/macOS
pip install torch qiskit qiskit-aer pandas numpy
```

---

\### Running

```bash
python hybrid_vqa.py
```

Typical console output (truncated):

```
Epoch 48 | Loss: 0.5291 | Accuracy: 76.48%
Epoch 49 | Loss: 0.5285 | Accuracy: 76.70%
Epoch 50 | Loss: 0.5280 | Accuracy: 76.48%
Test accuracy: 74.56%
```

---

\### How it works

| Block                    | Purpose                                                                                                                     |
| ------------------------ | --------------------------------------------------------------------------------------------------------------------------- |
| **create\_vqa\_circuit** | Encodes two real numbers as `RY` rotations, applies a 4‑parameter entangling ansatz, returns a `QuantumCircuit`.            |
| **StatevectorEstimator** | Simulates the circuit and evaluates expectation value of `Z⊗I`.                                                             |
| **VQALayerFunction**     | Custom `torch.autograd.Function`; forward pass runs the circuit, backward pass uses **parameter‑shift** to compute ∂⟨Z⟩/∂θ. |
| **VQALayer (nn.Module)** | Wraps the function; stores the 4 trainable parameters.                                                                      |
| **HybridModel**          | Classical → quantum → classical chain.                                                                                      |

---

\### Modding ideas

* **Wider classical front‑end** – replace `Linear(30, 2)` with a multi‑layer MLP.
* **More qubits or layers** – expand `create_vqa_circuit` and `ParameterVector`.
* **Alternative observables** – measure `XZ`, `ZZ`, etc., and concatenate outputs.
* **Batch training & schedulers** – use `DataLoader`, learning‑rate decay.

---

\### Citing
If you build on this code for a hackathon or paper, please acknowledge:

> *Hybrid VQA breast‑cancer demo (2025), inspired by IBM Q & Torch hybrid tutorials.*

Happy hacking!

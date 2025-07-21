from qiskit import QuantumCircuit
from qiskit.primitives import Sampler
from qiskit_aer import Aer
from qiskit.visualization import plot_histogram
import matplotlib.pyplot as plt
import numpy as np


def sat_oracle(n_vars, output_qubit_idx):
    """Creates an oracle for a 3-variable SAT clause: (x1 ∨ ¬x2 ∨ x3)"""
    qc = QuantumCircuit(n_vars + 1)

    # Clause: (x1 ∨ ¬x2 ∨ x3) is false only when x1=0, x2=1, x3=0
    # So we flip these to make a control gate that flips output if input matches the UNSAT assignment
    qc.x(0)      # x1 == 0 -> apply X to make it 1
    qc.x(2)      # x3 == 0 -> apply X to make it 1
    # x2 == 1 -> leave as-is

    # Multi-controlled NOT using ancilla (if needed)
    qc.mcx([0, 1, 2], output_qubit_idx)

    # Undo the X gates
    qc.x(0)
    qc.x(2)

    return qc


def diffuser(n_qubits):
    """Diffuser (inversion about the mean) circuit"""
    qc = QuantumCircuit(n_qubits)
    qc.h(range(n_qubits))
    qc.x(range(n_qubits))
    qc.h(n_qubits - 1)
    qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
    qc.h(n_qubits - 1)
    qc.x(range(n_qubits))
    qc.h(range(n_qubits))
    return qc


def grover_sat():
    qc = QuantumCircuit(2)
    qc.h([0, 1])
    qc.cz(0, 1)
    qc.h([0, 1])
    qc.measure_all()

    sampler = Sampler()
    result = sampler.run([qc]).result()

    # Convert quasi-probabilities to counts
    shots = 1024  # assume standard shot count
    quasi_dists = result.quasi_dists[0]
    counts = {k: int(v * shots) for k, v in quasi_dists.items()}

    print("Resulting counts:", counts)
    plot_histogram(counts)
    plt.show()


if __name__ == "__main__":
    grover_sat()

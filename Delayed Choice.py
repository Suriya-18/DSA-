from qiskit import QuantumCircuit, Aer, execute

qc = QuantumCircuit(2, 2)
qc.h(0)                 # Photon qubit
qc.h(1)                 # Control qubit
qc.cx(1, 0)             # If control is 1, apply H
qc.h(0)
qc.measure([0, 1], [0, 1])

sim = Aer.get_backend('qasm_simulator')
result = execute(qc, backend=sim, shots=1024).result()
counts = result.get_counts()
print(counts)

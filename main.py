from qiskit import QuantumCircuit, transpile
from qiskit_aer import Aer
from qiskit.visualization import plot_histogram
import matplotlib.pyplot as plt
import numpy as np

def parse_cnf_formula(formula_str):
    """
    Parse CNF formula from user input.
    Format: (x0 OR ~x1) AND (~x0 OR x2) AND (x1 OR x2)
    Returns: list of clauses, each clause is list of (var_index, is_positive)
    """
    formula_str = formula_str.replace(' ', '').upper()
    
    # Split by AND
    clauses_str = formula_str.split('AND')
    formula = []
    
    for clause_str in clauses_str:
        # Remove parentheses
        clause_str = clause_str.strip('()')
        
        # Split by OR
        literals_str = clause_str.split('OR')
        clause = []
        
        for literal_str in literals_str:
            literal_str = literal_str.strip()
            
            # Check if negated
            if literal_str.startswith('~'):
                is_positive = False
                var_name = literal_str[1:]
            else:
                is_positive = True
                var_name = literal_str
            
            # Extract variable index (assume format like x0, x1, etc.)
            if var_name.startswith('X'):
                var_index = int(var_name[1:])
            else:
                raise ValueError(f"Invalid variable format: {var_name}")
            
            clause.append((var_index, is_positive))
        
        formula.append(clause)
    
    return formula

def get_num_variables(formula):
    """Extract the number of unique variables from the formula"""
    variables = set()
    for clause in formula:
        for var_index, _ in clause:
            variables.add(var_index)
    return len(variables)

def clause_satisfied_circuit(qc, clause, var_qubits, ancilla_qubit):
    """
    Marks ancilla_qubit = 1 iff the clause is SATISFIED.
    """
    # For a clause to be satisfied, at least one literal must be true
    # We'll use the approach: satisfied = OR of all literals
    
    # Step 1: Prepare literals (flip variables that should be negated)
    for var_index, is_positive in clause:
        if not is_positive:  # If literal is negated
            qc.x(var_qubits[var_index])
    
    # Step 2: Compute OR using De Morgan's law: OR = NOT(AND(NOT literals))
    # First, flip all literal qubits
    literal_qubits = [var_qubits[var_index] for var_index, _ in clause]
    for qubit in literal_qubits:
        qc.x(qubit)
    
    # Multi-controlled X (AND of negated literals)
    qc.mcx(literal_qubits, ancilla_qubit)
    
    # Flip the result to get OR
    qc.x(ancilla_qubit)
    
    # Step 3: Uncompute the literal flips
    for qubit in literal_qubits:
        qc.x(qubit)
    
    # Step 4: Restore original variable states
    for var_index, is_positive in clause:
        if not is_positive:
            qc.x(var_qubits[var_index])

def oracle_sat(qc, formula, var_qubits, ancilla_qubits, flag_qubit, output_qubit):
    """
    Oracle that flips phase if ALL clauses are satisfied (valid SAT solution).
    """
    # Mark each clause as satisfied
    for i, clause in enumerate(formula):
        clause_satisfied_circuit(qc, clause, var_qubits, ancilla_qubits[i])
    
    # Compute AND of all clause satisfactions
    qc.mcx(ancilla_qubits, flag_qubit)
    
    # Phase flip if all clauses satisfied
    qc.cz(flag_qubit, output_qubit)
    
    # Uncompute
    qc.mcx(ancilla_qubits, flag_qubit)
    
    # Uncompute clause satisfactions
    for i, clause in reversed(list(enumerate(formula))):
        clause_satisfied_circuit(qc, clause, var_qubits, ancilla_qubits[i])

def diffuser(qc, qubits):
    """Grover diffuser (inversion about average)"""
    # Apply Hadamard to all qubits
    qc.h(qubits)
    
    # Flip all qubits
    qc.x(qubits)
    
    # Multi-controlled Z gate (phase flip on |111...1>)
    if len(qubits) > 1:
        qc.h(qubits[-1])
        qc.mcx(qubits[:-1], qubits[-1])
        qc.h(qubits[-1])
    else:
        qc.z(qubits[0])
    
    # Flip back
    qc.x(qubits)
    
    # Apply Hadamard again
    qc.h(qubits)

def calculate_iterations(num_vars, expected_solutions):
    """Calculate optimal number of Grover iterations"""
    N = 2 ** num_vars
    if expected_solutions == 0:
        return 1
    
    theta = np.arcsin(np.sqrt(expected_solutions / N))
    optimal_iterations = int(np.pi / (4 * theta) - 0.5)
    return max(1, optimal_iterations)

def grover_sat_solver(formula, num_vars, iterations=None):
    """
    Grover's algorithm for SAT solving
    """
    num_clauses = len(formula)
    
    # Calculate iterations if not provided
    if iterations is None:
        # Estimate: assume roughly 1/8 of assignments satisfy (heuristic)
        expected_solutions = max(1, 2**(num_vars-3))
        iterations = calculate_iterations(num_vars, expected_solutions)
    
    print(f"Using {iterations} Grover iterations for {num_vars} variables")
    
    # Create quantum circuit
    # qubits: variables + clause_ancillas + flag + output
    total_qubits = num_vars + num_clauses + 2
    qc = QuantumCircuit(total_qubits, num_vars)
    
    # Qubit assignments
    var_qubits = list(range(num_vars))
    ancilla_qubits = list(range(num_vars, num_vars + num_clauses))
    flag_qubit = num_vars + num_clauses
    output_qubit = num_vars + num_clauses + 1
    
    # Initialize variable qubits in equal superposition
    qc.h(var_qubits)
    
    # Prepare output qubit in |-> state
    qc.x(output_qubit)
    qc.h(output_qubit)
    
    # Apply Grover iterations
    for i in range(iterations):
        # Oracle: mark satisfying assignments
        oracle_sat(qc, formula, var_qubits, ancilla_qubits, flag_qubit, output_qubit)
        
        # Diffuser: amplify marked states
        diffuser(qc, var_qubits)
    
    # Measure variable qubits
    qc.measure(var_qubits, range(num_vars))
    
    return qc

def verify_solution(assignment, formula):
    """Verify if an assignment satisfies the formula"""
    for clause in formula:
        clause_satisfied = False
        for var_index, is_positive in clause:
            var_value = int(assignment[-(var_index+1)])  # Reverse indexing
            literal_value = var_value if is_positive else (1 - var_value)
            if literal_value:
                clause_satisfied = True
                break
        if not clause_satisfied:
            return False
    return True

def main():
    print("Grover SAT Solver")
    print("================")
    print("Enter your CNF formula using the format:")
    print("(x0 OR ~x1) AND (~x0 OR x2) AND (x1 OR x2)")
    print("Use ~ for negation, variables as x0, x1, x2, etc.")
    print()
    
    # Get user input
    formula_str = input("Enter CNF formula: ").strip()
    
    if not formula_str:
        # Use default example
        formula_str = "(x0 OR ~x1) AND (~x0 OR x2)"
        print(f"Using example formula: {formula_str}")
    
    try:
        # Parse the formula
        formula = parse_cnf_formula(formula_str)
        num_vars = get_num_variables(formula)
        
        print(f"\nParsed formula with {num_vars} variables and {len(formula)} clauses:")
        for i, clause in enumerate(formula):
            clause_str = " OR ".join([f"{'~' if not pos else ''}x{var}" for var, pos in clause])
            print(f"  Clause {i+1}: ({clause_str})")
        
        # Ask for number of iterations
        iterations_input = input(f"\nEnter number of iterations (press Enter for auto): ").strip()
        iterations = int(iterations_input) if iterations_input else None
        
        # Create and run quantum circuit
        qc = grover_sat_solver(formula, num_vars, iterations)
        
        # Simulate
        backend = Aer.get_backend('aer_simulator')
        qc_transpiled = transpile(qc, backend)
        job = backend.run(qc_transpiled, shots=1024)
        result = job.result()
        counts = result.get_counts()
        
        print("\nQuantum Circuit:")
        print("================")
        print(qc.draw(output='text'))

        
        print(f"\nMeasurement Results:")
        print("===================")
        
        # Sort results by count
        sorted_results = sorted(counts.items(), key=lambda x: x[1], reverse=True)
        
        satisfying_assignments = []
        for assignment, count in sorted_results:
            is_satisfying = verify_solution(assignment, formula)
            status = "✓ SAT" if is_satisfying else "✗ UNSAT"
            print(f"  {assignment}: {count} shots - {status}")
            
            if is_satisfying:
                satisfying_assignments.append(assignment)
        
        if satisfying_assignments:
            print(f"\nFound {len(satisfying_assignments)} satisfying assignment(s)!")
            print("Most frequent satisfying assignment:", sorted_results[0][0])
        else:
            print("\nNo satisfying assignments found in results.")
            print("The formula might be unsatisfiable, or more iterations may be needed.")
        
        # Plot histogram
        plot_histogram(counts, title="Grover SAT Solver Results")
        plt.show()
        
    except Exception as e:
        print(f"Error: {e}")
        print("Please check your formula format.")

if __name__ == "__main__":
    main()
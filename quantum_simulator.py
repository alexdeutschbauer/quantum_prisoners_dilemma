from itertools import product
import numpy as np

class QuantumSimulator:

    def __init__(self) -> None:
        self.zero = np.array([1, 0])
        self.one = np.array([0, 1])

        self.plus = (self.zero + self.one) / np.sqrt(2)
        self.minus = (self.zero - self.one) / np.sqrt(2)
        
        self.I = np.eye(2)
        self.Z = np.array([[1, 0], [0, -1]])
        self.X = np.array([[0, 1], [1, 0]])
        self.H = np.array([[1, 1], [1, -1]]) / np.sqrt(2)

        self.i_plus = (self.zero + 1j * self.one) / np.sqrt(2)
        self.i_minus = (self.zero - 1j * self.one) / np.sqrt(2)

        self.Y = np.array([[0, -1j], [1j, 0]])
        self.S = np.array([[1, 0], [0, 1j]])
        self.T = np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]])

        self.CNOT_01 = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])
        self.CNOT_10 = np.array([[1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0]])

        self.bell_state_00 = self.CNOT_01 @ self.combine_gates(self.H, self.I) @ self.combine_states(self.zero, self.zero)
        self.bell_state_01 = self.CNOT_01 @ self.combine_gates(self.H, self.I) @ self.combine_states(self.zero, self.one)
        self.bell_state_10 = self.CNOT_01 @ self.combine_gates(self.H, self.I) @ self.combine_states(self.one, self.zero)
        self.bell_state_11 = self.CNOT_01 @ self.combine_gates(self.H, self.I) @ self.combine_states(self.one, self.one)

        self.Bell_Measurement_Gate = self.combine_gates(self.H, self.I) @ self.CNOT_01

        self.TOFFOLI = np.array([
            [1, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 1, 0]
        ])

    def is_normalized(self, state) -> bool:
        return np.isclose(np.linalg.norm(state), 1)

    def probability_zero(self, state) -> float:
        return np.abs(state[0])**2
    
    def probability_one(self, state) -> float:
        return 1 - self.probability_zero(state)
    
    def apply_gate(self, gate, state) -> np.ndarray:
        return np.dot(gate, state)
    
    def is_inverse(self, gate) -> bool:
        return np.allclose(np.dot(gate, gate), self.I)
    
    def is_valid(self, gate) -> bool:
        return np.allclose(np.dot(gate, gate.T.conj()), self.I)
    
    def composite_gate(self, gates) -> np.ndarray:
        final_gate = np.eye(gates[-1].shape[0])
        for gate in gates[::-1]:
            final_gate = gate @ final_gate
        return final_gate

    def rotation_gate_x(self, theta) -> np.ndarray:
        return np.array([[np.cos(theta / 2), -1j * np.sin(theta / 2)], [-1j * np.sin(theta / 2), np.cos(theta / 2)]])
    
    def rotation_gate_y(self, theta) -> np.ndarray:
        return np.array([[np.cos(theta / 2), -np.sin(theta / 2)], [np.sin(theta / 2), np.cos(theta / 2)]])
    
    def rotation_gate_z(self, theta) -> np.ndarray:
        return np.array([[np.exp(-1j * theta / 2), 0], [0, np.exp(1j * theta / 2)]])
                        
    def get_rotation_by_axis(self, axis, theta) -> np.ndarray:
        match axis:
            case "x":
                return self.rotation_gate_x(theta)
            case "y":
                return self.rotation_gate_y(theta)
            case "z":
                return self.rotation_gate_z(theta)
            case _:
                raise ValueError("Invalid axis")
    
    def are_commuting(self, A, B) -> bool:
        return np.allclose(np.dot(A, B), np.dot(B, A))
    
    def kronecker(self, A, B) -> np.ndarray:
        # without using np.kron
        return np.vstack([np.hstack([A[i, j] * B for j in range(B.shape[1])]) for i in range(A.shape[0])])
    
    def combine_states(self, state1, state2) -> np.ndarray:
        return np.kron(state1, state2)
    
    def combine_gates(self, gate1, gate2) -> np.ndarray:
        return np.kron(gate1, gate2)
    
    def two_qubit_readout(self, state, bit) -> float:
        match bit:
            case '00':
                return np.abs(state[0])**2
            case '01':
                return np.abs(state[1])**2
            case '10':
                return np.abs(state[2])**2
            case '11':
                return np.abs(state[3])**2   

    def calculate_measurement_probabilitites(self, state):
        return np.array([self.two_qubit_readout(state, '00'), self.two_qubit_readout(state, '01'), self.two_qubit_readout(state, '10'), self.two_qubit_readout(state, '11')]) 

    def to_dual(self, ket):
        return np.array(ket).flatten().conj()
    
    def to_ket(self, bra):
        return np.array(bra)[...,None].conj()    

    def check_if_orthonormal(self, vector_list):
        for count1, v1 in enumerate(vector_list):
            for count2, v2 in enumerate(vector_list):
                overlap = self.to_dual(v1) @ v2
                if np.isclose(overlap, 0) and count1 != count2:
                    continue
                elif np.isclose(overlap, 1) and count1 == count2:
                    continue
                else:
                    return False
        return True               
    
    def init_entangled_state(self):
        return self.bell_state_00

    def message_encoding(self, initial_state, sending_bits:str):
        print('flipped')
        gate = self.composite_gate([np.linalg.matrix_power(self.X, int(sending_bits[1])), np.linalg.matrix_power(self.Z, int(sending_bits[0]))])
        return self.combine_gates(gate, self.I) @ initial_state

    def reverse_entanglement(self, received_state): #1
        return self.Bell_Measurement_Gate @ received_state

    def readout(self, final_qstate): #2
        return self.calculate_measurement_probabilitites(final_qstate)

    def decode(self, probs): #3
        mask_outcome = np.isclose(probs, 1)
        ALL_2B_OUTCOMES = np.array([['00'], ['01'], ['10'], ['11']])
        return ALL_2B_OUTCOMES[mask_outcome]

    def compute_2q_marginal_prob_for_o0(self, phi):
        return np.sum((np.abs(phi[0]) ** 2, np.abs(phi[1]) ** 2))

    def single_shot_from_p0(self, p_0):
        rng = np.random.default_rng()
        r = rng.uniform(0, 1)
        return 0 if r <= p_0 else 1

    def collaps_2q_state(self, phi, o_0, p_0):
        if o_0 == 0:
            return np.kron(self.to_dual(self.zero), self.I) @ phi / np.sqrt(p_0)
        else:
            return np.kron(self.to_dual(self.one), self.I) @ phi / np.sqrt(1 - p_0)

    def t_teleportation_init(self, psi):
        g1 = self.combine_gates(self.H, self.I)
        g2 = self.combine_gates(self.T, self.I)
        g3 = self.CNOT_10
        g4 = self.CNOT_01
        return g4 @ g3 @ g2 @ g1 @ self.combine_gates(self.zero, psi)
    
    def t_teleportation(self, psi):
        phi = self.t_teleportation_init(psi) 
        p_0 = self.compute_2q_marginal_prob_for_o0(phi)
        o_0 = self.single_shot_from_p0(p_0)
        phi_collapsed = self.collaps_2q_state(phi, o_0, p_0)
        return np.linalg.matrix_power(self.X @ self.S, o_0) @ phi_collapsed

    def quantum_teleportation_init(self, psi):
        layers = [ # Create list of gates of the teleportation algorithm.
            [self.I, self.H, self.I], 
            [self.I, self.CNOT_01], 
            [self.CNOT_01, self.I],
            [self.H, self.I, self.I]
        ]

        middle_gates = np.eye(2 ** 3, dtype=np.complex128)
        for layer in layers[::-1]:
            middle_gates @= self.multikron(*layer)
        return middle_gates @ self.multikron(psi, self.zero, self.zero)
    
    def quantum_teleportation_init_nobobaction(self, psi):
        layers = [ # Create list of gates of the teleportation algorithm.
            [self.CNOT_01, self.I], 
            [self.H, self.I, self.I], 
            [self.I, self.I, self.I],
        ]

        middle_gates = np.eye(2 ** 3, dtype=np.complex128)
        for layer in layers[::-1]:
            middle_gates @= self.multikron(*layer)
        return middle_gates @ self.multikron(psi, self.zero, self.zero)
    
    def quantum_teleportation_init_xbobaction(self, psi):
        layers = [ # Create list of gates of the teleportation algorithm.
            [self.CNOT_01, self.I], 
            [self.H, self.I, self.I], 
            [self.I, self.I, self.X],
        ]

        middle_gates = np.eye(2 ** 3, dtype=np.complex128)
        for layer in layers[::-1]:
            middle_gates @= self.multikron(*layer)
        return middle_gates @ self.multikron(psi, self.zero, self.zero)
    
    def quantum_teleportation_init_zxbobaction(self, psi):
        layers = [ # Create list of gates of the teleportation algorithm.
            [self.CNOT_01, self.I], 
            [self.H, self.I, self.I], 
            [self.I, self.I, self.Z],
            [self.I, self.I, self.X],
        ]

        middle_gates = np.eye(2 ** 3, dtype=np.complex128)
        for layer in layers[::-1]:
            middle_gates @= self.multikron(*layer)
        return middle_gates @ self.multikron(psi, self.zero, self.zero) 

    def quantum_teleportation(self, psi):
        # needs psi as an input.
        phi = self.quantum_teleportation_init(psi)
        # Do the first partial measurement
        p0_1 = self.compute_marginal_prob0(phi, 1)
        o1 = self.single_shot_from_p0(p0_1)
        phi_1 = self.collaps_state(phi, o1, p0_1, 1)
        # Do the second partial measurement.
        p0_0 = self.compute_marginal_prob0(phi, 0)
        o0 = self.single_shot_from_p0(p0_0)
        phi_2 = self.collaps_state(phi_1, o0, p0_0, 0)
        # Apply the gates depending on the outcomes of Alice.
        Bobs_gate = np.linalg.matrix_power(self.Z, o0) @ np.linalg.matrix_power(self.X, o1)
        return Bobs_gate @ phi_2 
    
    def quantum_teleportation_nobobaction(self, psi):
        # needs psi as an input.
        phi = self.quantum_teleportation_init_nobobaction(psi)
        # Do the first partial measurement
        p0_1 = self.compute_marginal_prob0(phi, 1)
        o1 = self.single_shot_from_p0(p0_1)
        phi_1 = self.collaps_state(phi, o1, p0_1, 1)
        # Do the second partial measurement.
        p0_0 = self.compute_marginal_prob0(phi, 0)
        o0 = self.single_shot_from_p0(p0_0)
        phi_2 = self.collaps_state(phi_1, o0, p0_0, 0)
        return phi_2
    
    def quantum_teleportation_xbobaction(self, psi):
        # needs psi as an input.
        phi = self.quantum_teleportation_init_xbobaction(psi)
        # Do the first partial measurement
        p0_1 = self.compute_marginal_prob0(phi, 1)
        o1 = self.single_shot_from_p0(p0_1)
        phi_1 = self.collaps_state(phi, o1, p0_1, 1)
        # Do the second partial measurement.
        p0_0 = self.compute_marginal_prob0(phi, 0)
        o0 = self.single_shot_from_p0(p0_0)
        phi_2 = self.collaps_state(phi_1, o0, p0_0, 0)
        return phi_2
    
    def quantum_teleportation_zxbobaction(self, psi):
        # needs psi as an input.
        phi = self.quantum_teleportation_init_zxbobaction(psi)
        # Do the first partial measurement
        p0_1 = self.compute_marginal_prob0(phi, 1)
        o1 = self.single_shot_from_p0(p0_1)
        phi_1 = self.collaps_state(phi, o1, p0_1, 1)
        # Do the second partial measurement.
        p0_0 = self.compute_marginal_prob0(phi, 0)
        o0 = self.single_shot_from_p0(p0_0)
        phi_2 = self.collaps_state(phi_1, o0, p0_0, 0)
        return phi_2
    
    ### GENERALIZATION ###

    def pad_1q_gates_Is(self, nqubits, list_single_q_gates, act_on_qubits) -> list:
        identities_padded = np.array([self.I] * nqubits, dtype=np.complex128)
        identities_padded[act_on_qubits] = np.array(list_single_q_gates)
        return identities_padded

    def calculate_measurement_probabilitites_generalized(psi):
        return np.abs(psi) ** 2
    
    def multikron(self, *args): 
        result = 1 + 0j
        for array in args[0]:
            result = np.kron(result, array)
        return result
    
    def compute_marginal_prob0(self, state, which_q):
        num_qubits = int(np.log2(len(state)))
        numbers_bin = [format(n, f'0{num_qubits}b') for n in range(len(state))]
        mask_array = np.array([bin_str[which_q] == '0' for bin_str in numbers_bin])
        outcome_probs = np.abs(state) ** 2
        return np.sum(outcome_probs[mask_array])

    def collaps_state(self, phi, o_0, p_0, which_q):
        num_qubits = int(np.log2(len(phi)))
        basis_state = self.to_dual(self.zero) if o_0 == 0 else self.to_dual(self.one)
        list_to_collaps = [basis_state if k == which_q else self.I for k in range(num_qubits)]
        matrix_to_collaps = self.multikron(list_to_collaps)
        return matrix_to_collaps @ phi / np.sqrt(p_0)
    
    def cnot(self, nqubits, qubits_list):
        M_00 = np.array([[1, 0], [0, 0]])
        M_11 = np.array([[0, 0], [0, 1]])
        LIST_LAYERS_CNOT = [[M_00, self.I], [M_11, self.X]]
        padded_layers = [self.pad_1q_gates_Is(nqubits, gate_list, qubits_list) for gate_list in LIST_LAYERS_CNOT]
        return np.sum([self.multikron(l) for l in padded_layers], axis=0)

    def toffoli(self, nqubits, qubits_list):
        M_00 = np.array([[1, 0], [0, 0]])
        M_11 = np.array([[0, 0], [0, 1]])
        LIST_LAYERS_TOFFOLI = [[M_00, M_00, self.I], [M_00, M_11, self.I], [M_11, M_00, self.I], [M_11, M_11, self.X]]
        padded_layers = [self.pad_1q_gates_Is(nqubits, gate_list, qubits_list) for gate_list in LIST_LAYERS_TOFFOLI]
        return np.sum([self.multikron(l) for l in padded_layers], axis=0)
    
    def AND(self, nqubits, qubit0, qubit1, qubit_store):
        return self.toffoli(nqubits, [qubit0, qubit1, qubit_store])

    def OR(self, nqubits, qubit0, qubit1, qubit_store):
        Xs_list_before = self.pad_1q_gates_Is(nqubits, [self.X, self.X, self.X], [qubit0, qubit1, qubit_store])
        Xs_dense_before = self.multikron(Xs_list_before)
        dense_Toffoli = self.toffoli(nqubits, [qubit0, qubit1, qubit_store])
        Xs_list_after = self.pad_1q_gates_Is(nqubits, [self.X, self.X], [qubit0, qubit1])
        Xs_dense_after = self.multikron(Xs_list_after)
        return Xs_dense_after @ dense_Toffoli @ Xs_dense_before
    
    def bitstr_representation(self, nqubits:int):
        return [format(i, f'0{nqubits}b') for i in range(2 ** nqubits)]
    
    def collaps_state_multi(self, state:np.ndarray, outcome_list:list, prob_list:list, qubit_list:list):
        inds = np.argsort(qubit_list)[::-1] # make sure to collapse 'from the back'
        for o, p, q in zip(np.array(outcome_list)[inds], np.array(prob_list)[inds], np.array(qubit_list)[inds]):
            state = self.collaps_state(state, o, p, q)
        return state

    def controlled_X_nqubits(self, nqubits:int, control_on_list:list, target_qubit:int):
        M_00 = np.array([[1, 0], [0, 0]]) # |0><0|
        M_11 = np.array([[0, 0], [0, 1]]) # |1><1|
        Ms_array = np.array([M_00, M_11]) # for convenience. 
        qubits_list = control_on_list + [target_qubit]
        combinations = list(product([0,1], repeat = (len(qubits_list) - 1)))[:-1] # all 2 ** nqubit combinations EXEPCT of all 1s.
        list_M_00s_M_11s = [np.append(Ms_array[list(c)], [self.I], axis=0) for c in combinations]
        list_M_00s_M_11s.append(np.append(Ms_array[[1] * (len(qubits_list) - 1)], [self.X], axis=0)) 
        padded_layers = [self.pad_1q_gates_Is(nqubits, gate_list, qubits_list) for gate_list in list_M_00s_M_11s]
        return np.real(np.sum([self.multikron(l) for l in padded_layers], axis=0))
    
    def XORs(self, nqubits:int, control_on_list:list, target_qubit:int):
        final_gate = np.eye(2 ** nqubits)
        for current_q in control_on_list:
            final_gate = self.cnot(nqubits, [current_q, target_qubit]) @ final_gate
        return final_gate
    
    def diffusion_operator(self, n:int, sf:int):
        ntot = n + sf + 1
        qlist_n = [i for i in range(n)]
        right_layer = self.multikron(self.pad_1q_gates_Is(ntot, [self.X @ self.H] * n + [self.H @ self.X], qlist_n + [ntot - 1]))
        controll_on_n = self.controlled_X_nqubits(ntot, qlist_n, ntot - 1)
        left_layer = self.multikron(self.pad_1q_gates_Is(ntot, [self.H @ self.X] * n + [self.X @ self.H], qlist_n + [ntot - 1]))
        return  left_layer @ controll_on_n @ right_layer
    
    def compute_amplification_steps(self, ntot_inputs:int, npositive_answers:int):
        rf = npositive_answers / ntot_inputs
        return int(np.round(np.pi / ( 4 * np.sqrt(rf))))
    
    def build_oracle_exact_cover(self, clauses_list:list):
        n = np.max(clauses_list) + 1 
        qlist_sf = [i + n for i in range(len(clauses_list))] 
        nqubits = n + len(clauses_list) + 1
        final_oracle = np.eye(2 ** nqubits, dtype=np.complex128)
        # The boolean XOR and AND realisation of clauses.
        for count, clause in enumerate(clauses_list):
            final_oracle = self.XORs(nqubits, clause, count + n ) @ final_oracle
            final_oracle = self.controlled_X_nqubits(nqubits, clause, count + n ) @ final_oracle # Like big AND
        # Mark the solutions with a -1.
        final_oracle = self.multikron(self.pad_1q_gates_Is(nqubits, [self.H @ self.X], [nqubits - 1])) @ final_oracle
        final_oracle = self.controlled_X_nqubits(nqubits, qlist_sf, nqubits - 1) @ final_oracle
        final_oracle = self.multikron(self.pad_1q_gates_Is(nqubits, [self.X @ self.H], [nqubits - 1])) @ final_oracle
        # Reverse the boolean operations (by applying them again).
        for count, clause in enumerate(clauses_list):
            final_oracle = self.XORs(nqubits, clause, count + n ) @ final_oracle
            final_oracle = self.controlled_X_nqubits(nqubits, clause, count + n ) @ final_oracle
        
        return final_oracle
from quantum_simulator import QuantumSimulator
import numpy as np
import streamlit as st


class PrisonersDilemma:
    '''
    Run app using the following command:
    streamlit run prisoners_dilemma.py
    '''
    def __init__(self):
        self.sim = QuantumSimulator()
        self.sentence = ['Alice and Bob both get 1 year.', 'Alice gets 5 years and Bob goes free.', 'Alice goes free and Bob gets 5 years.', 'Alice and Bob both get 3 years.']
        self.quantum_11 = (1/np.sqrt(2)) * (np.array([1, 0, 0, 0]) + 1.j * np.array([0, 0, 0, 1]))
        self.quantum_50 = (1/np.sqrt(2)) * (np.array([0, 1, 0, 0]) - 1.j * np.array([0, 0, 1, 0]))
        self.quantum_05 = (1/np.sqrt(2)) * (np.array([0, 0, 1, 0]) - 1.j * np.array([0, 1, 0, 0]))
        self.quantum_33 = (1/np.sqrt(2)) * (1.j * np.array([1, 0, 0, 0]) + np.array([0, 0, 0, 1]))
        self.final_states = np.array([self.quantum_11, self.quantum_50, self.quantum_05, self.quantum_33])
    
    def classical(self, alice_deflects, bob_deflects):
        init_state = self.sim.multikron([self.sim.zero] * 2)
        alice_gate = self.sim.X if alice_deflects else self.sim.I
        bob_gate = self.sim.X if bob_deflects else self.sim.I
        combined_gate = self.sim.combine_gates(alice_gate, bob_gate)
        final_state =  self.sim.apply_gate(combined_gate, init_state)
        probs = self.sim.calculate_measurement_probabilitites(final_state)
        return self.sentence[np.argmax(probs)]
    
    def quantum(self, alice_deflects, bob_deflects):
        init_state = self.quantum_11
        alice_gate = -1.j * self.sim.Y if alice_deflects else self.sim.I
        bob_gate = -1.j * self.sim.Y if bob_deflects else self.sim.I
        combined_gate = self.sim.combine_gates(alice_gate, bob_gate)
        final_state =  self.sim.apply_gate(combined_gate, init_state)
        for index, state in enumerate(self.final_states):
            if np.allclose(state, final_state):
                break
        return self.sentence[index]

    def magic(self, bob_deflects):
        init_state = self.quantum_11
        alice_gate = self.sim.composite_gate([self.sim.S, self.sim.H, self.sim.Y, self.sim.S, self.sim.Z])
        bob_gate = -1.j * self.sim.Y if bob_deflects else self.sim.I
        combined_gate = self.sim.combine_gates(alice_gate, bob_gate)
        final_state =  self.sim.apply_gate(combined_gate, init_state)
        if np.allclose(final_state, (1/np.sqrt(2)) * (self.quantum_33 + self.quantum_05)) or np.allclose(final_state, (1/np.sqrt(2)) * (self.quantum_33 - self.quantum_05)):
            return np.random.choice([self.sentence[2], self.sentence[3]])

    def run_app(self):
        st.title('Quantum Prisoner\'s Dilemma')
        col1, col2 = st.columns([0.6, 0.4])
        with col1:
            st.image('prisoners_dilemma.webp', use_container_width=True)
        with col2:
            technology = st.pills(r'$\textsf{Used Technology}$', ['classical', 'quantum'], default='classical')
            if technology == 'classical':
                alice = st.radio(r'$\textsf{Alice}$', ['confess', 'deflect'], index=0)
                bob = st.radio(r'$\textsf{Bob}$', ['confess', 'deflect'], index=0)
            if technology == 'quantum':
                magic = st.toggle('Magic Gate')
                if magic:
                    bob = st.radio(r'$\textsf{Bob}$', ['confess', 'deflect'], index=0)
                else:
                    alice = st.radio(r'$\textsf{Alice}$', ['confess', 'deflect'], index=0)
                    bob = st.radio(r'$\textsf{Bob}$', ['confess', 'deflect'], index=0)
            if st.button('Run'):
                if technology == 'classical':
                    st.write(self.classical(alice == 'deflect', bob == 'deflect'))
                if technology == 'quantum':
                    if magic:
                        st.write(self.magic(bob == 'deflect'))
                    else:
                        st.write(self.quantum(alice == 'deflect', bob == 'deflect'))

if __name__ == "__main__":
    pd = PrisonersDilemma()
    pd.run_app()

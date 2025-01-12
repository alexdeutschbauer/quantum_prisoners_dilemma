from quantum_simulator import QuantumSimulator
import numpy as np
import streamlit as st
from scipy.linalg import expm


class PrisonersDilemma:
    '''
    Run app using the following command:
    streamlit run prisoners_dilemma.py
    '''
    def __init__(self):
        self.sim = QuantumSimulator()
        self.sentence = ['Alice and Bob both get 1 year.', 'Alice gets 5 years and Bob goes free.', 'Alice goes free and Bob gets 5 years.', 'Alice and Bob both get 3 years.']
        self.C = self.sim.I
        self.D = np.array([[0, 1], 
                    [-1, 0]])
        self.Q = np.array([[1.j, 0], [0, -1.j]])
        
    def get_probs(self, alice_gate, bob_gate, init_state, J=np.eye(4)):
        combined_gate = self.sim.combine_gates(alice_gate, bob_gate)
        inter_state = self.sim.apply_gate(J, init_state)
        inter_state =  self.sim.apply_gate(combined_gate, inter_state)
        final_state = self.sim.apply_gate(J.T.conj(), inter_state)
        return self.sim.calculate_measurement_probabilitites(final_state)
    
    def classical(self, alice_deflects, bob_deflects):
        init_state = self.sim.multikron([self.sim.zero] * 2)
        alice_gate = self.D if alice_deflects else self.C
        bob_gate = self.D if bob_deflects else self.C
        probs = self.get_probs(alice_gate, bob_gate, init_state)
        idx = np.random.choice(range(4), p=probs)
        return self.sentence[idx], probs
    
    def quantum(self, alice_deflects, bob_deflects, gamma):
        init_state = self.sim.multikron([self.sim.zero] * 2)
        J = expm(np.kron(-1.j * gamma * self.D, self.D / 2))
        alice_gate = self.D if alice_deflects else self.C
        bob_gate = self.D if bob_deflects else self.C
        probs = self.get_probs(alice_gate, bob_gate, init_state, J)
        idx = np.random.choice(range(4), p=probs)
        return self.sentence[idx], probs

    def alice_m(self, bob_deflects, gamma):
        init_state = self.sim.multikron([self.sim.zero] * 2)
        J = expm(np.kron(-1.j * gamma * self.D, self.D / 2))
        alice_gate = self.sim.composite_gate([self.sim.S, self.sim.H, self.sim.Y, self.sim.S, self.sim.Z])
        bob_gate = -1.j * self.sim.Y if bob_deflects else self.sim.I
        probs = self.get_probs(alice_gate, bob_gate, init_state, J)
        idx = np.random.choice(range(4), p=probs)
        return self.sentence[idx], probs
    
    def both_q(self, gamma):
        init_state = self.sim.multikron([self.sim.zero] * 2)
        J = expm(np.kron(-1.j * gamma * self.D, self.D / 2))
        alice_gate = self.Q
        bob_gate = self.Q
        probs = self.get_probs(alice_gate, bob_gate, init_state, J)
        idx = np.random.choice(range(4), p=probs)
        return self.sentence[idx], probs

    def write_output(self, sentence, probs):
        st.write("### Results")
        results = {"Probability": probs, "Sentence": self.sentence}
        st.table(results)
        st.write(f'#### {sentence}')

    def run_app(self):
        st.set_page_config(layout="wide")
        st.title('Quantum Prisoner\'s Dilemma')
        col1, col2, _ = st.columns([0.4, 0.4, 0.2])
        with col1:
            st.image('prisoners_dilemma.webp', use_container_width=True)
        with col2:
            technology_options = [r'$\textsf{classical}$', r'$\textsf{quantum}$']
            answer_otions = [r'$\textsf{confess}$', r'$\textsf{deflect}$']
            st.write(r'### $\textsf{Used Technology}$')
            technology = st.pills('', technology_options, default=technology_options[0], label_visibility='collapsed')
            if technology == technology_options[0]:
                st.write(r'#### Alice')
                alice = st.radio('Alice', answer_otions, index=0, label_visibility='collapsed')
                st.write(r'#### Bob')
                bob = st.radio('Bob', answer_otions, index=0, label_visibility='collapsed')
            elif technology == technology_options[1]:
                gamma = st.slider(r'$\gamma$', 0.0, np.pi / 2, np.pi / 2)
                magic = st.toggle(r'$\textsf{Allow Magic Gates}$')
                if magic:
                    magic_options = [r'$\textsf{Alice applies } M$', r'$\textsf{Both apply }Q$']
                    magic_choice = st.radio(r'', magic_options, index=0, label_visibility='collapsed')
                    if magic_choice == magic_options[0]:
                        st.write(r'#### Bob')
                        bob = st.radio('Bob', answer_otions, index=0, label_visibility='collapsed')
                else:
                    st.write(r'#### Alice')
                    alice = st.radio('Alice', answer_otions, index=0, label_visibility='collapsed')
                    st.write(r'#### Bob')
                    bob = st.radio('Bob', answer_otions, index=0, label_visibility='collapsed')
            if st.button('Run'):
                if technology == technology_options[0]:
                    sentence, probs = self.classical(alice ==  answer_otions[1], bob ==  answer_otions[1])
                    self.write_output(sentence, probs)
                if technology == technology_options[1]:
                    if magic:
                        if magic_choice == magic_options[1]:
                            sentence, probs = self.both_q(gamma)
                            self.write_output(sentence, probs)
                        else:
                            sentence, probs = self.alice_m(bob ==  answer_otions[1], gamma)
                            self.write_output(sentence, probs)
                    else:
                        sentence, probs = self.quantum(alice == answer_otions[1], bob ==  answer_otions[1], gamma)
                        self.write_output(sentence, probs)

if __name__ == "__main__":
    pd = PrisonersDilemma()
    pd.run_app()

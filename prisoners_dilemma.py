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
        self.M = self.sim.composite_gate([self.sim.S, self.sim.H, self.sim.Y, self.sim.S, self.sim.Z])
        
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
    
    def magic(self, alice_choice, bob_choice, gamma):
        init_state = self.sim.multikron([self.sim.zero] * 2)
        J = expm(np.kron(-1.j * gamma * self.D, self.D / 2))
        options = [self.C, self.D, self.Q, self.M]
        alice_gate = options[alice_choice]
        bob_gate = options[bob_choice]
        probs = self.get_probs(alice_gate, bob_gate, init_state, J)
        idx = np.random.choice(range(4), p=probs)
        return self.sentence[idx], probs
    
    def create_u(self, theta, phi):
        return np.array([[np.exp(1.j * phi) * np.cos(theta / 2), np.sin(theta / 2)],
                        [-np.sin(theta / 2), np.exp(-1.j * phi) * np.cos(theta / 2)]])

    def advanced(self, theta_alice, phi_alice, theta_bob, phi_bob, gamma):
        init_state = self.sim.multikron([self.sim.zero] * 2)
        J = expm(np.kron(-1.j * gamma * self.D, self.D / 2))
        alice_gate = self.create_u(theta_alice, phi_alice)
        bob_gate = self.create_u(theta_bob, phi_bob)
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
            technology_options = [r'$\textsf{classical}$', r'$\textsf{quantum}$', r'$\textsf{advanced}$']
            answer_options = [r'$\textsf{confess}$', r'$\textsf{deflect}$']
            st.write(r'### $\textsf{Used Technology}$')
            technology = st.pills('Technology', technology_options, default=technology_options[0], label_visibility='collapsed')

            ########## Classical ##########
            if technology == technology_options[0]:
                st.write(r'#### Alice')
                alice = st.radio('Alice', answer_options, index=0, label_visibility='collapsed')
                st.write(r'#### Bob')
                bob = st.radio('Bob', answer_options, index=0, label_visibility='collapsed')
            
            ########## Quantum ##########
            elif technology == technology_options[1]:
                gamma = st.slider(r'$\gamma$', 0.0, np.pi / 2, np.pi / 2)
                magic = st.toggle(r'$\textsf{Allow Magic Gates}$')

                ########## No Magic ##########
                if not magic:
                    st.write(r'#### Alice')
                    alice = st.radio('Alice', answer_options, index=0, label_visibility='collapsed')
                    st.write(r'#### Bob')
                    bob = st.radio('Bob', answer_options, index=0, label_visibility='collapsed')

                ########## Magic ##########
                else:
                    magic_answer_options = answer_options + [r'$Q$', r'$\textsf{Magic move } M$']
                    st.write(r'#### Alice')
                    alice = st.radio('Alice', magic_answer_options, index=0, label_visibility='collapsed')
                    st.write(r'#### Bob')
                    bob = st.radio('Bob', magic_answer_options, index=0, label_visibility='collapsed')

            ########## Advanced ##########
            elif technology == technology_options[2]:
                cont = st.toggle(r'$\textsf{Continuous Sliders}$' , value=True)

                ########## Continuous ##########
                if cont:
                    gamma = st.slider(r'$\gamma$ ', 0.0, np.pi / 2, 0.0)
                    st.write(r'#### Alice')
                    theta_alice = st.slider(r'$\theta_A$', 0.0, np.pi, 0.0)
                    phi_alice = st.slider(r'$\phi_A$', 0.0, np.pi / 2, 0.0)
                    st.write(r'#### Bob')
                    theta_bob = st.slider(r'$\theta_B$', 0.0, np.pi, 0.0)
                    phi_bob = st.slider(r'$\phi_B$', 0.0, np.pi / 2, 0.0)

                ########## Discrete ##########
                else:
                    pis = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4, np.pi]
                    pis_select = [r'0', r'pi/4', r'pi/2', r'3pi/4', r'pi']
                    gamma = pis[pis_select.index(st.selectbox(r'$\gamma$', pis_select[:3], index=0))]

                    st.write(r'#### Alice')
                    theta_alice = pis[pis_select.index(st.selectbox(r'$\theta_A$', pis_select, index=0))]
                    phi_alice = pis[pis_select.index(st.selectbox(r'$\phi_A$', pis_select[:3], index=0))]
                    st.write(r'#### Bob')
                    theta_bob = pis[pis_select.index(st.selectbox(r'$\theta_B$', pis_select, index=0))]
                    phi_bob = pis[pis_select.index(st.selectbox(r'$\phi_B$', pis_select[:3], index=0))]

            ########## Run Game ##########
            if st.button('Run'):
                if technology == technology_options[0]:
                    sentence, probs = self.classical(alice ==  answer_options[1], bob ==  answer_options[1])
                elif technology == technology_options[1] and not magic:
                    sentence, probs = self.quantum(alice == answer_options[1], bob ==  answer_options[1], gamma)
                elif technology == technology_options[1] and magic:
                    sentence, probs = self.magic(magic_answer_options.index(alice), magic_answer_options.index(bob), gamma)
                elif technology == technology_options[2]:
                    sentence, probs = self.advanced(theta_alice, phi_alice, theta_bob, phi_bob, gamma)
                self.write_output(sentence, probs)
                        

if __name__ == "__main__":
    pd = PrisonersDilemma()
    pd.run_app()

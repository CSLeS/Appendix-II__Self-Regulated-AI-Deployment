# Import libraries
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import differential_evolution
import random

# Creating the class for imperfect evaluations
class Eval:
    def __init__(self, phi_1, phi_0, q_2, q_1, lmda, pi_1, pi_2, pi_3, pi_4, pi_5, pi_6, pi_7, pi_8, pi_9):
        self.phi_1, self.phi_0, self.q_2, self.q_1, self.lmda, self.pi_1, self.pi_2, self.pi_3, self.pi_4, self.pi_5, self.pi_6, self.pi_7, self.pi_8, self.pi_9 = phi_1, phi_0, q_2, q_1, lmda, pi_1, pi_2, pi_3, pi_4, pi_5, pi_6, pi_7, pi_8, pi_9

    # Method that computes the 'A' in the separating IC constraint
    def A(self):
        return ((1-self.q_2)*((1-self.lmda * self.phi_1)*(self.pi_6-self.pi_3)-self.lmda * self.phi_1 * self.pi_2) + self.lmda * self.q_2 * (1-self.phi_0) * (self.pi_2-self.pi_1)-self.q_2 * self.pi_2)
    
    # Method that computes the 'B' in the separating IC constraint
    def B(self):
        return (self.lmda*(self.q_2*(1-self.phi_0)*(self.pi_4-self.pi_5)+self.phi_1*(1-self.q_2)*self.pi_5)+self.q_2*self.pi_5)

    # Method that computes the 'C' in the separating IC constraint
    def C(self):
        return ((1-self.q_2)*(1-self.lmda*self.phi_1)*(self.pi_9-self.pi_6))

    # Method that computes the 'D' in the separating IC constraint
    def D(self):
        return (self.lmda*self.q_2*(1-self.phi_0)*(self.pi_7-self.pi_4)+(self.q_2*(1-self.lmda*(1-self.phi_0))+(1-self.q_2)*self.lmda*self.phi_1)*(self.pi_8-self.pi_5))

    # Method that computes the separating IC constraint 
    def IC(self, A, B, C, D):
        return ((self.phi_1*(1-self.q_1)*B + (1-self.phi_0)*self.q_1*D) / (-self.phi_1*(1-self.q_1)*A - (1-self.phi_0)*self.q_1*C))
    
# Showing an example of how to use the class. All probabilities are a half and payoffs start at 10 and decrease by 1.
eg = Eval (phi_1=0.5, phi_0=0.5, q_2=0.5, q_1=0.5, lmda=0.5, pi_1=10, pi_2=9, pi_3=8, pi_4=7, pi_5=6, pi_6=5, pi_7=4, pi_8=3, pi_9=2)
print("\n Example IC:", min(1,max(0,eg.IC(eg.A(), eg.B(), eg.C(), eg.D()))))



# General optimisation setup

# The objective function to be minimized
def b_neg_IC(params, q_2, q_1, lmda, pi_1, pi_2, pi_3, pi_4, pi_5, pi_6, pi_7, pi_8, pi_9):
    phi_1, phi_0 = params
    example = Eval(phi_1, phi_0, q_2, q_1, lmda, pi_1, pi_2, pi_3, pi_4, pi_5, pi_6, pi_7, pi_8, pi_9)
    return -min(1,max(0,(example.IC(example.A(), example.B(), example.C(), example.D())))) # Notice it is bounded between 0 and 1.

# Parameters
params = (0.5, 0.5, 0.5, 10, 9, 8, 7, 6, 5, 4, 3, 2)

# Initial guess for phi_1 and phi_0
initial_guess = [0.5, 0.5]

# Bounds for phi_1 and phi_0
bounds = [(0, 1), (0, 1)]

# Minimise the negative IC to maximize the IC, using differential evolu
result = differential_evolution(b_neg_IC, bounds=bounds, args=params)

# Display the results
print("\n Maximized IC:", -result.fun)
print("Optimal phi_1:", result.x[0])
print("Optimal phi_0:", result.x[1])



# PROPOSITION 1 (SECTION 4.3)

# Function to generate random parameters
def generate_random_params():
    q_2 = random.uniform(0, 1)
    q_1 = random.uniform(0, 1)
    lmda = random.uniform(0, 1)
    pi_values = sorted([random.uniform(10, 60) for _ in range(9)], reverse=True)
    return (q_2, q_1, lmda, *pi_values)

# Run the optimization with random parameters multiple times
num_iterations = 10000  # Reduce this number to speed up the process
counter = 0

for _ in range(num_iterations):
    random_params = generate_random_params()
    result = differential_evolution(b_neg_IC, bounds=bounds, args=params)
    
    if round(result.x[0], 2) == 1 and round(result.x[1], 2) == 1:
        counter += 1

    print(f"Completed {_ +1}/{num_iterations} iterations for ⟨phi_1=1,phi_0= 1⟩.")

print("")
print(f"In {counter} out of {num_iterations} iterations, the maximum IC is attained at phi_1 = 1 and phi_0 = 1.")



# MONOTONICITY PROBLEM (SECTION 4.3)
# Updated function to generate random parameters
def generate_random_params():
    q_2 = random.uniform(0, 1)
    q_1 = random.uniform(0, 1)
    lmda = random.uniform(0, 1)
    pi_values = sorted([random.uniform(10, 60) for _ in range(9)], reverse=True)
    return (q_2, q_1, lmda, *pi_values)

# Function to check if IC is monotonically increasing in phi_1 and phi_0
def check_monotonicity(params):
    q_2, q_1, lmda, pi_1, pi_2, pi_3, pi_4, pi_5, pi_6, pi_7, pi_8, pi_9 = params
    phi_1_values = np.linspace(0, 1, 10)
    phi_0_values = np.linspace(0, 1, 10)

    monotonic_phi_1 = True
    monotonic_phi_0 = True

    non_monotonic_cases = []

    for j in range(len(phi_0_values) - 1):
        for i in range(len(phi_1_values) - 1):
            example1 = Eval(phi_1_values[i], phi_0_values[j], q_2, q_1, lmda, pi_1, pi_2, pi_3, pi_4, pi_5, pi_6, pi_7, pi_8, pi_9)
            example2 = Eval(phi_1_values[i + 1], phi_0_values[j], q_2, q_1, lmda, pi_1, pi_2, pi_3, pi_4, pi_5, pi_6, pi_7, pi_8, pi_9)

            if min(1,max(0,example2.IC(example2.A(), example2.B(), example2.C(), example2.D()))) < min(1,max(0,example1.IC(example1.A(), example1.B(), example1.C(), example1.D()))):
                monotonic_phi_1 = False

            example1 = Eval(phi_1_values[i], phi_0_values[j], q_2, q_1, lmda, pi_1, pi_2, pi_3, pi_4, pi_5, pi_6, pi_7, pi_8, pi_9)
            example2 = Eval(phi_1_values[i], phi_0_values[j + 1], q_2, q_1, lmda, pi_1, pi_2, pi_3, pi_4, pi_5, pi_6, pi_7, pi_8, pi_9)

            if min(1,max(0,example2.IC(example2.A(), example2.B(), example2.C(), example2.D()))) < min(1,max(0,example1.IC(example1.A(), example1.B(), example1.C(), example1.D()))):
                monotonic_phi_0 = False
                non_monotonic_cases.append((phi_1_values[i], phi_0_values[j], phi_0_values[j + 1]))

        if not (monotonic_phi_1 and monotonic_phi_0):
            break

    return monotonic_phi_1, monotonic_phi_0, non_monotonic_cases


# Run the check for monotonicity with random parameters multiple times
num_iterations = 1000 # Reduce this number to speed up the process
phi_1_counter = 0
phi_0_counter = 0

non_monotonic_params = []

for _ in range(num_iterations):
    random_params = generate_random_params()
    monotonic_phi_1, monotonic_phi_0, non_monotonic_cases = check_monotonicity(random_params)

    if monotonic_phi_1:
        phi_1_counter += 1

    if monotonic_phi_0:
        phi_0_counter += 1
        non_monotonic_params.append((random_params, non_monotonic_cases))

    print(f"Completed {_ + 1}/{num_iterations} iterations.")

print("")
print(f"In {100 * phi_1_counter / num_iterations} per cent of the iterations, IC grows monotonically in phi_1.")
print(f"In {100 * phi_0_counter / num_iterations} per cent of the iterations, IC grows monotonically in phi_0.")

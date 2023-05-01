import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import differential_evolution
import random
import matplotlib.colors as mcolors
from scipy.interpolate import PchipInterpolator

# Creating the class for imperfect evaluations with a tradeoff 
class Eval_tr:
    def __init__(self, tau, phi_1, q_2, q_1, lmda, pi_1, pi_2, pi_3, pi_4, pi_5, pi_6, pi_7, pi_8, pi_9):
        self.tau, self.phi_1, self.q_2, self.q_1, self.lmda, self.pi_1, self.pi_2, self.pi_3, self.pi_4, self.pi_5, self.pi_6, self.pi_7, self.pi_8, self.pi_9 = tau, phi_1, q_2, q_1, lmda, pi_1, pi_2, pi_3, pi_4, pi_5, pi_6, pi_7, pi_8, pi_9

    # Method that computes the 'A' in the separating IC constraint
    def A_tr(self):
        return ((1-self.q_2)*((1-self.lmda * self.phi_1)*(self.pi_6-self.pi_3)-self.lmda * self.phi_1 * self.pi_2) + self.lmda * self.q_2 * (1-(1-self.phi_1**self.tau)) * (self.pi_2-self.pi_1)-self.q_2 * self.pi_2)
    
    # Method that computes the 'B' in the separating IC constraint
    def B_tr(self):
        return (self.lmda*(self.q_2*(1-(1-self.phi_1**self.tau))*(self.pi_4-self.pi_5)+self.phi_1*(1-self.q_2)*self.pi_5)+self.q_2*self.pi_5)

    # Method that computes the 'C' in the separating IC constraint
    def C_tr(self):
        return ((1-self.q_2)*(1-self.lmda*self.phi_1)*(self.pi_9-self.pi_6))

    # Method that computes the 'D' in the separating IC constraint
    def D_tr(self):
        return (self.lmda*self.q_2*(1-(1-self.phi_1**self.tau))*(self.pi_7-self.pi_4)+(self.q_2*(1-self.lmda*(1-(1-self.phi_1**self.tau)))+(1-self.q_2)*self.lmda*self.phi_1)*(self.pi_8-self.pi_5))

    # Method that computes the separating IC constraint 
    def IC_tr(self, A, B, C, D):
        denominator = (-self.phi_1 * (1 - self.q_1) * A - (1 - (1 - self.phi_1 ** self.tau)) * self.q_1 * C)
        if denominator == 0:
            return 1
        else:
            return ((self.phi_1 * (1 - self.q_1) * B + (1 - (1 - self.phi_1 ** self.tau)) * self.q_1 * D) / denominator)

# Showing an example of how to use the class. Tradeoff is 5, all probabilities are a half and payoffs start at 10 and decrease by 1.
eg = Eval_tr (tau=5, phi_1=1, q_2=0.5, q_1=0.5, lmda=0.5, pi_1=10, pi_2=9, pi_3=8, pi_4=7, pi_5=6, pi_6=5, pi_7=4, pi_8=3, pi_9=2)
print("\n Example IC:", eg.IC_tr(eg.A_tr(), eg.B_tr(), eg.C_tr(), eg.D_tr())) 



# General optimisation setup

# The objective function to be minimized
def b_neg_IC_tr(params, tau, q_2, q_1, lmda, pi_1, pi_2, pi_3, pi_4, pi_5, pi_6, pi_7, pi_8, pi_9):
    phi_1 = params[0]
    example = Eval_tr(tau, phi_1, q_2, q_1, lmda, pi_1, pi_2, pi_3, pi_4, pi_5, pi_6, pi_7, pi_8, pi_9)
    return -min(1,max(0,(example.IC_tr(example.A_tr(), example.B_tr(), example.C_tr(), example.D_tr())))) # Notice it is bounded between 0 and 1.

# Example parameter list
params = (20, 0.5, 0.5, 0.5, 10, 9, 8, 7, 6, 5, 4, 3, 2)

# Initial guess for phi_1
initial_guess = [0.5]

# Bounds for phi_1
bounds = [(0, 1)]

# Minimise the negative IC to maximize the IC
result = differential_evolution(b_neg_IC_tr, bounds=bounds, args=params)

# Display the example's results
print("\n Example maximized IC_tr:", -result.fun)
print("Example optimal phi_1:", result.x[0])



# FIGURE 5 (SECTION 4.3)
 
# Function to generate random parameters
def generate_random_params():
    q_2 = random.uniform(0, 1)
    q_1 = random.uniform(0, 1)
    lmda = random.uniform(0, 1)
    pi_values = sorted([random.uniform(10, 60) for _ in range(9)], reverse=True)
    return (q_2, q_1, lmda, *pi_values)

# tau values to be used
# tau_values = [0, 0.1, 0.5, 1, 2]
tau_values = [5, 15, 35, 50, 100]

# Dictionary to store optimal phi_1 values for each tau
optimal_phi_1 = {tau: [] for tau in tau_values}

# Run the optimization with random parameters multiple times
num_iterations = 10

for _ in range(num_iterations):
    random_params = generate_random_params()
    for tau in tau_values:
        params = (tau, *random_params)
        result = differential_evolution(b_neg_IC_tr, bounds=bounds, args=params)
        optimal_phi_1[tau].append(result.x[0])

    print(f"Completed {_ +1}/{num_iterations} iterations")

# Plot the scatterplots
fig, axs = plt.subplots(1, 5, figsize=(21, 3), sharey=True)

for idx, tau in enumerate(tau_values):
    axs[idx].scatter(range(num_iterations), optimal_phi_1[tau], s=30, alpha=0.1)
    axs[idx].set_title(f"τ = {tau}")  
    axs[idx].set_xticks([])  
    axs[idx].set_yticks([])  
    axs[idx].set_xlim(0, num_iterations)
    axs[idx].set_ylim(0, 1)

plt.tight_layout()
plt.show()



# FIGURE 6 (SECTION 4.4)

tau_values = [0.5, 5, 50]
r_values = [0.1, 0.125, 0.15, 0.175, 0.2, 0.225, 0.25, 0.275, 0.3, 0.325, 0.35, 0.375, 0.4, 0.425, 0.45, 0.475, 0.5, 0.525, 0.55, 0.575, 0.6, 0.625, 0.65, 0.675, 0.7, 0.725, 0.75, 0.775, 0.8, 0.825, 0.85, 0.875, 0.9, 0.925, 0.95, 0.975]
phi_1_values = [0.1, 0.125, 0.15, 0.175, 0.2, 0.225, 0.25, 0.275, 0.3, 0.325, 0.35, 0.375, 0.4, 0.425, 0.45, 0.475, 0.5, 0.525, 0.55, 0.575, 0.6, 0.625, 0.65, 0.675, 0.7, 0.725, 0.75, 0.775, 0.8, 0.825, 0.85, 0.875, 0.9, 0.925, 0.95, 0.975]

num_iterations = 10000 # Reduce this number to speed up the process

tables = {}
for tau in tau_values:
    tau_table = []
    for r in r_values:
        row_data = []
        for phi_1 in phi_1_values:
            count = 0
            for _ in range(num_iterations):
                q_2, q_1, lmda, pi_1, pi_2, pi_3, pi_4, pi_5, pi_6, pi_7, pi_8, pi_9 = generate_random_params()
                example = Eval_tr(tau, phi_1, q_2, q_1, lmda, pi_1, pi_2, pi_3, pi_4, pi_5, pi_6, pi_7, pi_8, pi_9)
                IC_tr_val = example.IC_tr(example.A_tr(), example.B_tr(), example.C_tr(), example.D_tr())
                one_r = 1 - r
                if one_r <= IC_tr_val:
                    count += 1
            row_data.append(count / num_iterations)
        tau_table.append(row_data)
    tables[tau] = tau_table


# Plot as 3D scatterplots
fig = plt.figure(figsize=(20, 6))
for idx, tau in enumerate(tau_values, start=1):
    ax = fig.add_subplot(1, 3, idx, projection='3d')
    x_data, y_data, z_data = [], [], []

    for i, r in enumerate(r_values):
        for j, phi_1 in enumerate(phi_1_values):
            x_data.append(r)
            y_data.append(phi_1)
            z_data.append(tables[tau][i][j])

    cmap = mcolors.LinearSegmentedColormap.from_list("", ["red", "yellow", "green"])
    scatter = ax.scatter(x_data, y_data, z_data, c=z_data, cmap=cmap, marker="o", alpha=0.5, s=7.5, edgecolors='none')

    ax.set_xlabel("r", labelpad=-15)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['0', '1'])
    ax.invert_xaxis()

    ax.set_ylabel("$\\phi_1$", labelpad=-15)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(['0', '1'])

    ax.set_zlabel("P(1-r ≤ IC)", labelpad=-15, x=1.1)
    ax.set_zticks([0, 1])
    ax.set_zticklabels(['0', '1'])

    ax.set_title(f"$\\tau$ = {tau}", y=0.975)
    ax.view_init(elev=20, azim=-30)
    ax.w_xaxis.set_pane_color((0.03, 0.03, 0.03, 0.03))
    ax.w_yaxis.set_pane_color((0.03, 0.03, 0.03, 0.03))
    ax.w_zaxis.set_pane_color((0.03, 0.03, 0.03, 0.03))

    # Set color of axis lines
    ax.xaxis.line.set_color("gray")
    ax.yaxis.line.set_color("gray")
    ax.zaxis.line.set_color("gray")

    # Remove gridlines
    ax.xaxis._axinfo['grid']['linewidth'] = 0
    ax.yaxis._axinfo['grid']['linewidth'] = 0
    ax.zaxis._axinfo['grid']['linewidth'] = 0
    ax.xaxis._axinfo['grid']['linestyle'] = "-"
    ax.yaxis._axinfo['grid']['linestyle'] = "-"
    ax.zaxis._axinfo['grid']['linestyle'] = "-"

    # Change the position of ticks to be closer to the graphs
    ax.tick_params(axis='both', which='both', pad=-4)

plt.tight_layout()
plt.show()



# FIGURE 7 (SECTION 4.4)

# Add the P_SE_times_1_minus_P_HC function
def P_SE_times_1_minus_P_HC(tau, phi_1, q_2, q_1, lmda, pi_1, pi_2, pi_3, pi_4, pi_5, pi_6, pi_7, pi_8, pi_9):
    example = Eval_tr(tau, phi_1, q_2, q_1, lmda, pi_1, pi_2, pi_3, pi_4, pi_5, pi_6, pi_7, pi_8, pi_9)
    P_SE = example.IC_tr(example.A_tr(), example.B_tr(), example.C_tr(), example.D_tr())
    P_HC = 1 - (q_1 + (1 - q_1) * lmda * phi_1) * (q_2 + (1 - q_2) * lmda * phi_1)
    return P_SE * (1 - P_HC)

# Modify the objective function
def b_neg_P_SE_times_1_minus_P_HC(params, tau, q_2, q_1, lmda, pi_1, pi_2, pi_3, pi_4, pi_5, pi_6, pi_7, pi_8, pi_9):
    phi_1 = params[0]
    return -P_SE_times_1_minus_P_HC(tau, phi_1, q_2, q_1, lmda, pi_1, pi_2, pi_3, pi_4, pi_5, pi_6, pi_7, pi_8, pi_9)

# Tau values for the first 2D plot
tau_values_2D = np.arange(0.05, 10, 0.1)

# New random parameters (NB: these can be adjusted to create various figures. E.g., adjust r/lmda if you are not assuming a=0)
def new_random_params():
    q_2 = random.uniform(0, 1)
    q_1 = random.uniform(0, 1)
    r = random.uniform(0.5+1e-10, 1) # better than random signal
    lmda = r # assuming a=0
    pi_values = sorted([random.uniform(10, 60) for _ in range(9)], reverse=True)
    return (q_2, q_1, lmda, *pi_values)

# Optimize for each tau and plot the results
optimal_phi_1_2D = []

num_iterations = 10000 # Reduce this number to speed up the process

for tau in tau_values_2D:
    print(f"Current tau value: {tau}")
    optimal_phi_1_sum = 0
    for _ in range(num_iterations):
        random_params = new_random_params()
        params = (tau, *random_params)
        result = differential_evolution(b_neg_P_SE_times_1_minus_P_HC, bounds=bounds, args=params)
        optimal_phi_1_sum += result.x[0]
    optimal_phi_1_avg = optimal_phi_1_sum / num_iterations
    optimal_phi_1_2D.append(optimal_phi_1_avg)

# Scatter phi_1 optimal values
plt.scatter(tau_values_2D, optimal_phi_1_2D, s=1.5, c='black')

# Include the implied phi_0
phi_0_opt = [1 - phi_1**tau for phi_1, tau in zip(optimal_phi_1_2D, tau_values_2D)]
plt.scatter(tau_values_2D, phi_0_opt, s=1.5, c='darkblue')


# Add lines that follow the data

# For phi_1: a gray transparent line 
# First, create a PCHIP interpolation of the data points
interpolator = PchipInterpolator(tau_values_2D, optimal_phi_1_2D)
# Generate new x and y values for a smooth curve
x_smooth = np.linspace(tau_values_2D.min(), tau_values_2D.max(), 500)
y_smooth = interpolator(x_smooth)
# Clip the y_smooth values at 1, so the gray line doesn't go above 1
y_smooth_clipped = np.clip(y_smooth, None, 1)
# Plot the gray transparent line with clipped y-values
plt.plot(x_smooth, y_smooth_clipped, color='gray', alpha=0.6)

# For phi_0: a blue transparent line
# Create a PCHIP interpolation of the phi_0 data points
interpolator_phi_0 = PchipInterpolator(tau_values_2D, phi_0_opt)
# Generate new x and y values for a smooth phi_0 curve
y_smooth_phi_0 = interpolator_phi_0(x_smooth)
# Plot the blue transparent line for phi_0
plt.plot(x_smooth, y_smooth_phi_0, color='darkblue', alpha=0.2)


# Add phi_1 and phi_0 labels next to their respective lines
phi_1_label_x = x_smooth[10]
phi_1_label_y = y_smooth_clipped[150]
plt.text(phi_1_label_x, phi_1_label_y, ' ', fontsize=12)

phi_0_label_x = x_smooth[10]
phi_0_label_y = y_smooth_phi_0[20]
plt.text(phi_0_label_x, phi_0_label_y, ' ', fontsize=12, color='darkblue')


plt.xlabel('τ')
plt.ylabel('')
plt.ylim([0, 1])
plt.title('Optimal phi_1,phi_0 for different tau values')
plt.show()
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import expit

# Define the sigmoid function parameters for multiple variations
# c0 - base fee - for delta P = 0
# c1 - max fee - for delta P = inf
# c2 - steepness of the curve
# c3 - inflection point of the curve

params = [
    # {"c0": 0.0, "c1": 1, "c2": 400, "c3": 0.01},
    # {"c0": 0.0, "c1": 1, "c2": 500, "c3": 0.01},
    # {"c0": 0.0, "c1": 1, "c2": 600, "c3": 0.01},
    # {"c0": 0.0, "c1": 1, "c2": 700, "c3": 0.0075},
    {"c0": 0.0, "c1": 1, "c2": 600, "c3": 0.01},
    {"c0": 0.0, "c1": 1, "c2": 800, "c3": 0.0075},
    {"c0": 0.0, "c1": 1, "c2": 1000, "c3": 0.005},
    # {"c0": 0.0, "c1": 1, "c2": 600, "c3": 0.01},
    # {"c0": 0.0, "c1": 1, "c2": 700, "c3": 0.01},
    # {"c0": 0.0, "c1": 1, "c2": 700, "c3": 0.0125},
    # {"c0": 0.0, "c1": 1, "c2": 400, "c3": 0.015},
    # {"c0": 0.0, "c1": 1, "c2": 500, "c3": 0.015},
    # {"c0": 0.0, "c1": 1, "c2": 600, "c3": 0.015},
    # {"c0": 0.0, "c1": 1, "c2": 700, "c3": 0.015},
]

# Define the price delta range (0% to 100% delta)
deltaP = np.linspace(0, 1.0, 500)  # Delta P in percentage decimals from 0% to 100%


# Sigmoid S-curve function
def sigmoid_fee(deltaP, c0, c1, c2, c3):
    exponent = c2 * (deltaP - c3)
    fee = c0 + c1 * expit(exponent)
    return fee


# Plot the curves for each set of parameters
plt.figure(figsize=(12, 8))
for i, param in enumerate(params):
    fees = sigmoid_fee(deltaP, param["c0"], param["c1"], param["c2"], param["c3"])
    plt.plot(
        deltaP * 100,
        fees * 100,
        label=f"Curve {i+1}: c0={param['c0']}, c1={param['c1']}, c2={param['c2']}, c3={param['c3']}",
        linewidth=2,
    )

plt.xlim(0, 6)

# Plot settings
plt.title("Sigmoid S-Curve Fee Model")
plt.xlabel("Price Delta (%)")
plt.ylabel("Abs Delta Swap Fee (%)")
plt.grid(True)
plt.legend()
plt.show()

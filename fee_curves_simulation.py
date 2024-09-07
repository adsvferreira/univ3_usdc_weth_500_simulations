import numpy as np
import matplotlib.pyplot as plt

# THIS CURVE WILL BE USED FOR CALCULATE THE FEE DELTA IN BPS.
# MOST LIKELY WE WILL HAVE DIFFERENT CURVES FOR THE INCREASE AND DECREASE FEE.
# INCREASE ABS - BETWWEN 0 AND ~10%
# DECREASE ABS - BETWEEN 0 AND BASE FEE (EX: 0.05%)


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
    exponent = -c2 * (deltaP - c3)
    fee = c0 + c1 / (1 + np.exp(exponent))
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

# plt.xticks(range(0, 10, 1))
# plt.yticks(range(0, 22000, 250), fontsize=6)
# plt.yticks(range(0, 22000, 100), fontsize=6)

# plt.xticks(range(0, 102, 2), fontsize=6)

plt.xlim(0, 6)
# plt.ylim(0, 2000)

# Plot settings
plt.title("Sigmoid S-Curve Variations for Abs Delta Swap Fee (%)")
plt.xlabel("Price Delta (%)")
plt.ylabel("Abs Delta Swap Fee (%)")
plt.grid(True)
plt.legend()
plt.show()

import matplotlib.pyplot as plt

# ---------- Plot 1: OR Blocking ----------
configs = ["3p4r", "3p5r", "4p5r"]
means = [0.0106, 0.0021, 0.0038]
lower = [0.0019, -0.00056, -0.00091]
upper = [0.0193, 0.00476, 0.00851]

plt.errorbar(configs, means,
             yerr=[[means[i]-lower[i] for i in range(3)],
                   [upper[i]-means[i] for i in range(3)]],
             fmt="o-")
plt.title("95% CI for OR Blocking Probability")
plt.ylabel("Probability")
plt.grid(True)
plt.show()

# ---------- Plot 2: Recovery FULL ----------
configs = ["3p4r", "3p5r", "4p5r"]
means = [0.0487, 0.00965, 0.01405]
lower = [0.0241, 0.00256, 0.00389]
upper = [0.0733, 0.01674, 0.02421]

plt.errorbar(configs, means,
             yerr=[[means[i]-lower[i] for i in range(3)],
                   [upper[i]-means[i] for i in range(3)]],
             fmt="o-")
plt.title("95% CI for Recovery FULL Probability")
plt.ylabel("Probability")
plt.grid(True)
plt.show()

# ---------- Plot 3: Prep Queue ----------
configs = ["3p4r", "3p5r", "4p5r"]
means = [2.64, 2.53, 0.65]
lower = [1.35, 1.27, 0.40]
upper = [3.93, 3.78, 0.89]

plt.errorbar(configs, means,
             yerr=[[means[i]-lower[i] for i in range(3)],
                   [upper[i]-means[i] for i in range(3)]],
             fmt="o-")
plt.title("95% CI for Prep Queue Length")
plt.ylabel("Queue Length")
plt.grid(True)
plt.show()

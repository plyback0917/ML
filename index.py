
import matplotlib.pyplot as plt
import numpy as np

# Generate sample data
x = np.linspace(0, 10, 100)
y = np.sin(x)

# Create a simple line plot
plt.figure(figsize=(8, 5))
plt.plot(x, y, label="Sine Wave", color="blue")
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.title("Simple Sine Wave Graph")
plt.legend()
plt.grid()

# Show the plot
plt.show()
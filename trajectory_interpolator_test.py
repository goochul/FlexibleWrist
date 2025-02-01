import numpy as np
import matplotlib.pyplot as plt

# Parameters
q_start = 0.0  # Start position
q_goal = 0.01  # Goal position
dq_max = 0.002  # Maximum velocity
ddq_max = 0.001  # Maximum acceleration

# Time for acceleration and deceleration phases
t1 = dq_max / ddq_max  # Time to reach maximum velocity
q1 = q_start + 0.5 * ddq_max * t1**2  # Position at the end of acceleration phase
q2 = q_goal - 0.5 * ddq_max * t1**2  # Position at the start of deceleration phase

# Time for constant velocity phase
t2 = t1 + (q2 - q1) / dq_max

# Total trajectory time
t_f = t2 + t1

# Time intervals
t = np.linspace(0, t_f, 1000)

# Initialize trajectory arrays
q = np.zeros_like(t)
v = np.zeros_like(t)
a = np.zeros_like(t)

for i, time in enumerate(t):
    if time <= t1:  # Acceleration phase
        q[i] = q_start + 0.5 * ddq_max * time**2
        v[i] = ddq_max * time
        a[i] = ddq_max
    elif time <= t2:  # Constant velocity phase
        q[i] = q1 + dq_max * (time - t1)
        v[i] = dq_max
        a[i] = 0
    else:  # Deceleration phase
        dt = time - t2
        q[i] = q2 + dq_max * dt - 0.5 * ddq_max * dt**2
        v[i] = dq_max - ddq_max * dt
        a[i] = -ddq_max

# Plotting the trajectory
plt.figure(figsize=(12, 8))

plt.subplot(3, 1, 1)
plt.plot(t, q, label='Position (q)')
plt.xlabel('Time (s)')
plt.ylabel('Position')
plt.title('Position vs. Time')
plt.grid()
plt.legend()

plt.subplot(3, 1, 2)
plt.plot(t, v, label='Velocity (v)', color='orange')
plt.xlabel('Time (s)')
plt.ylabel('Velocity')
plt.title('Velocity vs. Time')
plt.grid()
plt.legend()

plt.subplot(3, 1, 3)
plt.plot(t, a, label='Acceleration (a)', color='green')
plt.xlabel('Time (s)')
plt.ylabel('Acceleration')
plt.title('Acceleration vs. Time')
plt.grid()
plt.legend()

plt.tight_layout()
plt.show()
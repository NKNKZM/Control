import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# 定义系统参数
m = 1.0   # 质量
k = 2.0   # 弹簧刚度
b = 0.5   # 阻尼系数

# 定义哈密顿函数 H(q, p)
def hamiltonian(q, p):
    return (p**2) / (2 * m) + 0.5 * k * q**2

# 定义梯度 ∇H(q, p)
def grad_H(q, p):
    dHdq = k * q       # ∂H/∂q
    dHdp = p / m       # ∂H/∂p
    return np.array([dHdq, dHdp])

# 定义端口哈密顿系统动态方程
def port_hamiltonian(t, x, u_func):
    q, p = x
    u = u_func(t)  # 输入力 F(t)
    
    # 结构矩阵
    J = np.array([[0, 1], [-1, 0]])
    R = np.array([[0, 0], [0, b]])
    g = np.array([0, 1])
    
    # 计算动态方程 dx/dt = (J - R)∇H + g u
    dxdt = (J - R) @ grad_H(q, p) + g * u
    return dxdt

# 定义输入力函数 (例如：阶跃输入)
def input_force(t):
    return 1.0 if t >= 1.0 else 0.0  # t=1s 时施加 F=1

# 初始状态
x0 = [0.0, 0.0]  # q=0, p=0

# 仿真时间
t_span = (0, 10)
t_eval = np.linspace(0, 10, 1000)

# 求解ODE
sol = solve_ivp(
    port_hamiltonian,
    t_span,
    x0,
    t_eval=t_eval,
    args=(input_force,),
    method='RK45'
)

# 提取结果
q = sol.y[0]
p = sol.y[1]
t = sol.t

# 计算速度 v = p / m
v = p / m

# 绘制结果
plt.figure(figsize=(10, 6))
plt.plot(t, q, label='Displacement (q)')
plt.plot(t, v, label='Velocity (v)')
plt.xlabel('Time (s)')
plt.ylabel('State')
plt.title('Port-Hamiltonian System Simulation (Mass-Spring-Damper)')
plt.legend()
plt.grid(True)
plt.show()
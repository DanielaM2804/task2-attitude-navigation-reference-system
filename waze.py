import numpy as np                      # cálculos numéricos
import pandas as pd                    # (no obligatorio pero útil)
import matplotlib.pyplot as plt        # gráficas
from matplotlib.animation import FuncAnimation  # animación GIF

data = np.loadtxt("tello_imu_example.csv", delimiter=",", skiprows=1)

t  = data[:,0]   # tiempo
p  = data[:,1]   # velocidad angular x
q  = data[:,2]   # velocidad angular y
r  = data[:,3]   # velocidad angular z
ax = data[:,4]   # aceleración x cuerpo
ay = data[:,5]   # aceleración y cuerpo
az = data[:,6]   # aceleración z cuerpo

# CONFIGURACION INICIAL
N = len(t)                                # número de muestras
dt = np.diff(t, prepend=t[0])             # paso de tiempo

# ángulos de Euler
phi   = np.zeros(N)
theta = np.zeros(N)
psi   = np.zeros(N)

# gravedad
G = 9.80665

# vectores NED
acc_ned = np.zeros((N,3))   # aceleración
vel_ned = np.zeros((N,3))   # velocidad
pos_ned = np.zeros((N,3))   # posición

# CUATERNIONES
quat = np.zeros((N,4))      # [q0 q1 q2 q3]
theta_q = np.zeros(N)       # ángulo del cuaternión
eje_q = np.zeros((N,3))     # eje del cuaternión

# MATRIZ ROTACION

def R_zyx_long(phi, theta, psi):

    cph, sph = np.cos(phi), np.sin(phi)
    cth, sth = np.cos(theta), np.sin(theta)
    cps, sps = np.cos(psi), np.sin(psi)

    return np.array([
        [cth*cps,                     cth*sps,                    -sth],
        [sph*sth*cps-cph*sps,         sph*sth*sps+cph*cps,        sph*cth],
        [cph*sth*cps+sph*sps,         cph*sth*sps-sph*cps,        cph*cth]
    ])


# MATRIZ PQR → EULER

def R_pqr_matrix(phi,theta):

    cph, sph = np.cos(phi), np.sin(phi)
    cth, sth = np.cos(theta), np.sin(theta)

    return np.array([
        [1, sph*sth/cth, cph*sth/cth],
        [0, cph,        -sph],
        [0, sph/cth,     cph/cth]
    ])

# EULER → QUATERNION

def dcm_to_quaternion(C):
    tr = np.trace(C)

    if tr > 0:
        S = np.sqrt(tr + 1.0) * 2.0
        q0 = 0.25 * S
        q1 = (C[2,1] - C[1,2]) / S
        q2 = (C[0,2] - C[2,0]) / S
        q3 = (C[1,0] - C[0,1]) / S

    elif (C[0,0] > C[1,1]) and (C[0,0] > C[2,2]):
        S = np.sqrt(1.0 + C[0,0] - C[1,1] - C[2,2]) * 2.0
        q0 = (C[2,1] - C[1,2]) / S
        q1 = 0.25 * S
        q2 = (C[0,1] + C[1,0]) / S
        q3 = (C[0,2] + C[2,0]) / S

    elif C[1,1] > C[2,2]:
        S = np.sqrt(1.0 + C[1,1] - C[0,0] - C[2,2]) * 2.0
        q0 = (C[0,2] - C[2,0]) / S
        q1 = (C[0,1] + C[1,0]) / S
        q2 = 0.25 * S
        q3 = (C[1,2] + C[2,1]) / S

    else:
        S = np.sqrt(1.0 + C[2,2] - C[0,0] - C[1,1]) * 2.0
        q0 = (C[1,0] - C[0,1]) / S
        q1 = (C[0,2] + C[2,0]) / S
        q2 = (C[1,2] + C[2,1]) / S
        q3 = 0.25 * S

    q_vec = np.array([q0, q1, q2, q3], dtype=float)
    q_vec = q_vec / np.linalg.norm(q_vec)
    return q_vec

# QUATERNION → ANGULO-EJE

def quaternion_to_angle_axis(q_vec):
    q0 = np.clip(q_vec[0], -1.0, 1.0)
    theta_rot = 2.0 * np.arccos(q0)

    s = np.sqrt(max(1.0 - q0**2, 0.0))
    if s < 1e-8:
        eje = np.array([1.0, 0.0, 0.0], dtype=float)
    else:
        eje = q_vec[1:] / s

    return theta_rot, eje

# INTEGRAR ANGULOS

for i in range(1,N):

    H = R_pqr_matrix(phi[i-1],theta[i-1])
    pqr = np.array([p[i-1],q[i-1],r[i-1]])

    euler_dot = H @ pqr

    phi[i]   = phi[i-1]   + euler_dot[0]*dt[i]
    theta[i] = theta[i-1] + euler_dot[1]*dt[i]
    psi[i]   = psi[i-1]   + euler_dot[2]*dt[i]

# CALCULAR CUATERNIONES DESDE EULER

for i in range(N):
    C = R_zyx_long(phi[i], theta[i], psi[i])
    quat[i] = dcm_to_quaternion(C)
    theta_q[i], eje_q[i] = quaternion_to_angle_axis(quat[i])

# ACELERACION NED

for i in range(N):

    Rot_ned = R_zyx_long(phi[i],theta[i],psi[i]).T

    A_body = np.array([ax[i], ay[i], az[i]])

    # RESTAR gravedad (corrección importante)
    A_ned = Rot_ned @ A_body + np.array([0,0,G])

    acc_ned[i] = A_ned

# INTEGRAR VELOCIDAD Y POSICION

for i in range(1,N):

    vel_ned[i] = vel_ned[i-1] + acc_ned[i] * dt[i]
    pos_ned[i] = pos_ned[i-1] + vel_ned[i] * dt[i]

# EXTRAER COMPONENTES

PN = pos_ned[:,0]
PE = pos_ned[:,1]
PD = pos_ned[:,2]

VN = vel_ned[:,0]
VE = vel_ned[:,1]
VD = vel_ned[:,2]

AN = acc_ned[:,0]
AE = acc_ned[:,1]
AD = acc_ned[:,2]

# FIGURAS (3x3)

plt.style.use("seaborn-v0_8-whitegrid")

fig, axs = plt.subplots(3,3, figsize=(14,10))

# POSICION

axs[0,0].plot(t, PN)
axs[0,0].set_title("Posición Norte")

axs[0,1].plot(t, PE)
axs[0,1].set_title("Posición Este")

axs[0,2].plot(t, PD)
axs[0,2].set_title("Altura")

# VELOCIDAD

axs[1,0].plot(t, VN)
axs[1,0].set_title("Velocidad Norte")

axs[1,1].plot(t, VE)
axs[1,1].set_title("Velocidad Este")

axs[1,2].plot(t, VD)
axs[1,2].set_title("Velocidad Down")

# ACELERACION

axs[2,0].plot(t, AN)
axs[2,0].set_title("Aceleración Norte")

axs[2,1].plot(t, AE)
axs[2,1].set_title("Aceleración Este")

axs[2,2].plot(t, AD)
axs[2,2].set_title("Aceleración Down")

plt.tight_layout()
plt.show()

# GRAFICA DE CUATERNIONES

plt.figure(figsize=(10,6))

plt.plot(t, quat[:,0], label="q0")
plt.plot(t, quat[:,1], label="q1")
plt.plot(t, quat[:,2], label="q2")
plt.plot(t, quat[:,3], label="q3")

plt.title("Cuaterniones")
plt.xlabel("Tiempo [s]")
plt.ylabel("Valor")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# GRAFICA ANGULO-EJE DEL CUATERNION

plt.figure(figsize=(10,6))

plt.plot(t, np.degrees(theta_q), label="Ángulo quaternion [deg]")
plt.plot(t, eje_q[:,0], label="Eje x")
plt.plot(t, eje_q[:,1], label="Eje y")
plt.plot(t, eje_q[:,2], label="Eje z")

plt.title("Representación ángulo-eje")
plt.xlabel("Tiempo [s]")
plt.ylabel("Valor")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# GIF SOLO TRAYECTORIA XY

fig2, ax = plt.subplots(figsize=(6,6))

line, = ax.plot([], [], lw=2)
point, = ax.plot([], [], "ro")

ax.set_xlim(min(PE), max(PE))
ax.set_ylim(min(PN), max(PN))

ax.set_xlabel("Este")
ax.set_ylabel("Norte")
ax.set_title("Trayectoria XY")

text = ax.text(0.02,0.95,"", transform=ax.transAxes)

def update(i):

    line.set_data(PE[:i], PN[:i])
    point.set_data([PE[i]], [PN[i]])

    text.set_text(
        f"t={t[i]:.2f}s\n"
        f"N={PN[i]:.2f}\n"
        f"E={PE[i]:.2f}"
    )

    return line, point, text

# reducir frames para GIF
step = max(1, len(t)//250)

ani = FuncAnimation(
    fig2,
    update,
    frames=range(0,len(t),step),
    interval=20
)

# guardar gif
ani.save("trayectoria.gif", writer="pillow", fps=20)

plt.show()
import numpy as np
import sympy as sp

from lib.direct_kinematic import Link, DirectKinematic, joint_angles_subs

q1, q2 = sp.symbols('q1 q2')
q3, q4 = sp.symbols('q3 q4')


j0 = Link([q1, 45, 15, np.pi/2])
j1 = Link([q2, 0, 59, np.pi])
j2 = Link([q3, 0, 13, -np.pi/2])

dk = DirectKinematic([j0, j1, j2])

initial_guess = [0.1, 0.1, 0.1]
theta_i = initial_guess

epsilon = 1e-3
error = 1

desired_pose = [
	67, 0, 40
]

pose_error = np.array([.0 for _ in range(3)])


def n(r):
	return np.arctan2(
		np.sin(r),
		np.cos(r)
	)


generic_jacobian = dk.get_generic_jacobian()


while error >= epsilon:
	jacobian = np.array(
		generic_jacobian.subs(
			joint_angles_subs(theta_i)
		).evalf(),
		dtype=np.float64
	)

	jacobian_pinv = np.linalg.pinv(jacobian)
	current_pose = dk.get_htm(theta_i)

	for i in range(3):
		pose_error[i] = desired_pose[i] - current_pose[i, 3]

	theta_i += (jacobian_pinv @ pose_error)
	error = np.linalg.norm(pose_error)

	print(current_pose[:3, 3])


t = n(
	np.array(theta_i, dtype=np.float64)
)

print(' ')
print(f'Theta {t}')
print(' ')

sp.print_jscode(dk.get_htm(t))

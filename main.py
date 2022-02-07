import numpy as np
import sympy as sp
from lib.direct_kinematic import Link, DirectKinematic, joint_angles_subs, inverse_transformation
from lib.frame import translation_matrix, x_rotation_matrix


def n(r):
	return np.arctan2(
		np.sin(r),
		np.cos(r)
	)


q1, q2 = sp.symbols('q1 q2')
q3, q4 = sp.symbols('q3 q4')
q5, q6 = sp.symbols('q3 q4')

l1, l2 = sp.symbols('l1 l2')
a1, a2, a3 = sp.symbols('a1 a2 a3')

j0 = Link([q1, 10, 0, sp.pi/2])
j1 = Link([q2, 0, 10, 0])

dk = DirectKinematic([j0, j1])

sp.print_jscode(
	dk.get_htm([-np.pi, 0])
)

initial_guess = [.2, .1]
theta_i = initial_guess

epsilon_vb = 1e-3
error = 1

desired_pose = [-10, 0, 10, 0, 0, 0]  # d
pose_error = np.array([0, 0, 0, 0, 0, 0])

generic_jacobian = dk.get_generic_jacobian()

while error >= epsilon_vb:
	jacobian = np.array(
		generic_jacobian.subs(
			joint_angles_subs(theta_i)
		).evalf(),
		dtype=np.float64
	)
	
	current_pose = dk.get_htm(theta_i)
	
	for i in range(3):
		pose_error[i] = desired_pose[i] - current_pose[i, 3]
		pose_error[i+3] = desired_pose[i+3] - current_pose[i, i]

	jacobian_pinv = np.linalg.pinv(jacobian)
	theta_i += jacobian_pinv @ pose_error
	
	error = np.sum(np.abs(pose_error))
	print(error)

t = n(
	np.array(theta_i, dtype=np.float64)
)

print(' ')
print(f'Theta {t}')
print(' ')

sp.print_jscode(dk.get_htm(t))

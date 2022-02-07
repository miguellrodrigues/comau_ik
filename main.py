import numpy as np
import sympy as sp
from lib.direct_kinematic import Link, DirectKinematic, joint_angles_subs, inverse_transformation, omega
from lib.frame import arbitrary_vector_rotation_matrix, x_rotation_matrix, translation_matrix, g_inv


def n(r):
	return np.arctan2(
		np.sin(r),
		np.cos(r)
	)


q1, q2 = sp.symbols('q1 q2')

l1, l2 = sp.symbols('l1 l2')

j0 = Link([q1, 0, 1, 0])
j1 = Link([q2, 0, 1, 0])

dk = DirectKinematic([j0, j1])

sp.print_jscode(
	dk.get_htm([np.pi, 0])
)

# rot = arbitrary_vector_rotation_matrix(np.radians(30), [1, 0, 0])

# w, theta = omega(rot)
#
# g = g_inv(theta, w)
# v = g @ np.array([0, 0, 1])

initial_guess = [np.pi, 0]
theta_i = initial_guess

epsilon_vb = 1e-3
error = 1

desired_pose = sp.Matrix([
	[-1, 0, 0, -2],
	[0, -1, 0, 0],
	[0, 0, -1, 0],
	[0, 0, 0, 1]
])

generic_jacobian = dk.get_generic_jacobian()
generic_inverse_htm = inverse_transformation(dk.get_generic_htm())

while error >= epsilon_vb:
	Tbs = generic_inverse_htm.subs(joint_angles_subs(theta_i)).evalf()
	Tbd = Tbs @ desired_pose

	rot = Tbd[:3, :3]
	
	w, theta = omega(rot)
	
	# sp.print_jscode(log_tbd)
	
	g = np.array(g_inv(theta, w))
	v = np.array(g @ Tbd[:3, 3])
	
	w = w[1]
	print(w, v)
	
	jacobian = np.array(
		generic_jacobian.subs(
			joint_angles_subs(theta_i)
		).evalf(),
		dtype=np.float64
	)
	jacobian_pinv = np.linalg.pinv(jacobian)

	#
	# current_pose = dk.get_htm(theta_i)
	#
	# for i in range(3):
	# 	pose_error[i] = desired_pose[i] - current_pose[i, 3]
	# 	pose_error[i+3] = desired_pose[i+3] - current_pose[i, i]
	#

	# theta_i += jacobian_pinv @ s
	#
	# error = np.sum(np.abs(pose_error))
	# print(error)

t = n(
	np.array(theta_i, dtype=np.float64)
)

print(' ')
print(f'Theta {t}')
print(' ')

sp.print_jscode(dk.get_htm(t))

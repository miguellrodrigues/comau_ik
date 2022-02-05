import numpy as np
import sympy as sp

from lib.direct_kinematic import Link, DirectKinematic

q1, q2 = sp.symbols('q1 q2')
q3, q4 = sp.symbols('q3 q4')
q5, q6 = sp.symbols('q5 q6')


j0 = Link([q1, 450, 150, np.pi/2])
j1 = Link([q2, 0, 590, np.pi])
j2 = Link([q3, 0, 130, -np.pi/2])
# j3 = Link([q4, 647, 0, -np.pi/2])
# j4 = Link([q5, 0, 0, np.pi/2])
# j5 = Link([q6, 0, 10, 0])

dk = DirectKinematic([j0, j1, j2])

# sp.print_jscode(dk.get_htm([
# 	0,
# 	np.pi,
# 	0,
# ]).evalf())

# sp.print_jscode(dk.get_transformation(1, 2))

# joint_values = np.array([
# 	np.pi,
# 	np.pi
# ])
#

# jacobian = sp.matrix2numpy(dk.get_jacobian([0, 0, 0, 0, 0, 0]), dtype=np.float64)
#
# print(np.linalg.pinv(jacobian))

initial_guess = [0, 3.18, 0]
theta_i = initial_guess

epsilon = 1e-2
error = 1

desired_pose = [
	-570, 0, 450,
	1, 1, 1
]

pose_error = [.0 for _ in range(6)]

while error > epsilon:
	jacobian = sp.matrix2numpy(dk.get_jacobian(theta_i), dtype=np.float64)
	jacobian_pinv = np.linalg.pinv(jacobian)
	
	current_pose = dk.get_htm(theta_i)

	for i in range(3):
		pose_error[i] = desired_pose[i] - current_pose[i, 3]
	
	pose_error[3] = 0
	pose_error[4] = 0
	pose_error[5] = 0

	print(pose_error)

	theta_i += .1 * (jacobian_pinv @ pose_error)
	error = np.sum(np.abs(pose_error))

print(' ')
print(theta_i)
print(' ')

sp.print_jscode(dk.get_htm(theta_i))

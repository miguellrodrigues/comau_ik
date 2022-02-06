import numpy as np
import sympy as sp

from lib.direct_kinematic import Link, DirectKinematic

q1, q2 = sp.symbols('q1 q2')
q3, q4 = sp.symbols('q3 q4')
q5, q6 = sp.symbols('q3 q4')

j0 = Link([q1, 45, 15, np.pi/2])
j1 = Link([q2, 0, 59, np.pi])
j2 = Link([q3, 0, 13, -np.pi/2])
#
# j3 = Link([q4, 64, 0, -np.pi/2])
# j4 = Link([q5, 0, 0, np.pi/2])
# j5 = Link([q6, 0, 10, 0])

dk = DirectKinematic([j0, j1, j2])

sp.print_jscode(dk.get_htm([
	np.pi/2,
	np.pi/2,
	0,
]).evalf())

#
# t01 = dk.get_transformation(0, 1)
# sp.print_jscode(t01)
# dk.get_jacobian([q1, q2])
#
# sp.print_jscode(sp.simplify(t01))

# joint_values = np.array([
# 	np.pi,
# 	np.pi
# ])
#

#
# print(np.linalg.pinv(jacobian))
#

initial_guess = [2, 2, 0]
theta_i = initial_guess

epsilon = .1
error = 1

desired_pose = [
	0, 18, 117,
	0, 0, 0
]

pose_error = [.0 for _ in range(6)]


def n(r):
	return np.arctan2(
		np.sin(r),
		np.cos(r)
	)


while error > epsilon:
	jacobian = sp.matrix2numpy(dk.get_jacobian(theta_i), dtype=np.float64)
	jacobian_pinv = np.linalg.pinv(jacobian)

	current_pose = dk.get_htm(theta_i)

	for i in range(3):
		pose_error[i] = desired_pose[i] - current_pose[i, 3]

	pose_error[3] = desired_pose[3] - current_pose[0, 0]
	pose_error[4] = desired_pose[4] - current_pose[1, 1]
	pose_error[5] = desired_pose[5] - current_pose[2, 2]

	# print(current_pose[:3, 3])

	theta_i += (jacobian_pinv @ pose_error)
	error = np.sum(np.abs(pose_error[:3]))

	print(current_pose[:3, 3])


t = n(
	np.array(theta_i, dtype=np.float64)
)

print(' ')
print(f'Theta {t}')
print(' ')

sp.print_jscode(dk.get_htm(t))

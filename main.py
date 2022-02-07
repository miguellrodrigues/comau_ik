import numpy as np
import sympy as sp

from lib.direct_kinematic import Link, DirectKinematic, joint_angles_subs, inverse_transformation
from lib.frame import translation_matrix


def n(r):
	return np.arctan2(
		np.sin(r),
		np.cos(r)
	)


q1, q2 = sp.symbols('q1 q2')
q3, q4 = sp.symbols('q3 q4')
q5, q6 = sp.symbols('q3 q4')

l1, l2 = sp.symbols('l1 l2')

j0 = Link([q1, 0, 10, 0])
j1 = Link([q2, 0, 10, 0])

dk = DirectKinematic([j0, j1])

initial_guess = [.0, .0]
theta_i = initial_guess

epsilon_vb = 1
error = 1

desired_pose = translation_matrix(5, 10, 0)

vb = np.eye(4)

generic_jacobian = dk.get_generic_jacobian()
generic_htm_inverse = inverse_transformation(dk.get_generic_htm())

while error >= epsilon_vb:
	Tbs = generic_htm_inverse.subs(joint_angles_subs(theta_i)).evalf()
	Tbd = sp.matrix2numpy(
		Tbs @ desired_pose,
		dtype=np.float64
	)
	
	d_tbd = np.diag(Tbd)
	v = np.log(d_tbd)
	
	e = np.sum(v**2)

	jacobian = np.array(
		generic_jacobian.subs(
			joint_angles_subs(theta_i)
		).evalf(),
		dtype=np.float64
	)

	jacobian_pinv = np.linalg.pinv(jacobian)
	theta_i += jacobian_pinv.ravel() @ v
	print(theta_i)
	


t = n(
	np.array(theta_i, dtype=np.float64)
)

print(' ')
print(f'Theta {t}')
print(' ')

sp.print_jscode(dk.get_htm(t))

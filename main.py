import numpy as np
import sympy as sp

from lib.direct_kinematic import Link, DirectKinematic

q1, q2 = sp.symbols('q1 q2')
q3, q4 = sp.symbols('q3 q4')
q5, q6 = sp.symbols('q5 q6')


j0 = Link([q1, 450, 150, np.pi/2])
j1 = Link([q2, 0, 590, np.pi])
j2 = Link([q3, 0, 130, -np.pi/2])
j3 = Link([q4, 647, 0, -np.pi/2])
j4 = Link([q5, 0, 0, np.pi/2])
j5 = Link([q6, 0, 10, 0])

dk = DirectKinematic([j0, j1, j2, j3, j4, j5])

# sp.print_jscode(dk.get_htm([
# 	0,
# 	0,
# 	0,
# 	0,
# 	0,
# 	0
# ]).evalf())

# sp.print_jscode(dk.get_transformation(1, 2))

# joint_values = np.array([
# 	np.pi,
# 	np.pi
# ])
#
jacobian = sp.matrix2numpy(dk.get_jacobian([0, 0, 0, 0, 0, 0]), dtype=np.float64)

print(np.linalg.pinv(jacobian))

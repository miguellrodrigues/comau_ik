import numpy as np
import sympy as sp

from lib.direct_kinematic import Link, DirectKinematic

q1, q2, q3, q4 = sp.symbols('q1 q2 q3 q4')

# l2 = .2
# l3 = .15
# l4 = .1
# l5 = .08
#
# d3 = .1

j0 = Link([q1, 0, .2, 0])
j1 = Link([q2, 0, .15, np.pi])
j2 = Link([q3, .2, 0, 0])
j3 = Link([q4, .08, 0, 0])

dk = DirectKinematic([j0, j1, j2, j3])

# sp.print_jscode(dk.get_htm([
# 	np.pi/4,
# 	np.pi/4,
# 	0,
# 	np.pi / 2,
# ]).evalf())

joint_values = np.array([
	np.pi/4,
	np.pi/4,
	0,
	np.pi / 2,
])

sp.pprint(dk.get_jacobian(joint_values))

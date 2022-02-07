import numpy as np
import sympy as sp

from lib.frame import x_rotation_matrix, z_rotation_matrix, translation_matrix

"""
This module contains the functions to compute the direct kinematic of the
different robots.

tm -> transformation matrix
htm -> homogeneous transformation matrix
dhp -> denavit-hartenberg parameters
"""


class Link:
	"""
	This class represents a link of the robot.
	
	Attributes:
		:attr:'`dhp` (list): Denavit-Hartenberg parameters.
	"""
	
	def __init__(self, dhp, link_type='rotational'):
		self.dhp = dhp
		self.link_type = link_type
		self.transformation_matrix = self.compute_tm()
	
	def compute_tm(self):
		rz = z_rotation_matrix(self.dhp[0])
		tz = translation_matrix(0, 0, self.dhp[1])
		tx = translation_matrix(self.dhp[2], 0, 0)
		rx = x_rotation_matrix(self.dhp[3])
		
		return rz @ tz @ tx @ rx
	
	def get_tm(self, joint_angles=None):
		tm = self.transformation_matrix
		
		if joint_angles is not None:
			tm = tm.subs(joint_angles)
		
		return tm
	
	"""
		This function computes the homogeneous transformation matrix of the
		link.
		
		:param: `i` (int): Index of the DH-Parameter to update.
		:param: `v` (float): The new value of the parameter.
	"""
	
	def update_tm(self, i, v):
		self.dhp[i] = v
		
		self.transformation_matrix = self.compute_tm()


"""
	Compute the homogeneous transformation matrix
	from link i (start) to link j (end).
"""


def compute_transformation(links, start, end, joint_angles=None):
	if end == 0:
		return sp.eye(4)
	
	tm = links[start].get_tm(joint_angles)
	
	for i in range(start + 1, end):
		tm_i = links[i].get_tm(joint_angles)
		tm = tm @ tm_i
	
	return tm
	
	
def inverse_transformation(tm):
	inverse = sp.eye(4)
	
	transposed = tm.T
	
	v_t = tm[:3, 3]  # translation vector
	v_r = transposed[:3, :3]  # rotation matrix
	
	# multiply each column of vr with vt
	
	v_t_i = -v_r @ v_t
	
	inverse[:3, 3] = v_t_i
	inverse[:3, :3] = v_r
	
	return inverse


def joint_angles_subs(joint_angles):
	return [(f'q{i + 1}', joint_angles[i]) for i in range(len(joint_angles))]


def norm(r):
	return sp.atan2(sp.sin(r) ,sp.cos(r))


def omega(r):
	if r == np.eye(3):
		return r, np.pi
	
	tr = np.sum(r[i, i] for i in range(3))
	
	theta = sp.symbols('theta')
	
	e = sp.Eq(
		1 + 2 * sp.cos(theta),
		tr
	)
	
	s = sp.solvers.solve(e, theta)
	
	t = next(s for s in s if s != 0)
	
	if tr == -1:
		omega_hat = (1 / (np.sqrt(2 * (1 + r[2, 2]))) * np.array([r[0, 2], r[1, 2], 1 + r[2, 2]]))
	else:
		omega_hat = (1 / (2 * sp.sin(t))) * (r - r.T)
	
	return omega_hat, norm(t)


class DirectKinematic:
	def __init__(self, links):
		self.links = links
		self.generic_htm = compute_transformation(links, 0, len(self.links))
		self.generic_jacobian = self.compute_generic_jacobian()  # self.get_jacobian([])
	
	def compute_generic_jacobian(self):
		htm = self.generic_htm
		
		end_effector_pos = htm[:3, 3]

		# derive each position (x, y, z) with respect of all thetas
		# J = Matrix 2xjoints
		
		jacobian = sp.Matrix(sp.symarray('j', (6, len(self.links))))
		r0 = sp.Matrix([0, 0, 1])
		
		for i in range(len(self.links)):
			d_p_qi = sp.diff(end_effector_pos, f'q{i + 1}').T

			for j in range(3):
				jacobian[j, i] = d_p_qi[j]
			
			jacobian[3:, i] = r0
			
			transformation = self.links[i].get_tm()
			r0 = transformation[:3, 2]
	
		return jacobian
	
	def get_generic_jacobian(self):
		return self.generic_jacobian
	
	"""
			Returns the robot homogeneous transformation matrix in generic form.
	"""
	
	def get_generic_htm(self):
		return self.generic_htm
	
	"""
		Returns the homogeneous transformation matrix of the robot in the
		given configuration (from link 'start' to link 'end').
	"""
	
	def get_transformation(self, start, end, joint_angles=None):
		tf = compute_transformation(self.links, start, end, joint_angles)
		return tf
	
	"""
		Return the robot homogeneous transformation matrix in the given
		joint_angles.
	"""
	
	def get_htm(self, joint_angles):
		return self.generic_htm.subs(
			joint_angles_subs(joint_angles)
		).evalf()
	
	"""
		Compute the Jacobian matrix considering the rotations.
	"""
	
	def get_jacobian(self):
		htm = self.get_generic_htm()
		
		# for rotational joints, the Jacobian is [z_{i - 1} * (p - p_{i - 1}); z_{i - 1}]
		# for prismatic joints, the jacobian is [z_{i - 1}; 0]
		
		# z_{i - 1} is te third column of 0R_{i - 1}
		# p is the position of the end-effector
		# p_{i - 1} is the position of the i-th joint
		
		# Compute the Jacobian matrix
		
		len_joints = len(self.links)
		
		j = sp.Matrix(sp.symarray('j', (6, len_joints)))
		
		# position of the end-effector
		p = htm[:3, 3]
		
		p_i_minus_1 = sp.Matrix([0, 0, 0])
		z_i_minus_1 = sp.Matrix([0, 0, 1])
		
		for i in range(1, len_joints + 1):
			p_diff = (p - p_i_minus_1).T
			j_r = np.cross(z_i_minus_1.T, p_diff)

			stack = np.vstack((j_r[0], z_i_minus_1.T))
			
			j[:, i - 1] = stack.flatten()
			
			transformation = self.links[i - 1].get_tm()
			# transformation = self.get_transformation(i-1, i, joint_angles_subs(joint_angles))
			
			p_i_minus_1 = transformation[:3, 3]
			z_i_minus_1 = transformation[:3, 2]
		
		return j

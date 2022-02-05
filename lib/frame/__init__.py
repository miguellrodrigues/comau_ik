import numpy as np
import sympy as sp


def x_rotation_matrix(yaw):
	"""
	Rotation matrix around the x axis
	"""
	return sp.Matrix([[1, 0, 0, 0], [0, sp.cos(yaw), -sp.sin(yaw), 0], [0, sp.sin(yaw), sp.cos(yaw), 0], [0, 0, 0, 1]])


def y_rotation_matrix(pitch):
	"""
	Rotation matrix around the y axis
	"""
	return sp.Matrix(
		[[sp.cos(pitch), 0, sp.sin(pitch), 0], [0, 1, 0, 0], [-sp.sin(pitch), 0, sp.cos(pitch), 0], [0, 0, 0, 1]])


def z_rotation_matrix(roll):
	"""
	Rotation matrix around the z axis
	"""
	return sp.Matrix(
		[[sp.cos(roll), -sp.sin(roll), 0, 0], [sp.sin(roll), sp.cos(roll), 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])


def x_y_z_rotation_matrix(yaw, pitch, roll):
	"""
	Rotation matrix around the x, y and z axis
	"""
	return x_rotation_matrix(yaw) @ y_rotation_matrix(pitch) @ z_rotation_matrix(roll)


def arbitrary_vector_rotation_matrix(theta, v):
	"""
	Rotation matrix around an arbitrary vector
	"""
	
	v = v / np.linalg.norm(v)
	v_x = sp.Matrix([[0, -v[2], v[1], 0], [v[2], 0, -v[0], 0], [-v[1], v[0], 0, 0], [0, 0, 0, 1]])
	
	return sp.eye(4) + v_x + (v_x @ v_x) * (1 - sp.cos(theta))


def translation_matrix(dx, dy, dz):
	return sp.Matrix([[1, 0, 0, dx], [0, 1, 0, dy], [0, 0, 1, dz], [0, 0, 0, 1]])


class Frame:
	def __init__(self, x, y, z, yaw=0, pitch=0, roll=0, name=''):
		self.position = translation_matrix(x, y, z)
		
		self.rotation = x_y_z_rotation_matrix(yaw, pitch, roll)
		
		self.name = name
	
	def translate(self, dx, dy, dz):
		self.position = translation_matrix(dx, dy, dz) @ self.position
		
		return self.position
	
	def rotate(self, yaw, pitch, roll):
		self.rotation = x_y_z_rotation_matrix(yaw, pitch, roll) @ self.rotation
	
	def rotate_around_arbitrary_vector(self, theta, v):
		self.rotation = arbitrary_vector_rotation_matrix(theta, v) @ self.rotation
	
	def get_x_component(self):
		return self.position[0, 3]
	
	def get_y_component(self):
		return self.position[1, 3]
	
	def get_z_component(self):
		return self.position[2, 3]
	
	def rotation_matrix(self):
		return self.rotation
	
	def rotation_to(self, other):
		yaw = sp.atan2(other.rotation[2, 1], other.rotation[2, 2]) - sp.atan2(self.rotation[2, 1], self.rotation[2, 2])
		pitch = sp.atan2(other.rotation[2, 0], other.rotation[2, 2]) - sp.atan2(self.rotation[2, 0], self.rotation[2, 2])
		roll = sp.atan2(other.rotation[1, 0], other.rotation[0, 0]) - sp.atan2(self.rotation[1, 0], self.rotation[0, 0])

		return sp.Matrix([yaw, pitch, roll])

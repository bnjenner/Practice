import sys
import random
import numpy as np

# Computational Graph Class
class CompGraph():

	def __init__(self, a: float, b: float):
		self.A = a
		self.B = b
		self.n = None
		self.iter = 10000
		self.rate = 0.0001
		self.thresh = 0.00001

		'''
		True_A = 2
		True_B = 3
		Loss = Sum Squared Error
		'''

	# Evaluation of Graph
	def __eval(self, c):
		return (self.A * self.B) + (self.B * (c**2))

	# Forward Propagation
	def __forward_prop(self, x):
		_x = np.zeros(self.n)
		for i in range(self.n):
			_x[i] = self.__eval(x[i])
		return _x

	# Backward Propagation
	def __backward_prop(self, x, Y, err):
		_A, _B = 0, 0
		for i in range(self.n):
			_A += err[i] * self.B
			_B += err[i] * (self.A + (x[i]**2))
		return _A, _B

	# Public Training Function
	def train(self, x, Y):
		
		self.n = len(x)
		x0 = np.asarray(x)
		Y0 = np.asarray(Y)

		# Forward Propagation and Loss
		for i in range(self.iter):
			_x = self.__forward_prop(x0)
			_err = Y0 - _x
			_A, _B = self.__backward_prop(_x, Y0, _err)
			_dA = (self.rate * _A)
			_dB = (self.rate * _B)
			self.A += _dA
			self.B += _dB

			if abs(_dA) + abs(_dB) < self.thresh:
				break


# Main
def main():

	# Data
	x, Y = [], []
	
	# Read in Data
	infile = sys.argv[1]
	with open(infile, "r") as file:
		file.readline() # Skip header
		for l in file:
			cols = l.strip().split("\t")
			x.append(float(cols[0]))
			Y.append(float(cols[1]))


	# Create Graph Object
	graph = CompGraph(random.random(), random.random())

	print("// TARGET PARAMETERS:")
	print("    A = 2\n    B = 3")
	print("// INIT PARAMETERS:")
	print(f"    A = {graph.A}\n    B = {graph.B}")
	
	# Train Graph
	graph.train(x, Y)

	print("// FINAL PARAMETERS:")
	print(f"    A = {graph.A}\n    B = {graph.B}")



if __name__ == "__main__":
	main()
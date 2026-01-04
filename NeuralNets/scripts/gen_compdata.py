import random

def comp(c):
	A = 2
	B = 3 
	return (A * B) + (B * (c**2))



num = 100
random_x = [random.random() for _ in range(num)]
Y = [comp(c) for c in random_x]

print("X\tY")
for i in range(num):
	print(random_x[i], Y[i], sep = "\t")
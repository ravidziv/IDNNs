import math


def sampleStandardDeviation(x):
	"""calculates the sample standard deviation"""
	sumv = 0.0
	for i in x:
		sumv += (i) ** 2
	return math.sqrt(sumv / (len(x) - 1))


def pearson(x, y):
	"""calculates the PCC"""
	scorex, scorey = [], []
	for i in x:
		scorex.append((i) / sampleStandardDeviation(x))
	for j in y:
		scorey.append((j) / sampleStandardDeviation(y))
	# multiplies both lists together into 1 list (hence zip) and sums the whole list
	return (sum([i * j for i, j in zip(scorex, scorey)])) / (len(x) - 1)

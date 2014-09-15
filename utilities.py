import sys, os

def isfloat(x):
	"""
	Check if argument is float
	"""
	try:
		a = float(x)
	except ValueError:
		return False
	else:
		return True

def isint(x):
	"""
	Check if argument is int
	"""
	try:
		a = float(x)
		b = int(a)
	except ValueError:
		return False
	else:
		return a == b

def isNum(x):
	"""
	Check if string argument is numerical
	"""
	return isfloat(x) or isint(x)

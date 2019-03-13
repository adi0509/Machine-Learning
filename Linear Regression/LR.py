import numpy as np
import matplotlib.pyplot as plt

def plot(x,y,b,m):

	#plot all the data points in the scatter plot
	plt.scatter(x,y,color="m", marker="o", s=20)
	
	y_pred = m*x + b

	plt.plot(x,y_pred, color='g')

	plt.xlabel('x')
	plt.ylabel('y')
	# plt.show()
	plt.show(block=False)
	plt.pause(1.2)
	plt.close()

def calculateCoeff(x,y,intial_b,intial_m, num_iteration, learning_rate):

	b = intial_b
	m = intial_m
	for i in range(num_iteration):
		[b,m] = gradient(x,y,b,m, learning_rate)
		plot(x,y,b,m)
		# print('slope: ', m, '  intercept: ', b)
	return [b,m]

def gradient(x,y, b, m, learning_rate):
	b_gradient = 0
	m_gradient = 0
	n = float(len(x))
	#now the theory here is, we will differentiate w.r.t. b(which is intercept) and m(which is slope) and 
	#subtract it to the previous value such that we can obtain where we have to change the value of b & m such
	#we can reach to minimum value. If gradient we get is negaative then it will be added to the previous value
	# and if it is negative then it will subtracted from the previos values
	for i in range(len(x)):
		#in the below equation we are subtracting MEAN SQUARED ERROR and differentiate it w.r.t 'b' & 'm'
		b_gradient += -((2/n) *(y[i] - (m*x[i]+b)))
		m_gradient += -((2/n) * x[i] * (y[i] - (m*x[i]+b)))
	new_b = b - (learning_rate*b_gradient)
	new_m = m - (learning_rate*m_gradient)
	print('m: ', m, '  b: ', b, ' b_gra: ', b_gradient, ' m_gra: ', m_gradient)
	return [new_b, new_m]



def main(): 
	#sample data
    x = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]) 

    y = np.array([1, 3, 2, 5, 7, 8, 8, 9, 10, 12]) 
    intial_b = 0
    intial_m = -1
    num_iteration = 1000
    learning_rate = 0.001

    plot(x,y,intial_b,intial_m)
    [b,m] = calculateCoeff(x,y, intial_b, intial_m, num_iteration, learning_rate)
    plot(x,y,b,m)


if __name__=="__main__":
	main()
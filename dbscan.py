from sklearn.datasets import make_moons
import numpy as np

def distance(a, b):
	return np.linalg.norm(a - b)

# Add outliers to the half-moons
def addNoise(X, y, noiseLevel = 0.01):
	# Amount of noisy points
	amountNoise = int(noiseLevel * len(y))

	# Select random points to add noise
	index = np.random.choice(len(X), size = amountNoise)

	# Add noise
	noise = np.random.random((amountNoise, len(X[0]))) - 0.5
	X[index, :] += noise

def main():
	# moons_X: Data, moon_y: Labels
	moons_X, moon_y = make_moons(n_samples = 2000)
	addNoise(moons_X, moon_y)

if __name__ == "__main__":
	main()

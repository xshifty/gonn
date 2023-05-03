package gonn

func relu(x float64) float64 {
	if x > 0 {
		return x
	}

	return x / 100
}

func reluPrime(x float64) float64 {
	if x > 0 {
		return 1
	}

	return 0.01
}

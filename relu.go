package gonn

const ReluFunction = "relu"

func relu(x float64) float64 {
	if x > 0 {
		return x
	}

	return x / 100
}

func reluDerivative(x float64) float64 {
	if x > 0 {
		return 1
	}

	return 0.01
}

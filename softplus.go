package gonn

import "math"

func softplus(x float64) float64 {
	return math.Log(1 + math.Exp(x))
}

func softplusPrime(x float64) float64 {
	return 1 / (1 + math.Exp(-x))
}

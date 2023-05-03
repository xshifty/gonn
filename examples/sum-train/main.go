package main

import (
	"fmt"
	"github.com/xshifty/gonn"
	"math"
)

func main() {
	fmt.Println("GoNN sum-train demo")

	n := gonn.NewNetwork(gonn.ActivationFunctionRelu, 2, 4, 4, 1)

	inputs := [][]float64{
		{2, 2},
		{0, 1},
		{14, 6},
		{14, 16},
		{math.Pi, math.E},
		{47, 32},
	}

	outputs := [][]float64{
		{4},
		{1},
		{20},
		{30},
		{math.Pi + math.E},
		{79},
	}

	iter := 10000
	fmt.Printf("Training for %d iterations", iter)

	for t := 0; t < iter; t++ {
		for i := range inputs {
			n.Train(inputs[i], outputs[i], 0.00001)
		}
		if t%int(float32(iter)*0.1) == 0 {
			fmt.Printf(".")
		}
	}
	fmt.Println("Done")

	n.Save("sum-relu-checkpoint.gob.gz")
}

package main

import (
	"fmt"
	"github.com/xshifty/gonn"
)

func main() {
	fmt.Println("GoNN train demo")

	n := gonn.NewNetwork(2, 32, 32, 1)

	inputs := [][]float64{
		{2, 2},
		{0, 1},
		{14, 6},
		{14, 16},
	}

	outputs := [][]float64{
		{4.0},
		{1.0},
		{20.0},
		{30.0},
	}

	iter := 1000000
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

	n.Save("checkpoint.gob.gz")
}

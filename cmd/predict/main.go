package main

import (
	"fmt"
	"github.com/xshifty/gonn"
	"math"
)

func main() {
	fmt.Println("GoNN predict demo")

	n := gonn.NewFromCheckpoint("checkpoint.gob.gz")

	inputs := [][]float64{
		{2, 2},
		{0, 1},
		{14, 6},
		{14, 16},
		{7, 7},
		{1, 1},
		{44, 12},
		{1.5, 2.5},
	}

	for i := range inputs {
		p := math.Round(n.Predict(inputs[i])[0])
		fmt.Printf("Predicting %v => %f\n", inputs[i], p)
	}
}

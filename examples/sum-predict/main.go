package main

import (
	"fmt"
	"github.com/xshifty/gonn"
	"math"
)

func main() {
	fmt.Println("GoNN sum-predict demo")

	n := gonn.NewFromCheckpoint("sum-relu-checkpoint.gob.gz")

	inputs := [][]float64{
		{2, 2},
		{0, 1},
		{14, 6},
		{14, 16},
		{7, 7},
		{1, 1},
		{44, 12},
		{1.5, 2.5},
		{6.5, 5.5},
		{math.Pi, math.E},
		{3210, 2238},
	}

	for i := range inputs {
		p := int(math.Round(n.Predict(inputs[i])[0]))
		fmt.Printf("Predicting %v => %d\n", inputs[i], p)
	}
}

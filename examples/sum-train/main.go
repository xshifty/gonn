package main

import (
	"fmt"
	"github.com/xshifty/gonn"
	"math"
)

func main() {
	fmt.Println("GoNN sum-train demo")

	n := gonn.NewNetwork(2, 8, 1)

	n.SetOption(gonn.OptionRandomizeBias, false)
	n.SetOption(gonn.OptionRandomizeWeights, true)

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

	epochs := 100000
	fmt.Printf("Training for %d epochs...", epochs)
	n.Train(inputs, outputs, epochs, 0.0001)
	fmt.Println("Done")
	n.Save("sum-relu-checkpoint.gob.gz")
}

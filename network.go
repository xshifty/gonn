package gonn

import (
	"compress/gzip"
	"encoding/gob"
	"fmt"
	"log"
	"math/rand"
	"os"
)

type synapse struct {
	weight float64
	in     float64
	out    float64
}

func (s *synapse) String() string {
	return fmt.Sprintf("Synapse: %f", s.weight)
}

func (s *synapse) fire(x float64) float64 {
	if x == 0 {
		return 0
	}

	s.in = x
	s.out = s.in * s.weight

	return s.out
}

func (s *synapse) adjust(factor float64) {
	s.weight += factor * s.in
}

type neuron struct {
	synapses []*synapse
	bias     *synapse
	z        float64
	a        float64
}

func (n *neuron) fire(in []float64) float64 {
	if len(in) != len(n.synapses) {
		log.Fatal("Input length does not match synapse length")
	}

	n.a = 0.0
	for i := range n.synapses {
		n.a += n.synapses[i].fire(in[i])
	}
	n.a += n.bias.fire(1.0)

	n.a *= .1

	return n.a
}

func (n *neuron) activate() float64 {
	n.z = relu(n.a)
	return n.z
}

type layer struct {
	neurons []*neuron
}

func (l *layer) output() []float64 {
	out := make([]float64, len(l.neurons))
	for i := range l.neurons {
		out[i] = l.neurons[i].activate()
	}

	return out
}

type network struct {
	layers []*layer
}

type networkData struct {
	Layers []struct {
		Neurons []struct {
			Bias struct {
				Weight float64
				In     float64
				Out    float64
			}
			Synapses []struct {
				Weight float64
				In     float64
				Out    float64
			}
			A float64
			Z float64
		}
	}
}

type Network interface {
	String() string
	Train(in, out []float64, rate float64)
	Predict(in []float64) []float64
	Save(path string)
}

func (n *network) feedForward(input []float64) []float64 {
	in := input
	for i := range n.layers {
		out := make([]float64, len(n.layers[i].neurons))
		for j := range n.layers[i].neurons {
			n.layers[i].neurons[j].fire(in)
			out[j] = n.layers[i].neurons[j].activate()
		}
		in = out
	}

	return n.layers[len(n.layers)-1].output()
}

func (n *network) backPropagate(pred, expected []float64, rate float64) {
	if len(pred) != len(expected) {
		log.Fatal("Prediction length does not match expected length")
	}

	// Calculate output layer error
	predError := make([]float64, len(pred))
	for i := range pred {
		predError[i] = expected[i] - pred[i]
	}

	// Calculate output layer delta
	predDelta := make([]float64, len(pred))
	for i := range pred {
		predDelta[i] = predError[i] * reluPrime(pred[i])
	}

	ll := n.layers[len(n.layers)-1]

	// Calculate output layer adjustment
	for i := range pred {
		for j := range ll.neurons[i].synapses {
			ll.neurons[i].synapses[j].adjust(rate * predDelta[i])
		}
		ll.neurons[i].bias.adjust(rate * predDelta[i])
	}

	// Calculate hidden layer delta
	for i := len(n.layers) - 2; i >= 0; i-- {
		for j := range n.layers[i].neurons {
			n.layers[i].neurons[j].z = relu(n.layers[i].neurons[j].a)
		}

		for j := range n.layers[i].neurons {
			hdelta := 0.0
			for k := range n.layers[i+1].neurons {
				for p := range predDelta {
					hdelta += n.layers[i+1].neurons[k].synapses[j].weight * predDelta[p]
				}
			}
			n.layers[i].neurons[j].a = hdelta
		}
	}

	// Calculate hidden layer adjustment
	for i := len(n.layers) - 2; i >= 0; i-- {
		for j := range n.layers[i].neurons {
			for k := range n.layers[i].neurons[j].synapses {
				n.layers[i].neurons[j].synapses[k].adjust(rate * n.layers[i].neurons[j].a)
			}
			n.layers[i].neurons[j].bias.adjust(rate * n.layers[i].neurons[j].a)
		}
	}
}

// String returns a string representation of the network
func (n *network) String() string {
	s := ""
	for i := range n.layers {
		s += fmt.Sprintf("Layer %d:\n", i)
		for j := range n.layers[i].neurons {
			s += fmt.Sprintf("\tNeuron(%d)\n", j)
			s += fmt.Sprintf("\t\tBias => %f\n", n.layers[i].neurons[j].bias.weight)
			for k := range n.layers[i].neurons[j].synapses {
				s += fmt.Sprintf("\t\tSynapse(%d) => %f\n", k, n.layers[i].neurons[j].synapses[k].weight)
			}
		}
	}

	return s
}

// Train trains the network with the given input and output
func (n *network) Train(in, out []float64, rate float64) {
	prediction := n.feedForward(in)
	n.backPropagate(prediction, out, rate)
}

// Predict predicts the output for the given input
func (n *network) Predict(in []float64) []float64 {
	return n.feedForward(in)
}

// Save saves the network to the given binary gziped file path
func (n *network) Save(path string) {
	f, err := os.OpenFile(path, os.O_CREATE|os.O_WRONLY|os.O_TRUNC, 0644)
	if err != nil {
		log.Fatal(err)
	}
	defer f.Close()

	gz, err := gzip.NewWriterLevel(f, gzip.BestCompression)
	if err != nil {
		log.Fatal(err)
	}
	defer gz.Close()

	nd := networkData{}

	for i := range n.layers {
		nd.Layers = append(nd.Layers, struct {
			Neurons []struct {
				Bias struct {
					Weight float64
					In     float64
					Out    float64
				}
				Synapses []struct {
					Weight float64
					In     float64
					Out    float64
				}
				A float64
				Z float64
			}
		}{})

		for j := range n.layers[i].neurons {
			nd.Layers[i].Neurons = append(nd.Layers[i].Neurons, struct {
				Bias struct {
					Weight float64
					In     float64
					Out    float64
				}
				Synapses []struct {
					Weight float64
					In     float64
					Out    float64
				}
				A float64
				Z float64
			}{})

			nd.Layers[i].Neurons[j].Bias.Weight = n.layers[i].neurons[j].bias.weight
			nd.Layers[i].Neurons[j].Bias.In = n.layers[i].neurons[j].bias.in
			nd.Layers[i].Neurons[j].Bias.Out = n.layers[i].neurons[j].bias.out

			for k := range n.layers[i].neurons[j].synapses {
				nd.Layers[i].Neurons[j].Synapses = append(nd.Layers[i].Neurons[j].Synapses, struct {
					Weight float64
					In     float64
					Out    float64
				}{})

				nd.Layers[i].Neurons[j].Synapses[k].Weight = n.layers[i].neurons[j].synapses[k].weight
				nd.Layers[i].Neurons[j].Synapses[k].In = n.layers[i].neurons[j].synapses[k].in
				nd.Layers[i].Neurons[j].Synapses[k].Out = n.layers[i].neurons[j].synapses[k].out
			}

			nd.Layers[i].Neurons[j].A = n.layers[i].neurons[j].a
			nd.Layers[i].Neurons[j].Z = n.layers[i].neurons[j].z
		}
	}

	err = gob.NewEncoder(gz).Encode(nd)
	if err != nil {
		log.Fatal(err)
	}
}

// NewNetwork creates a new neural network with the given number of inputs and layers
func NewNetwork(in int, layers ...int) *network {
	n := network{
		layers: make([]*layer, len(layers)),
	}
	s := in

	for i := range layers {
		n.layers[i] = &layer{
			neurons: make([]*neuron, layers[i]),
		}
		for j := range n.layers[i].neurons {
			n.layers[i].neurons[j] = &neuron{
				synapses: make([]*synapse, s),
				bias: &synapse{
					weight: rand.Float64(),
					in:     0.0,
					out:    0.0,
				},
				z: 0.0,
				a: 0.0,
			}
			for k := range n.layers[i].neurons[j].synapses {
				n.layers[i].neurons[j].synapses[k] = &synapse{
					weight: rand.Float64(),
					in:     0.0,
					out:    0.0,
				}
			}
		}
		s = layers[i]
	}

	return &n
}

// NewFromCheckpoint loads a network from a binary gziped checkpoint file
func NewFromCheckpoint(path string) *network {
	f, err := os.OpenFile(path, os.O_RDONLY, 0644)
	if err != nil {
		log.Fatal(err)
	}
	defer f.Close()

	gz, err := gzip.NewReader(f)
	if err != nil {
		log.Fatal(err)
	}

	nd := networkData{}
	err = gob.NewDecoder(gz).Decode(&nd)
	if err != nil {
		log.Fatal(err)
	}

	n := network{
		layers: make([]*layer, len(nd.Layers)),
	}

	for i := range nd.Layers {
		n.layers[i] = &layer{
			neurons: make([]*neuron, len(nd.Layers[i].Neurons)),
		}
		for j := range n.layers[i].neurons {
			n.layers[i].neurons[j] = &neuron{
				synapses: make([]*synapse, len(nd.Layers[i].Neurons[j].Synapses)),
				bias: &synapse{
					weight: nd.Layers[i].Neurons[j].Bias.Weight,
					in:     nd.Layers[i].Neurons[j].Bias.In,
					out:    nd.Layers[i].Neurons[j].Bias.Out,
				},
				z: nd.Layers[i].Neurons[j].Z,
				a: nd.Layers[i].Neurons[j].A,
			}
			for k := range n.layers[i].neurons[j].synapses {
				n.layers[i].neurons[j].synapses[k] = &synapse{
					weight: nd.Layers[i].Neurons[j].Synapses[k].Weight,
					in:     nd.Layers[i].Neurons[j].Synapses[k].In,
					out:    nd.Layers[i].Neurons[j].Synapses[k].Out,
				}
			}
		}
	}

	return &n
}

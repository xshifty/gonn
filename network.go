package gonn

import (
	"compress/gzip"
	"encoding/gob"
	"log"
	"os"
)

type synapse struct {
	weight float64
	in     float64
	out    float64
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

	n.z = 0.0
	for i := range n.synapses {
		n.z += n.synapses[i].fire(in[i])
	}
	n.z += n.bias.fire(1.0)

	return n.z
}

func (n *neuron) activate(af string) float64 {
	if af == ActivationFunctionRelu {
		n.a = relu(n.z)
	}
	if af == ActivationFunctionSigmoid {
		n.a = sigmoid(n.z)
	}

	return n.a
}

type layer struct {
	neurons []*neuron
}

func (l *layer) output(af string) []float64 {
	out := make([]float64, len(l.neurons))
	for i := range l.neurons {
		out[i] = l.neurons[i].activate(af)
	}

	return out
}

type network struct {
	activationFunction string
	layers             []*layer
}

type Network interface {
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
			out[j] = n.layers[i].neurons[j].activate(n.activationFunction)
		}
		in = out
	}

	return n.layers[len(n.layers)-1].output(n.activationFunction)
}

func (n *network) backPropagate(prd, expctd []float64, rate float64) {
	if len(prd) != len(expctd) {
		log.Fatal("Prediction length does not match expected length")
	}

	errs := make([]float64, len(prd))
	for i := range prd {
		errs[i] = expctd[i] - prd[i]
	}

	deltas := make([]float64, len(prd))
	for i := range prd {
		if n.activationFunction == ActivationFunctionRelu {
			deltas[i] = errs[i] * reluDerivative(prd[i])
		}
		if n.activationFunction == ActivationFunctionSigmoid {
			deltas[i] = errs[i] * sigmoidDerivative(prd[i])
		}
	}

	ll := n.layers[len(n.layers)-1]
	for i := range prd {
		for j := range ll.neurons[i].synapses {
			ll.neurons[i].synapses[j].adjust(rate * deltas[i])
		}
		ll.neurons[i].bias.adjust(rate * deltas[i])
	}

	for i := len(n.layers) - 2; i >= 0; i-- {
		for j := range n.layers[i].neurons {
			hdelta := 0.0
			for _, h := range n.layers[i+1].neurons {
				for p := range deltas {
					hdelta += h.synapses[j].weight * deltas[p]
				}
			}
			for k := range n.layers[i].neurons[j].synapses {
				n.layers[i].neurons[j].synapses[k].adjust(rate * hdelta)
			}
			n.layers[i].neurons[j].bias.adjust(rate * hdelta)
		}
	}
}

// Train trains the network with the given input and output
func (n *network) Train(in, out []float64, rate float64) {
	p := n.feedForward(in)
	n.backPropagate(p, out, rate)
}

// Predict predicts the output for the given input
func (n *network) Predict(in []float64) []float64 {
	return n.feedForward(in)
}

// Save saves the network to the given binary gziped file path
func (n *network) Save(path string) {
	if err := os.Remove(path); err != nil && !os.IsNotExist(err) {
		log.Fatal(err)
	}

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

	if err := gob.NewEncoder(gz).Encode(buildNetworkDto(n)); err != nil {
		log.Fatal(err)
	}
}

// NewNetwork creates a new neural network with the given number of inputs and layers
func NewNetwork(af string, in int, layers ...int) *network {
	n := network{
		activationFunction: af,
		layers:             make([]*layer, len(layers)),
	}
	s := in

	for i := range layers {
		n.layers[i] = &layer{
			neurons: make([]*neuron, layers[i]),
		}
		for j := range n.layers[i].neurons {
			n.layers[i].neurons[j] = &neuron{
				synapses: make([]*synapse, s),
				bias:     &synapse{weight: 1.0},
			}
			for k := range n.layers[i].neurons[j].synapses {
				n.layers[i].neurons[j].synapses[k] = &synapse{weight: 1.0}
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

	return NewNetworkFromData(&nd)
}

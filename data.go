package gonn

type synapseData struct {
	Weight float64
	In     float64
	Out    float64
}

type neuronData struct {
	Bias     synapseData
	Synapses []synapseData
	Z        float64
	A        float64
}
type layerData struct {
	Neurons []neuronData
}
type networkData struct {
	ActivationFunction string
	Layers             []layerData
}

func buildNetworkDto(n *network) *networkData {
	nd := networkData{
		ActivationFunction: n.activationFunction,
	}

	for i := range n.layers {
		nd.Layers = append(nd.Layers, layerData{})

		for j := range n.layers[i].neurons {
			nd.Layers[i].Neurons = append(nd.Layers[i].Neurons, neuronData{})

			nd.Layers[i].Neurons[j].Bias.Weight = n.layers[i].neurons[j].bias.weight
			nd.Layers[i].Neurons[j].Bias.In = n.layers[i].neurons[j].bias.in
			nd.Layers[i].Neurons[j].Bias.Out = n.layers[i].neurons[j].bias.out

			for k := range n.layers[i].neurons[j].synapses {
				nd.Layers[i].Neurons[j].Synapses = append(nd.Layers[i].Neurons[j].Synapses, synapseData{})

				nd.Layers[i].Neurons[j].Synapses[k].Weight = n.layers[i].neurons[j].synapses[k].weight
				nd.Layers[i].Neurons[j].Synapses[k].In = n.layers[i].neurons[j].synapses[k].in
				nd.Layers[i].Neurons[j].Synapses[k].Out = n.layers[i].neurons[j].synapses[k].out
			}

			nd.Layers[i].Neurons[j].A = n.layers[i].neurons[j].a
			nd.Layers[i].Neurons[j].Z = n.layers[i].neurons[j].z
		}
	}

	return &nd
}

// NewNetworkFromData loads a network from a networkData struct
func NewNetworkFromData(nd *networkData) *network {
	n := network{
		activationFunction: nd.ActivationFunction,
		layers:             make([]*layer, len(nd.Layers)),
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

package main

import (
	log "github.com/sirupsen/logrus"
	"math"
	"math/rand"
	"time"
)

func RandomNeuronInit(neuron *NeuronUnit, p int) {
	rand.Seed(time.Now().UnixNano())
	neuron.Weights = make([]float64, p)
	for i := range neuron.Weights {
		neuron.Weights[i] = rand.Float64()
	}
	neuron.Bias = rand.Float64()
}

type Pattern struct {
	Features             []float64
	SingleRawExpectation string
	SingleExpectation    float64
	MultipleExpectation  []float64
}
type transferFunction func(float64) float64

func sigmoid(x float64) float64 {
	return 1.0 / (1.0 + math.Exp(-x))
}

type NeuronUnit struct {
	Weights []float64
	Bias    float64
	Lrate   float64
	Value   float64
	Delta   float64
}

type NeuralLayer struct {
	NeuronUnits []NeuronUnit
	Length      int
}

type MultiLayerNetwork struct {
	L_rate       float64
	NeuralLayers []NeuralLayer
	T_func       transferFunction
	T_func_d     transferFunction
}

func PrepareLayer(n int, p int) (l NeuralLayer) {

	l = NeuralLayer{NeuronUnits: make([]NeuronUnit, n), Length: n}

	for i := 0; i < n; i++ {
		RandomNeuronInit(&l.NeuronUnits[i], p)
	}

	log.WithFields(log.Fields{
		"level":               "info",
		"msg":                 "multilayer perceptron init completed",
		"neurons":             len(l.NeuronUnits),
		"lengthPreviousLayer": l.Length,
	}).Info("Complete NeuralLayer init.")

	return

}

func PrepareMLPNet(l []int, lr float64, tf transferFunction, trd transferFunction) (mlp MultiLayerNetwork) {

	// setup learning rate and transfer function
	mlp.L_rate = lr
	mlp.T_func = tf
	mlp.T_func_d = trd

	// setup layers
	mlp.NeuralLayers = make([]NeuralLayer, len(l))

	// for each layers specified
	for il, ql := range l {

		// if it is not the first
		if il != 0 {

			// prepare the GENERIC layer with specific dimension and correct number of links for each NeuronUnits
			mlp.NeuralLayers[il] = PrepareLayer(ql, l[il-1])

		} else {

			// prepare the INPUT layer with specific dimension and No links to previous.
			mlp.NeuralLayers[il] = PrepareLayer(ql, 0)

		}

	}

	log.WithFields(log.Fields{
		"level":          "info",
		"msg":            "multilayer perceptron init completed",
		"layers":         len(mlp.NeuralLayers),
		"learningRate: ": mlp.L_rate,
	}).Info("Complete Multilayer Perceptron init.")

	return

}

func Execute(mlp *MultiLayerNetwork, s *Pattern, options ...int) (r []float64) {

	// new value
	nv := 0.0

	// result of execution for each OUTPUT NeuronUnit in OUTPUT NeuralLayer
	r = make([]float64, mlp.NeuralLayers[len(mlp.NeuralLayers)-1].Length)

	// show pattern to network =>
	for i := 0; i < len(s.Features); i++ {

		// setup value of each neurons in first layers to respective features of pattern
		mlp.NeuralLayers[0].NeuronUnits[i].Value = s.Features[i]

	}

	// execute - hiddens + output
	// for each layers from first hidden to output
	for k := 1; k < len(mlp.NeuralLayers); k++ {

		// for each neurons in focused level
		for i := 0; i < mlp.NeuralLayers[k].Length; i++ {

			// init new value
			nv = 0.0

			// for each neurons in previous level (for k = 1, INPUT)
			for j := 0; j < mlp.NeuralLayers[k-1].Length; j++ {

				// sum output value of previous neurons multiplied by weight between previous and focused neuron
				nv += mlp.NeuralLayers[k].NeuronUnits[i].Weights[j] * mlp.NeuralLayers[k-1].NeuronUnits[j].Value

				log.WithFields(log.Fields{
					"level":                 "debug",
					"msg":                   "multilayer perceptron execution",
					"len(mlp.NeuralLayers)": len(mlp.NeuralLayers),
					"layer:  ":              k,
					"neuron: ":              i,
					"previous neuron: ":     j,
				}).Debug("Compute output propagation.")

			}

			// add neuron bias
			nv += mlp.NeuralLayers[k].NeuronUnits[i].Bias

			// compute activation function to new output value
			mlp.NeuralLayers[k].NeuronUnits[i].Value = mlp.T_func(nv)

			log.WithFields(log.Fields{
				"level":                 "debug",
				"msg":                   "setup new neuron output value after transfer function application",
				"len(mlp.NeuralLayers)": len(mlp.NeuralLayers),
				"layer:  ":              k,
				"neuron: ":              i,
				"outputvalue":           mlp.NeuralLayers[k].NeuronUnits[i].Value,
			}).Debug("Setup new neuron output value after transfer function application.")

		}

	}

	// get ouput values
	for i := 0; i < mlp.NeuralLayers[len(mlp.NeuralLayers)-1].Length; i++ {

		// simply accumulate values of all neurons in last level
		r[i] = mlp.NeuralLayers[len(mlp.NeuralLayers)-1].NeuronUnits[i].Value

	}

	return r

}
func BackPropagate(mlp *MultiLayerNetwork, s *Pattern, o []float64, options ...int) (r float64) {

	var no []float64
	// execute network with pattern passed over each level to output
	if len(options) == 1 {
		no = Execute(mlp, s, options[0])
	} else {
		no = Execute(mlp, s)
	}

	// init error
	e := 0.0

	// compute output error and delta in output layer
	for i := 0; i < mlp.NeuralLayers[len(mlp.NeuralLayers)-1].Length; i++ {

		// compute error in output: output for given pattern - output computed by network
		e = o[i] - no[i]

		// compute delta for each neuron in output layer as:
		// error in output * derivative of transfer function of network output
		mlp.NeuralLayers[len(mlp.NeuralLayers)-1].NeuronUnits[i].Delta = e * mlp.T_func_d(no[i])

	}

	// backpropagate error to previous layers
	// for each layers starting from the last hidden (len(mlp.NeuralLayers)-2)
	for k := len(mlp.NeuralLayers) - 2; k >= 0; k-- {

		// compute actual layer errors and re-compute delta
		for i := 0; i < mlp.NeuralLayers[k].Length; i++ {

			// reset error accumulator
			e = 0.0

			// for each link to next layer
			for j := 0; j < mlp.NeuralLayers[k+1].Length; j++ {

				// sum delta value of next neurons multiplied by weight between focused neuron and all neurons in next level
				e += mlp.NeuralLayers[k+1].NeuronUnits[j].Delta * mlp.NeuralLayers[k+1].NeuronUnits[j].Weights[i]

			}

			// compute delta for each neuron in focused layer as error * derivative of transfer function
			mlp.NeuralLayers[k].NeuronUnits[i].Delta = e * mlp.T_func_d(mlp.NeuralLayers[k].NeuronUnits[i].Value)

		}

		// compute weights in the next layer
		// for each link to next layer
		for i := 0; i < mlp.NeuralLayers[k+1].Length; i++ {

			// for each neurons in actual level (for k = 0, INPUT)
			for j := 0; j < mlp.NeuralLayers[k].Length; j++ {

				// sum learning rate * next level next neuron Delta * actual level actual neuron output value
				mlp.NeuralLayers[k+1].NeuronUnits[i].Weights[j] +=
					mlp.L_rate * mlp.NeuralLayers[k+1].NeuronUnits[i].Delta * mlp.NeuralLayers[k].NeuronUnits[j].Value

			}

			// learning rate * next level next neuron Delta * actual level actual neuron output value
			mlp.NeuralLayers[k+1].NeuronUnits[i].Bias += mlp.L_rate * mlp.NeuralLayers[k+1].NeuronUnits[i].Delta

		}

	}

	// compute global errors as sum of abs difference between output execution for each neuron in output layer
	// and desired value in each neuron in output layer
	for i := 0; i < len(o); i++ {

		r += math.Abs(no[i] - o[i])

	}

	// average error
	r = r / float64(len(o))

	return

}

func MLPTrain(mlp *MultiLayerNetwork, patterns []Pattern, mapped []string, epochs int) {

	epoch := 0
	output := make([]float64, len(mapped))

	// for fixed number of epochs
	for {

		// for each pattern in training set
		for _, pattern := range patterns {

			// setup desired output for each unit
			for io, _ := range output {
				output[io] = 0.0
			}
			// setup desired output for specific class of pattern focused
			output[int(pattern.SingleExpectation)] = 1.0
			// back propagation
			BackPropagate(mlp, &pattern, output)

		}

		log.WithFields(log.Fields{
			"level":  "info",
			"place":  "validation",
			"method": "MLPTrain",
			"epoch":  epoch,
		}).Debug("Training epoch completed.")

		// if max number of epochs is reached
		if epoch > epochs {
			// exit
			break
		}
		// increase number of epoch
		epoch++

	}

	log.WithFields(log.Fields{
		"level":  "info",
		"place":  "validation",
		"method": "MLPTrain",
		"epoch":  epoch,
	}).Info("Training completed.")
	// print weights in main function

}

func main() {
	//print infos
	log.WithFields(log.Fields{
		"level": "info",
		"place": "main",
		"msg":   "single layer perceptron train and test over sonar dataset",
	}).Info("Compute single layer perceptron on sonar data set (binary classification problem)")

}

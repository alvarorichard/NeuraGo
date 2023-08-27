package main

import (
	mn "github.com/made2591/go-perceptron-go/model/neural"
	mu "github.com/made2591/go-perceptron-go/util"
	v "github.com/made2591/go-perceptron-go/validation"
	log "github.com/sirupsen/logrus"
	"os"
	"sync"
)

func init() {
	// Output to stdout instead of the default stderr
	log.SetOutput(os.Stdout)
	// Only log the warning severity or above.
	log.SetLevel(log.InfoLevel)
}

type processFunc func()

func processSLP() {
	log.WithFields(log.Fields{
		"level": "info",
		"place": "main",
		"msg":   "single layer perceptron train and test over sonar dataset",
	}).Info("Compute single layer perceptron on sonar data set (binary classification problem)")

	// Code for Single Layer Perceptron
}

func processSection(sectionName string, process processFunc) {
	log.WithFields(log.Fields{
		"level": "info",
		"place": "main",
		"msg":   sectionName,
	}).Info("Computing...")

	process()

	log.WithFields(log.Fields{
		"level": "info",
		"place": "main",
		"msg":   sectionName,
	}).Info("Completed")
}

func processMLP() {
	log.WithFields(log.Fields{
		"level": "info",
		"place": "main",
		"msg":   "multi layer perceptron train and test over iris dataset",
	}).Info("Compute backpropagation multi layer perceptron on sonar data set (binary classification problem)")

	// ... Code for Multilayer Perceptron
}

func main() {
	var wg sync.WaitGroup
	wg.Add(3)

	go func() {
		defer wg.Done()

		log.WithFields(log.Fields{
			"level": "info",
			"place": "main",
			"msg":   "single layer perceptron train and test over sonar dataset",
		}).Info("Compute single layer perceptron on sonar data set (binary classification problem)")

		var filePath string = "./res/sonar.all_data.csv"
		var percentage float64 = 0.67
		var shuffle = 1
		var bias float64 = 0.0
		var learningRate float64 = 0.01
		var epochs int = 500
		var folds int = 5
		var patterns, _, _ = mn.LoadPatternsFromCSVFile(filePath)
		var neuron mn.NeuronUnit = mn.NeuronUnit{Weights: make([]float64, len(patterns[0].Features)), Bias: bias, Lrate: learningRate}
		var scores []float64 = v.KFoldValidation(&neuron, patterns, epochs, folds, shuffle)
		var neuron2 mn.NeuronUnit = mn.NeuronUnit{Weights: make([]float64, len(patterns[0].Features)), Bias: bias, Lrate: learningRate}
		var scores2 []float64 = v.RandomSubsamplingValidation(&neuron2, patterns, percentage, epochs, folds, shuffle)

		log.WithFields(log.Fields{
			"level":  "info",
			"place":  "main",
			"scores": scores,
		}).Info("Scores reached: ", scores)

		log.WithFields(log.Fields{
			"level":  "info",
			"place":  "main",
			"scores": scores2,
		}).Info("Scores reached: ", scores2)
	}()

	go func() {
		defer wg.Done()

		log.WithFields(log.Fields{
			"level": "info",
			"place": "main",
			"msg":   "multi layer perceptron train and test over iris dataset",
		}).Info("Compute backpropagation multi layer perceptron on sonar data set (binary classification problem)")

		var filePath = "./res/iris.all_data.csv"
		var learningRate = 0.01
		var percentage = 0.67
		var shuffle = 1
		var epochs = 500
		var folds = 3
		var patterns, _, mapped = mn.LoadPatternsFromCSVFile(filePath)
		var layers []int = []int{len(patterns[0].Features), 20, len(mapped)}
		var mlp mn.MultiLayerNetwork = mn.PrepareMLPNet(layers, learningRate, mn.SigmoidalTransfer, mn.SigmoidalTransferDerivate)
		var scores = v.MLPKFoldValidation(&mlp, patterns, epochs, folds, shuffle, mapped)
		var mlp2 mn.MultiLayerNetwork = mn.PrepareMLPNet(layers, learningRate, mn.SigmoidalTransfer, mn.SigmoidalTransferDerivate)
		var scores2 = v.MLPRandomSubsamplingValidation(&mlp2, patterns, percentage, epochs, folds, shuffle, mapped)

		log.WithFields(log.Fields{
			"level":  "info",
			"place":  "main",
			"scores": scores,
		}).Info("Scores reached: ", scores)

		log.WithFields(log.Fields{
			"level":  "info",
			"place":  "main",
			"scores": scores2,
		}).Info("Scores reached: ", scores2)
	}()

	go func() {
		defer wg.Done()

		log.WithFields(log.Fields{
			"level": "info",
			"place": "main",
			"msg":   "multi layer perceptron train and test over iris dataset",
		}).Info("Compute training algorithm on elman network using iris data set (binary classification problem)")

		var learningRate = 0.01
		var shuffle = 1
		var epochs = 500
		var patterns = mn.CreateRandomPatternArray(8, 30)
		var mlp mn.MultiLayerNetwork = mn.PrepareElmanNet(len(patterns[0].Features)+10,
			10, len(patterns[0].MultipleExpectation), learningRate,
			mn.SigmoidalTransfer, mn.SigmoidalTransferDerivate)
		var mean, _ = v.RNNValidation(&mlp, patterns, epochs, shuffle)

		log.WithFields(log.Fields{
			"level":     "info",
			"place":     "main",
			"precision": mu.Round(mean, .5, 2),
		}).Info("Scores reached: ", mu.Round(mean, .5, 2))
	}()

	wg.Wait()
}

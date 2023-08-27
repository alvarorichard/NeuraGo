# NeuraGo

## Neural Network Training and Validation Example

This project contains a Go program that demonstrates the implementation and evaluation of different neural network models for classification tasks. The code focuses on training and evaluating neural network models using various architectures and validation techniques.

## Features
The code includes the following main features:

## Single Layer Perceptron (SLP):

Training and testing of a Single Layer Perceptron on the "sonar" dataset.
Demonstration of k-fold cross-validation and random subsampling validation.

## Multi-Layer Perceptron (MLP):

Training and testing of a Multi-Layer Perceptron on the "iris" dataset.
Presentation of k-fold cross-validation and random subsampling validation.


## Elman Network:

Training and testing of an Elman Network on randomly generated data.
Usage of a specific network architecture and logging of validation results.

## How to Run the Code

Follow the steps below to run the program:

## Clone the Repository:

Clone this repository to your system.

```
git clone https://github.com/alvarorichard/NeuraGo.git
cd NeuraGo
```
## Install Dependencies:

The code uses external packages. You need to download the dependencies using the following command:

```bash
go get github.com/made2591/go-perceptron-go/model/neural
go get github.com/sirupsen/logrus
```

## Running the Program:

Run the program using the go run command followed by the main file's name:
```bash
go run main.go
```
The program will start executing the various training and validation tasks for the different neural network models and datasets.

## Notes

Ensure that all dependencies are installed correctly.
You can adjust the training parameters such as learning rates, epochs, and validation folds as needed for your experiment.
This project is based on the tutorial "Build a Multilayer Perceptron with Golang"

## Contributions Welcome

Contributions to this project are welcome! If you have any ideas, bug fixes, or enhancements, feel free to submit a pull request. Please make sure to follow the contribution guidelines and adhere to the code of conduct.










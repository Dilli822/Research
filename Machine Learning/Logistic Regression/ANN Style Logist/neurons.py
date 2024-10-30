import numpy as np
import pandas as pd

class Perceptron:
    def __init__(self, num_inputs=2, learning_rate=0.1, initial_weights=None, initial_bias=0.0):
        if initial_weights is None:
            self.weights = np.zeros(num_inputs)
        else:
            self.weights = np.array(initial_weights)
        self.bias = initial_bias
        self.learning_rate = learning_rate
        
    def activation_function(self, z):
        return 1 if z >= 0 else 0
    
    def predict(self, inputs):
        inputs = np.array(inputs)
        z = np.dot(self.weights, inputs) + self.bias
        return self.activation_function(z)
    
    def train(self, training_data, epochs):
        history = {
            'weights': [],
            'bias': [],
            'errors': []
        }
        
        for epoch in range(epochs):
            total_error = 0
            print(f"\nEpoch {epoch + 1}/{epochs}")
            
            for inputs, target in training_data:
                inputs = np.array(inputs)
                output = self.predict(inputs)
                error = target - output
                total_error += abs(error)
                
                # Update weights and bias only if there's an error
                if error != 0:
                    weight_updates = self.learning_rate * error * inputs
                    bias_update = self.learning_rate * error
                    
                    self.weights += weight_updates
                    self.bias += bias_update
                    
                    print(f"Inputs: {inputs}, Target: {target}, Output: {output}")
                    print(f"Error: {error}")
                    print(f"Weight Updates: {weight_updates}, New Weights: {self.weights}")
                    print(f"Bias Update: {bias_update}, New Bias: {self.bias}")
                
                history['weights'].append(self.weights.copy())
                history['bias'].append(self.bias)
                history['errors'].append(error)
            
            if total_error == 0:
                print(f"\nConverged at epoch {epoch + 1}")
                break
                
        return history

def train_and_test_gate(gate_type):
    print(f"\nTraining for {gate_type} Operation:")
    
    if gate_type == "NOT":
        perceptron = Perceptron(num_inputs=1, learning_rate=0.1, initial_weights=[-1.0], initial_bias=0.5)
        training_data = [
            (np.array([0]), 1),
            (np.array([1]), 0)
        ]
        test_cases = [(0,), (1,)]
    else:
        perceptron = Perceptron(num_inputs=2, learning_rate=0.1)
        if gate_type == "AND":
            training_data = [
                (np.array([0, 0]), 0),
                (np.array([0, 1]), 0),
                (np.array([1, 0]), 0),
                (np.array([1, 1]), 1)
            ]
        else:  # OR
            training_data = [
                (np.array([0, 0]), 0),
                (np.array([0, 1]), 1),
                (np.array([1, 0]), 1),
                (np.array([1, 1]), 1)
            ]
        test_cases = [(0, 0), (0, 1), (1, 0), (1, 1)]
    
    history = perceptron.train(training_data, epochs=15)
    
    print(f"\nTesting {gate_type} gate:")
    for test_input in test_cases:
        output = perceptron.predict(test_input)
        if gate_type == "NOT":
            print(f"{gate_type}({test_input[0]}) = {output}")
        else:
            print(f"{gate_type}({test_input[0]}, {test_input[1]}) = {output}")
    
    return perceptron, history

if __name__ == "__main__":
    # Train and test all gates
    gates = ["AND", "OR", "NOT"]
    results = {}
    
    for gate in gates:
        perceptron, history = train_and_test_gate(gate)
        results[gate] = {
            'perceptron': perceptron,
            'history': history
        }
        
        # Display final weights and bias
        print(f"\nFinal {gate} gate parameters:")
        print(f"Weights: {perceptron.weights}")
        print(f"Bias: {perceptron.bias}")
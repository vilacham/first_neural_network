import numpy as np


class NeuralNetwork(object):
    def __init__(self, N_input, N_hidden, N_output, learning_rate):
        # Set number of nodes in input, hidden and output layers.
        self.N_input = N_input
        self.N_hidden = N_hidden
        self.N_output = N_output

        # Initialize weights
        self.w_input_hidden = np.random.normal(0.0, self.N_input ** -0.5,
                                               (self.N_input, self.N_hidden))

        self.w_hidden_output = np.random.normal(0.0, self.N_hidden ** -0.5,
                                                (self.N_hidden, self.N_output))
        self.lr = learning_rate
        
        # Set self.activation_function to your implemented sigmoid function
        self.activation_function = lambda x: 1 / (1 + np.exp(-x))

    def train(self, features, targets):
        """ Train the network on batch of features and targets. 
        
            Arguments
            ---------
            features: 2D array (rows are data records, columns are features)
            targets: 1D array (target values)
        """
        n_records = features.shape[0]
        delta_w_i_h = np.zeros(self.w_input_hidden.shape)
        delta_w_h_o = np.zeros(self.w_hidden_output.shape)
        for X, y in zip(features, targets):
            
            final_outputs, hidden_outputs = self.forward_pass_train(X)
            delta_w_i_h, delta_w_h_o = self.backpropagation(final_outputs,
                                                            hidden_outputs, X,
                                                            y, delta_w_i_h,
                                                            delta_w_h_o)
        self.update_weights(delta_w_i_h, delta_w_h_o, n_records)

    def forward_pass_train(self, X):
        """ Implement forward pass here 
         
            Arguments
            ---------
            X: features batch
        """
        # Hidden layer
        hidden_inputs = np.dot(X, self.w_input_hidden)
        hidden_outputs = self.activation_function(hidden_inputs)

        # Output layer
        final_inputs = np.dot(hidden_outputs, self.w_hidden_output)
        final_outputs = final_inputs
        
        return final_outputs, hidden_outputs

    def backpropagation(self, final_output, hidden_output, X, y, delta_w_i_h,
                        delta_w_h_o):
        """ Implement backpropagation

            Arguments
            ---------
            final_output: output from forward pass
            hidden_output: output from hidden layer
            y: target (i.e. label) batch
            delta_w_i_h: change in weights from input to hidden layers
            delta_w_h_o: change in weights from hidden to output layers
        """
        # Output error (difference between desired target and actual output)
        error = y - final_output

        # Hidden layer's contribution to the error (product between weights of
        # the hidden layer and output error)
        hidden_error = np.dot(self.w_hidden_output, error)

        # Backpropagated error terms (product between the respective error and
        # the derivative of the respective activation function). For the output
        # layer, f'(x) = 1. For the hidden layer, f'(x) = f(x) * (1 - f(x)).
        output_error_term = error
        hidden_error_term = hidden_error * hidden_output * (1 - hidden_output)

        # Weight step (input to hidden: product between hidden error term and
        # input values)
        delta_w_i_h += hidden_error_term * X[:, None]
        # Weight step (hidden to output: product between output error term and
        # hidden layer activation values)
        delta_w_h_o += output_error_term * hidden_output[:,  None]
        return delta_w_i_h, delta_w_h_o

    def update_weights(self, delta_w_i_h, delta_w_h_o, n_records):
        """ Update weights on gradient descent step
         
            Arguments
            ---------
            delta_w_i_h: change in weights from input to hidden layers
            delta_w_h_o: change in weights from hidden to output layers
            n_records: number of records
        """
        # Update hidden-to-output weights with gradient descent step (by
        # multiplying the learning rate by the respective weight step and
        # dividing this product by the number of records, and finally adding
        # the result to the respective weights)
        self.w_hidden_output += self.lr * delta_w_h_o / n_records
        # Update input-to-hidden weights with gradient descent step (same
        # operations as above)
        self.w_input_hidden += self.lr * delta_w_i_h / n_records

    def run(self, features):
        """ Run a forward pass through the network with input features 
        
            Arguments
            ---------
            features: 1D array (feature values)
        """
        # Hidden layer
        hidden_inputs = np.dot(features, self.w_input_hidden)
        hidden_outputs = self.activation_function(hidden_inputs)
        
        # Output layer
        final_inputs = np.dot(hidden_outputs, self.w_hidden_output)
        final_outputs = final_inputs
        
        return final_outputs


# Set hyperparameters below
iterations = 10000
learning_rate = 0.3
hidden_nodes = 13
output_nodes = 1

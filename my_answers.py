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
        final_outputs = self.activation_function(final_inputs)
        
        return final_outputs, hidden_outputs

    def backpropagation(self, final_outputs, hidden_outputs, X, y, delta_weights_i_h, delta_weights_h_o):
        ''' Implement backpropagation
         
            Arguments
            ---------
            final_outputs: output from forward pass
            y: target (i.e. label) batch
            delta_weights_i_h: change in weights from input to hidden layers
            delta_weights_h_o: change in weights from hidden to output layers

        '''
        #### Implement the backward pass here ####
        ### Backward pass ###

        # TODO: Output error - Replace this value with your calculations.
        error = None # Output layer error is the difference between desired target and actual output.
        
        # TODO: Calculate the hidden layer's contribution to the error
        hidden_error = None
        
        # TODO: Backpropagated error terms - Replace these values with your calculations.
        output_error_term = None
        
        hidden_error_term = None
        
        # Weight step (input to hidden)
        delta_weights_i_h += None
        # Weight step (hidden to output)
        delta_weights_h_o += None
        return delta_weights_i_h, delta_weights_h_o

    def update_weights(self, delta_weights_i_h, delta_weights_h_o, n_records):
        ''' Update weights on gradient descent step
         
            Arguments
            ---------
            delta_weights_i_h: change in weights from input to hidden layers
            delta_weights_h_o: change in weights from hidden to output layers
            n_records: number of records

        '''
        self.w_hidden_output += None # update hidden-to-output weights with gradient descent step
        self.w_input_hidden += None # update input-to-hidden weights with gradient descent step

    def run(self, features):
        ''' Run a forward pass through the network with input features 
        
            Arguments
            ---------
            features: 1D array of feature values
        '''
        
        #### Implement the forward pass here ####
        # TODO: Hidden layer - replace these values with the appropriate calculations.
        hidden_inputs = None # signals into hidden layer
        hidden_outputs = None # signals from hidden layer
        
        # TODO: Output layer - Replace these values with the appropriate calculations.
        final_inputs = None # signals into final output layer
        final_outputs = None # signals from final output layer 
        
        return final_outputs


#########################################################
# Set your hyperparameters here
##########################################################
iterations = 100
learning_rate = 0.1
hidden_nodes = 2
output_nodes = 1

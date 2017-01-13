import math

class PyNNet(object):

    def __init__(self, neuronCounts, connections, tanh = True):
        self.neuronCounts = neuronCounts
        self.connections = connections
        self.tanh = tanh
        self.values = None

    def calc(self, inputs):
        if len(inputs) != self.neuronCounts[0]:
            raise ValueError("Wrong amount of inputs")

        # calc each layer
        inputs.append(1) # append bias
        self.values = (inputs, {})
        for layerIndex in range(1, len(self.neuronCounts)):
            for neuron in range(self.neuronCounts[layerIndex]):
                data = (neuron, layerIndex)
                self.__h_calc_neuron(data)
            self.values[1][len(self.values[1])] = 1 # add bias
            self.values = (self.values[1], {})

        # ignores auto-added bias by choosing range with neuronCounts
        return [self.values[0][i] for i in range(self.neuronCounts[-1])]


    def __h_calc_neuron(self, data): # (neuron, layerIndex):
        neuron      = data[0]
        connections = self.__h_getConnectionsTo(data[1], neuron)

        sumOfAll = 0
        sumOfConnections = 0
        for prevN in range(len(connections)):
            sumOfAll += self.values[0][prevN] * connections[prevN]
            sumOfConnections += connections[prevN]

        if self.tanh:
            self.values[1][neuron] = math.tanh(sumOfAll / sumOfConnections)
        else:
            self.values[1][neuron] = sumOfAll / sumOfConnections


    def __h_getConnectionsTo(self, layerIndex, neuron):
        connectionsPrev = 0
        for i in range(1, layerIndex):
            connectionsPrev += (self.neuronCounts[i - 1] + 1) * self.neuronCounts[i]

        return [ self.connections[connectionsPrev:][i] for i in range(self.neuronCounts[layerIndex - 1] + 1) ]

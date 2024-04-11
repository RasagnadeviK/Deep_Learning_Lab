import numpy as np
class McCulloch:
    def __init__(self,w,t):
        self.w=w
        self.t=t
    def activate(self,input):
        s=0
        r=len(input)
        for i in range(r):
            s+=input[i]*self.w[i]
        outputs = np.where(s >= self.t, 1, 0)
        return outputs
def single_layer_nn(inputs,r):
    n1 = r // 2 if r >= 2 else 1
    w1=np.random.uniform(-1, 1, size=(n1,r))
    t1=np.random.uniform(-1, 1, size=n1)
    layer_outputs = []
    for i,j in zip(w1,t1):
        neuron = McCulloch(i,j)
        neuron_output = neuron.activate(inputs)
        layer_outputs.append(neuron_output)
    return layer_outputs
def multiple_layer_nn(inputs,r):
    n1 = r // 2 if r >= 2 else 1
    w1=np.random.uniform(-1, 1, size=(n1,r))
    t1=np.random.uniform(-1, 1, size=n1)
    w2=np.random.uniform(-1, 1, size=(n1,r))
    t2=np.random.uniform(-1, 1, size=(n1,r))
    layer1_outputs = []
    for i,j in zip(w1,t1):
        neuron=McCulloch(i,j)
        neuron_output=neuron.activate(inputs)
        layer1_outputs.append(neuron_output)
    layer2_neuron=McCulloch(w2,t2)
    layer2_output=layer2_neuron.activate(layer1_outputs)
    return layer2_output
input_size = np.random.randint(1, 10)
input_data = np.random.uniform(0, 1, size=input_size)
single_layer_result = single_layer_nn(input_data,input_size)
print("Single-layer NN Output:", single_layer_result)
multiple_layer_result = multiple_layer_nn(input_data,input_size)
print("Multiple-layer NN Output:", multiple_layer_result)
import mnist_loader
import os
import pickle

training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
import network
net = network.Network([784, 10, 10])
net.SGD(training_data, 30, 10, 3.0, test_data=test_data)

# Guardar los pesos y sesgos finales
here = os.path.dirname(__file__)
trained_net_path = os.path.normpath(os.path.join(here, '..', 'data', 'trained_net.pkl'))
with open(trained_net_path, 'wb') as f:
    pickle.dump((net.weights, net.biases), f)
print(f"\nModelo guardado en {trained_net_path}")
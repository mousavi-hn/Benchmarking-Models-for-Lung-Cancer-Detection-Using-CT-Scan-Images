from keras import layers
import pennylane as qml
import jax
import jax.numpy as jnp

class QuantumLayer(layers.Layer):
    def __init__(self, n_qubits, q_depth=2, **kwargs):
        super().__init__(**kwargs)
        self.n_qubits = n_qubits
        self.q_depth = q_depth
        self.dev = qml.device("default.qubit", wires=n_qubits)

        @qml.qnode(self.dev, interface="jax")
        def circuit(inputs, weights):
            qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation="Y")
            qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

        self.circuit = circuit

    def build(self, input_shape):
        self.q_weights = self.add_weight(
            name="q_weights",
            shape=(self.q_depth, self.n_qubits, 3),
            initializer="glorot_uniform",
            trainable=True,
        )
        super().build(input_shape)

    def call(self, inputs):
        inputs = jnp.asarray(inputs)
        weights = jnp.asarray(self.q_weights)

        def single_forward(x):
            return self.circuit(x, weights)

        outputs = jax.vmap(single_forward)(inputs)
        return outputs

    def get_config(self):
        config = super().get_config()
        config.update({
            "n_qubits": self.n_qubits,
            "q_depth": self.q_depth,
        })
        return config
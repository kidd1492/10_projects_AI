import pickle


# Save Model Parameters


def save_model(model, filepath='model.pkl'):
    """
    Saves only the parameters (weights and biases) of each layer.
    """
    state = []

    for layer in model.layers:
        layer_params = {}

        # Save weights if present
        if hasattr(layer, "w"):
            layer_params["w"] = layer.w

        # Save biases if present
        if hasattr(layer, "b"):
            layer_params["b"] = layer.b

        state.append(layer_params)

    with open(filepath, 'wb') as f:
        pickle.dump(state, f)

    print(f"Model saved to {filepath}")



# Load Model Parameters

def load_model(model, filepath='model.pkl'):
    """
    Loads parameters into an existing model instance.
    The architecture must match the saved model.
    """
    with open(filepath, 'rb') as f:
        state = pickle.load(f)

    for layer, layer_state in zip(model.layers, state):

        if hasattr(layer, "w") and "w" in layer_state:
            layer.w = layer_state["w"]

        if hasattr(layer, "b") and "b" in layer_state:
            layer.b = layer_state["b"]

    print(f"Model loaded from {filepath}")


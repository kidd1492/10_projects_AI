# **AI Learning Roadmap — 10 Project Curriculum**

This repository contains a complete, end‑to‑end learning path for understanding and building modern AI systems.  
The curriculum is designed to take you from **first principles** (math, geometry, regression, classification) all the way to **building real neural networks**, both from scratch and with PyTorch.

The philosophy behind these projects is simple:

- **Start with intuition, not abstraction.**  
- **Build everything yourself before using a framework.**  
- **See the geometry behind the math.**  
- **Understand how shapes flow through a neural network.**  
- **Learn by coding, visualizing, and experimenting.**

Each project builds directly on the previous one, forming a coherent mental model of how machine learning works under the hood.

[Neural Network Lerning Hub](https://kidd1492.github.io/neural_nets_home.html)

## **Section 1: Foundations (Math, Regression, Classification, XOR)**  
These five projects establish the core ideas of **linear algebra → optimization → nonlinearity**.

### **1. Single‑Feature Linear Regression**
- MSE, gradients, update  
- r, R², OLS  
- Geometry of projection  

### **2. Multi‑Feature Regression**
- Vectorized dot products  
- Weight vector as a direction in feature space  
- Plane instead of line  
- Residual vectors  

### **3. Binary Classification**
- Logistic regression  
- Sigmoid  
- Cross‑entropy  
- Decision boundary geometry  

### **4. Neural Network for XOR**
- Hidden layer  
- Activation functions  
- Why linear models fail  
- Geometry of separating non‑linearly separable data  

### **5. The Design Matrix**
- Why ML always uses a matrix  
- How neural networks generalize it  
- How shapes flow through the system  
- How this is a neural network layer  

[Neural Network Math](https://kidd1492.github.io/intro_nn.html)
A simple “Neural Networks Math” explanation with three images summarizing what these five projects teach.

---

## **Section 2: Building Real Neural Networks (Projects 6–10)**  
This section moves from single neurons to full neural network architectures — first from scratch, then with PyTorch.

### **6. Build a Neural Network Class From Scratch**
- A `Layer` class  
- A `NeuralNetwork` class  
- Forward pass  
- Backprop  
- Training loop  
- Loss functions  
- Activation functions  
- Modular architecture  

### **7. Introduction to PyTorch (Tensors, Autograd, Modules)**
- How PyTorch replaces your manual gradients  
- How `nn.Module` mirrors your custom class  
- How autograd works  
- How optimizers work  
- How to rewrite your Project 5 network in PyTorch  

### **8. Recurrent Neural Networks (RNNs)**
- Sequence modeling  
- Hidden state  
- Unrolling  
- Vanishing gradients  
- PyTorch’s `nn.RNN`  

### **9. LSTM Networks**
- The natural evolution of Project 7  
- Gates  
- Cell state  
- Long‑term memory  
- Why LSTMs solve vanishing gradients  
- PyTorch’s `nn.LSTM`  

### **10. GRU or CNN (Your Choice)**
- Explore GRU as a simplified gated RNN, **or**  
- Explore CNNs for spatial feature extraction  
- Compare trade‑offs and ideal use cases  

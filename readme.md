# SKProof
## About
### Supported models
- MLPClassifier with ReLU activation function

## How it works

## Planned improvements
- Code optimization
- Support for more models

## Prerequisites

## Installation
Installing skproof package is done using pip with command `pip install starknet_skproof`

## Example
Example using Iris dataset and MLPClassifier
```python
from starknet_skproof.mlp.MLPClassifierProver import MLPClassifierProver
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import load_iris

print('Loading dataset...')

# Load test data
iris = load_iris()
X = iris.data
y = iris.target

print('Training MLPClassifier...')

# Train classifier
mlp = MLPClassifier((2,3), activation='relu', max_iter=2000)
mlp.fit(X, y)

# Generate proof for the first row
mlpcp = MLPClassifierProver(
    mlp,
    'src/main.cairo',
    './zkfloat/zkfloat.cairo',
    7
)

mlpcp.prove(X[:1,:])
```
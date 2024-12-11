import numpy as np

# 1. Define the dataset
# Features: [minutes_played, shots_on_target, total_shots, key_passes, team_possession, opponent_strength]
X = np.array([
    [90, 4, 6, 3, 65, 1],
    [75, 3, 5, 2, 60, 2],
    [85, 5, 7, 4, 70, 3],
    [90, 6, 8, 5, 72, 4],
    [70, 2, 3, 1, 55, 1]
])

# Target (goals scored by Messi)
y = np.array([2, 1, 3, 2, 0])

# Add a column of ones to X for the bias term (intercept)
X = np.c_[np.ones(X.shape[0]), X]  # Shape: (n_samples, n_features + 1)

# 2. Implement Linear Regression
def train_linear_regression(X, y, learning_rate=0.0001, epochs=10000):
    # Initialize weights (w) randomly
    w = np.random.randn(X.shape[1])
    
    for epoch in range(epochs):
        # Predictions
        y_pred = X @ w
        
        # Compute the error
        error = y_pred - y
        
        # Compute gradients
        gradients = (1 / len(y)) * (X.T @ error)
        
        # Gradient clipping (optional)
        max_gradient = 1.0
        gradients = np.clip(gradients, -max_gradient, max_gradient)
        
        # Update weights
        w -= learning_rate * gradients
        
    return w

# Train the model
weights = train_linear_regression(X, y)
print(f"Trained Weights: {weights}")

# 3. Predict Messi's goals for a new match
def predict(X, weights):
    X = np.c_[np.ones(X.shape[0]), X]  # Add the bias term
    return X @ weights

# Example match: [minutes_played, shots_on_target, total_shots, key_passes, team_possession, opponent_strength]
new_match = np.array([[90, 6, 9, 9, 75, 2]])
predicted_goals = predict(new_match, weights)
print(f"Predicted Goals: {predicted_goals[0]:.2f}")


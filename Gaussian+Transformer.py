import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from datetime import datetime
import uuid

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Gaussian Process Implementation
class GaussianProcess:
    def __init__(self, length_scale=1.0, sigma_f=1.0, sigma_n=1e-2):
        self.length_scale = length_scale
        self.sigma_f = sigma_f
        self.sigma_n = sigma_n

    def rbf_kernel(self, X1, X2):
        # Compute pairwise squared distances
        sqdist = np.sum(X1**2, 1).reshape(-1, 1) + np.sum(X2**2, 1) - 2 * np.dot(X1, X2.T)
        return self.sigma_f**2 * np.exp(-0.5 / self.length_scale**2 * sqdist)

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
        self.K = self.rbf_kernel(X, X) + self.sigma_n**2 * np.eye(len(X))
        self.L = np.linalg.cholesky(self.K)  # Cholesky decomposition for stability

    def predict(self, X_test):
        K_s = self.rbf_kernel(X_test, self.X_train)
        K_ss = self.rbf_kernel(X_test, X_test) + self.sigma_n**2 * np.eye(len(X_test))
        
        # Solve for alpha using Cholesky decomposition
        alpha = np.linalg.solve(self.L.T, np.linalg.solve(self.L, self.y_train))
        
        # Mean prediction
        mu = K_s.dot(alpha)
        
        # Variance (uncertainty)
        v = np.linalg.solve(self.L, K_s.T)
        var = K_ss - v.T.dot(v)
        std = np.sqrt(np.maximum(np.diag(var), 0))  # Ensure non-negative variance
        
        return mu, std

# Transformer Model
class TransformerModel(nn.Module):
    def __init__(self, input_dim=1, d_model=64, nhead=4, num_layers=2, dropout=0.1):
        super(TransformerModel, self).__init__()
        self.input_linear = nn.Linear(input_dim, d_model)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True),
            num_layers=num_layers
        )
        self.output_linear = nn.Linear(d_model, 1)
        self.pos_encoder = nn.Parameter(torch.zeros(1, 60, d_model))  # sequence_length = 60

    def forward(self, x):
        # x shape: (batch, seq_len, input_dim)
        x = self.input_linear(x)  # Project to d_model
        x = x + self.pos_encoder[:, :x.size(1), :]  # Add positional encoding
        x = self.transformer(x)  # Transformer encoder
        x = self.output_linear(x[:, -1, :])  # Take the last time step
        return x

# Load and preprocess the dataset
data = pd.read_csv('yahoo_stock.csv')

# Remove duplicate dates (weekends/holidays) by keeping the first occurrence
data['Date'] = pd.to_datetime(data['Date'])
data = data.drop_duplicates(subset='Date', keep='first')

# Extract the 'Close' prices and dates
dates = data['Date'].values
prices = data['Close'].values.reshape(-1, 1)

# Normalize the prices
scaler = MinMaxScaler()
prices_normalized = scaler.fit_transform(prices)

# Create sequences for time series (use past 60 days to predict the next day)
sequence_length = 60
X, y = [], []
for i in range(len(prices_normalized) - sequence_length):
    X.append(prices_normalized[i:i + sequence_length])
    y.append(prices_normalized[i + sequence_length])
X = np.array(X)
y = np.array(y)

# Split into training and testing sets (80% train, 20% test)
train_size = int(0.8 * len(X))
X_train, y_train = X[:train_size], y[:train_size]
X_test, y_test = X[train_size:], y[train_size:]
dates_test = dates[train_size + sequence_length:]

# Convert to PyTorch tensors
X_train = torch.FloatTensor(X_train)
y_train = torch.FloatTensor(y_train)
X_test = torch.FloatTensor(X_test)
y_test = torch.FloatTensor(y_test)

# Initialize and train the Transformer model
transformer_model = TransformerModel()
optimizer = torch.optim.Adam(transformer_model.parameters(), lr=0.001)
criterion = nn.MSELoss()

# Training the Transformer
num_epochs = 50
for epoch in range(num_epochs):
    transformer_model.train()
    optimizer.zero_grad()
    output = transformer_model(X_train)
    loss = criterion(output, y_train)
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Get Transformer predictions
transformer_model.eval()
with torch.no_grad():
    train_preds = transformer_model(X_train).numpy()
    test_preds = transformer_model(X_test).numpy()

# Compute residuals for GP
residuals_train = y_train.numpy() - train_preds

# Train GP on residuals
gp = GaussianProcess(length_scale=1.0, sigma_f=1.0, sigma_n=1e-2)
X_train_gp = np.arange(len(X_train)).reshape(-1, 1)
gp.fit(X_train_gp, residuals_train)

# Predict residuals for test set
X_test_gp = np.arange(len(X_train), len(X_train) + len(X_test)).reshape(-1, 1)
residuals_pred, residuals_std = gp.predict(X_test_gp)

# Combine Transformer predictions with GP residual predictions
final_preds = test_preds + residuals_pred

# Inverse transform predictions and actual values
final_preds = scaler.inverse_transform(final_preds.reshape(-1, 1))
y_test_actual = scaler.inverse_transform(y_test.numpy())
test_preds = scaler.inverse_transform(test_preds.reshape(-1, 1))
residuals_std_scaled = scaler.inverse_transform(residuals_std.reshape(-1, 1))

# Plot results
plt.figure(figsize=(12, 6))
plt.plot(dates_test, y_test_actual, label='Actual Prices', color='blue')
plt.plot(dates_test, test_preds, label='Transformer Predictions', color='orange')
plt.plot(dates_test, final_preds, label='Hybrid (Transformer + GP) Predictions', color='green')
plt.fill_between(dates_test, 
                 final_preds.flatten() - 1.96 * residuals_std_scaled.flatten(),
                 final_preds.flatten() + 1.96 * residuals_std_scaled.flatten(),
                 color='green', alpha=0.2, label='GP Uncertainty (95% CI)')
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.title('Stock Price Prediction with Transformer + Gaussian Process')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('stock_price_prediction.png')
plt.close()

# Save predictions to a CSV file
results = pd.DataFrame({
    'Date': dates_test,
    'Actual': y_test_actual.flatten(),
    'Transformer': test_preds.flatten(),
    'Hybrid': final_preds.flatten(),
    'Lower_CI': (final_preds.flatten() - 1.96 * residuals_std_scaled.flatten()),
    'Upper_CI': (final_preds.flatten() + 1.96 * residuals_std_scaled.flatten())
})
results.to_csv('stock_price_predictions.csv', index=False)
print("Predictions saved to 'stock_price_predictions.csv'")
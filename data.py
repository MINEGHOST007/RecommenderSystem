import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import streamlit as st
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

# Define the BMI calculation function
def calculate_bmi(height, weight):
    """Calculate BMI based on height (in cm) and weight (in kg)."""
    height_m = height / 100  # Convert height to meters
    return round(weight / (height_m ** 2), 2)

# Define the Autoencoder and ClusteringLayer with matched dimensions
class Autoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dims):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[1], hidden_dims[2]),
            nn.ReLU(),
            nn.Linear(hidden_dims[2], hidden_dims[3]),
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dims[3], hidden_dims[2]),
            nn.ReLU(),
            nn.Linear(hidden_dims[2], hidden_dims[1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[1], hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(hidden_dims[0], input_dim),
            nn.Sigmoid(),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded

class ClusteringLayer(nn.Module):
    def __init__(self, n_clusters, embedding_dim):
        super(ClusteringLayer, self).__init__()
        self.clusters = nn.Parameter(torch.Tensor(n_clusters, embedding_dim))
        torch.nn.init.xavier_normal_(self.clusters.data)

    def forward(self, x):
        q = 1.0 / (1.0 + torch.sum((x.unsqueeze(1) - self.clusters) ** 2, dim=2))
        q = q ** (2.0 / (2.0 - 1))
        q = q / torch.sum(q, dim=1, keepdim=True)
        return q

class DEC(nn.Module):
    def __init__(self, input_dim, hidden_dims, n_clusters):
        super(DEC, self).__init__()
        self.autoencoder = Autoencoder(input_dim, hidden_dims)
        self.clustering = ClusteringLayer(n_clusters, hidden_dims[-1])

    def forward(self, x):
        encoded, decoded = self.autoencoder(x)
        cluster_assignment = self.clustering(encoded)
        return cluster_assignment, decoded

def target_distribution(q):
    weight = q ** 2 / q.sum(0)
    return (weight.T / weight.sum(1)).T

# Load the dataset and preprocess
df = pd.read_csv('enhanced_ml_project_dataset.csv')

# Apply weight adjustments to the dataset
def apply_weights(df, height, weight, chest, waist, hip, gender, age, brand, fit_feedback, return_exchange_status, loyalty_status):
    weight = 1.0  # Start with a neutral weight

    # Gender-based adjustments
    if gender == 'Female':
        weight *= 1.1  # Slightly increase weight for female recommendations

    # Age-based adjustments
    if age > 55:
        weight *= 1.2
    elif 40 < age <= 55:
        weight *= 1.1

    # Brand-based adjustments
    if brand == 'Zara':
        weight *= 0.9
    elif brand == 'H&M':
        weight *= 1.1
    elif brand == 'Nike':
        weight *= 0.95
    elif brand == 'Uniqlo':
        weight *= 1.05

    # Fit feedback-based adjustments
    if fit_feedback == 'Too Small':
        weight *= 0.8
    elif fit_feedback == 'Too Large':
        weight *= 1.2

    # Return/Exchange status adjustments
    if return_exchange_status == 'Returned':
        weight *= 0.8
    elif return_exchange_status == 'Exchanged':
        weight *= 0.9

    # Loyalty status adjustments
    if loyalty_status == 'Frequent Buyer':
        weight *= 1.3
    elif loyalty_status == 'Occasional Buyer':
        weight *= 1.1

    return weight

# Main page user inputs
st.title("AI-Powered Apparel Size Chart Generator")

height = st.slider("Height (cm)", 150, 200, 175)
weight = st.slider("Weight (kg)", 45, 120, 70)
chest = st.slider("Chest (cm)", 75, 135, 100)
waist = st.slider("Waist (cm)", 60, 110, 80)
hip = st.slider("Hip (cm)", 80, 130, 95)
gender = st.selectbox("Gender", ['Male', 'Female', 'Non-Binary'])
age = st.slider("Age", 18, 70, 35)
brand = st.selectbox("Preferred Brand", ['Zara', 'H&M', 'Nike', 'Levi\'s', 'Uniqlo'])
fit_feedback = st.selectbox("Fit Feedback", ['Too Small', 'Fit Well', 'Too Large'])
return_exchange_status = st.selectbox("Return/Exchange Status", ['Kept', 'Returned', 'Exchanged'])
loyalty_status = st.selectbox("Loyalty Status", ['Frequent Buyer', 'Occasional Buyer', 'New Buyer'])

# Collect base_size from the user
base_size = st.selectbox("Base Size", ['XS', 'S', 'M', 'L', 'XL', 'XXL'])
item_type = st.selectbox("Type of Clothing", ['T-shirt', 'Jeans', 'Dress', 'Jacket', 'Skirt', 'Shorts', 'Sweater'])

# Apply the weight adjustments to the dataset
df['Weights'] = df.apply(lambda row: apply_weights(
    df, height, weight, chest, waist, hip, gender, age, brand, fit_feedback, return_exchange_status, loyalty_status), axis=1)

# Select relevant features and scale them
features = df[['Height (cm)', 'Weight (kg)', 'Chest (cm)', 'Waist (cm)', 'Hip (cm)', 'BMI', 'Base Size', 'Weights']]
features = pd.get_dummies(features, columns=['Base Size'])

scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)
features_tensor = torch.tensor(features_scaled, dtype=torch.float32)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
features_tensor = features_tensor.to(device)

# DataLoader setup
dataset = TensorDataset(features_tensor)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

# Initialize and pretrain the Autoencoder
input_dim = features_tensor.shape[1]
hidden_dims = [500, 500, 200, 10]
autoencoder = Autoencoder(input_dim, hidden_dims)

def pretrain_autoencoder(autoencoder, dataloader, epochs=100, lr=1e-3):
    autoencoder.to(device)
    optimizer = optim.Adam(autoencoder.parameters(), lr=lr)
    criterion = nn.MSELoss()
    autoencoder.train()
    
    for epoch in range(epochs):
        total_loss = 0
        for batch in dataloader:
            x_batch, = batch
            x_batch = x_batch.to(device)
            _, decoded = autoencoder(x_batch)
            loss = criterion(decoded, x_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        if epoch % 10 == 0:
            print(f'Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(dataloader):.4f}')

# Pretrain the autoencoder
pretrain_autoencoder(autoencoder, dataloader)

# Initialize DEC model
n_clusters = 5
dec_model = DEC(input_dim, hidden_dims, n_clusters)

# Load pretrained autoencoder weights into DEC model
dec_model.autoencoder.load_state_dict(autoencoder.state_dict())

def train_dec(dec, dataloader, epochs=200, lr=1e-3):
    dec.to(device)
    optimizer = optim.Adam(dec.parameters(), lr=lr)
    criterion = nn.KLDivLoss(reduction='sum')
    dec.train()

    for epoch in range(epochs):
        total_loss = 0
        for batch in dataloader:
            x_batch, = batch
            x_batch = x_batch.to(device)
            cluster_assignment, decoded = dec(x_batch)

            # Calculate the target distribution
            p = target_distribution(cluster_assignment).detach()

            # Calculate the loss and optimize
            loss = criterion(torch.log(cluster_assignment), p)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        if epoch % 10 == 0:
            print(f'Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(dataloader):.4f}')

# Train the DEC model
train_dec(dec_model, dataloader)

# Load the pre-trained DEC model
dec_model.eval()

# Button for user to submit sizes
if st.button("Submit and Get Recommendations"):
    # Combine user inputs into a feature vector and apply weights
    user_features = torch.tensor([[height, weight, chest, waist, hip, calculate_bmi(height, weight)] + 
                                  [1 if base_size == size else 0 for size in ['XS', 'S', 'M', 'L', 'XL', 'XXL']] + 
                                  [apply_weights(df, height, weight, chest, waist, hip, gender, age, brand, fit_feedback, return_exchange_status, loyalty_status)]], 
                                  dtype=torch.float32)
    user_features = user_features.to(device)

    # Predict cluster assignment
    with torch.no_grad():
        encoded, _ = dec_model.autoencoder(user_features)
        cluster_assignments = dec_model.clustering(encoded).argmax(dim=1).item()

    st.write(f"**Your assigned cluster:** {cluster_assignments}")

    # Provide recommendations for one product from each brand based on the assigned cluster
    st.write("### Product Recommendations")
    
    size_charts = {
        0: {"Height (cm)": 170, "Weight (kg)": 65, "Chest (cm)": 95, "Waist (cm)": 75, "Hip (cm)": 90},
        1: {"Height (cm)": 175, "Weight (kg)": 70, "Chest (cm)": 100, "Waist (cm)": 80, "Hip (cm)": 95},
        2: {"Height (cm)": 180, "Weight (kg)": 85, "Chest (cm)": 105, "Waist (cm)": 90, "Hip (cm)": 100},
        3: {"Height (cm)": 165, "Weight (kg)": 60, "Chest (cm)": 90, "Waist (cm)": 70, "Hip (cm)": 85},
        4: {"Height (cm)": 185, "Weight (kg)": 95, "Chest (cm)": 110, "Waist (cm)": 100, "Hip (cm)": 105}
    }

    if cluster_assignments in size_charts:
        for brand_name, size_chart in size_charts.items():
            recommended_size = get_brand_size(brand, base_size)
            st.write(f"**{brand}:** {recommended_size} - Recommended for {item_type} based on your size.")
    else:
        st.write("Sorry, we couldn't find a recommended size chart for your measurements.")

    # Save the size chart as a CSV file
    if st.button("Save Size Chart"):
        if cluster_assignments in size_charts:
            size_chart_df = pd.DataFrame([size_charts[cluster_assignments]])
            size_chart_df.to_csv('recommended_size_chart.csv', index=False)
            st.write("Size chart saved as `recommended_size_chart.csv`.")
        else:
            st.write("No size chart to save.")

import streamlit as st
import numpy as np

def generate_gbm(n_simulations, n_steps, initial_price, mu, sigma, time_interval, interest_rate, conversion_ratio, kN):
    dt = time_interval / n_steps
    sqrt_dt = np.sqrt(dt)

    stock_prices = np.zeros((n_simulations, n_steps + 1))
    stock_prices[:, 0] = initial_price

    for i in range(1, n_steps + 1):
        # Generate random numbers from a normal distribution
        random_numbers = np.random.normal(0, 1, size=n_simulations)
        # Calculate the stock price using the GBM formula
        stock_prices[:, i] = stock_prices[:, i - 1] * np.exp((mu - 0.5 * sigma**2) * dt + sigma * sqrt_dt * random_numbers)

    last_T = stock_prices[:, n_steps:n_steps+1]  # Last simulated stock prices

    CB = np.exp(-1 * interest_rate * time_interval) * np.maximum(kN, (conversion_ratio * last_T))

    Bond_Price = np.mean(CB)

    return Bond_Price

# Streamlit app
st.title("GBM Bond Pricing Simulator")

# Input widgets
n_simulations = st.slider("Number of Simulations", min_value=1, max_value=100, value=10)
n_steps = st.slider("Number of Steps", min_value=1, max_value=10, value=3)
initial_price = st.number_input("Initial Price", value=22.75)
mu = st.number_input("Mu", value=0.0526)
sigma = st.number_input("Sigma", value=0.55598 / np.sqrt(252))
time_interval = st.number_input("Time Interval", value=0.5)
interest_rate = st.number_input("Interest Rate", value=0.0526)
conversion_ratio = st.number_input("Conversion Ratio", value=37.037)
kN = st.number_input("kN", value=1000)

# Button to trigger the calculation
if st.button("Calculate Bond Price"):
    bond_price = generate_gbm(n_simulations, n_steps, initial_price, mu, sigma, time_interval, interest_rate, conversion_ratio, kN)
    st.success(f"Estimated Bond Price: {bond_price:.2f}")

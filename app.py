import streamlit as st
import numpy as np

def generate_gbm(n_simulations, n_steps, initial_price, mu, sigma, time_interval, dividend_yield):
    dt = time_interval / n_steps
    sqrt_dt = np.sqrt(dt)

    stock_prices = np.zeros((n_simulations, n_steps + 1))
    stock_prices[:, 0] = initial_price

    for i in range(1, n_steps + 1):
        # Generate random numbers from a normal distribution
        random_numbers = np.random.normal(0, 1, size=n_simulations)
        # Calculate the stock price using the GBM formula
        stock_prices[:, i] = stock_prices[:, i - 1] * np.exp((mu - dividend_yield - 0.5 * sigma**2) * dt + sigma * sqrt_dt * random_numbers)

    return stock_prices

def bond_price(yield_to_maturity, coupon_rate, maturity, periods_per_year, face_value):
    r = yield_to_maturity / periods_per_year  # Assume rate input is annual, and convert to periodic
    n = int(maturity * periods_per_year)  # Total number of periods, converted to an integer

    # Time points
    time_points = np.linspace(0, maturity, num=n+1)

    # Initialize array to store bond prices over time
    bond_prices = np.zeros(n+1)

    for i in range(1, n+1):
        t = time_points[i]
        # Calculate present value of coupon payments
        pv_coupons = (coupon_rate * face_value / periods_per_year * np.exp( -1 * r * t ))
        if t == time_points[-1]: #if time point is equal to maturity, include a cash flow for face value
          pv_face_value = face_value * np.exp( -1 * r * t )
        else:
          pv_face_value = 0

        # Total bond price at time t
        bond_prices[i] = pv_coupons + pv_face_value
    return np.sum(bond_prices) #sum cash flows as final price

def bond_prices_over_time(yield_to_maturity, coupon_rate, maturity, periods_per_year, face_value):
  #Number of periods to be simulated
  straight_bond_value = np.zeros(maturity*periods_per_year+1)
  count = 0
  #Calculates each time step until maturity
  reduced_maturities = np.arange(0, maturity + 1/periods_per_year, 1/periods_per_year)
  for decrease_step in reduced_maturities:
    #Calculate the bond price as if it were due on each of the time steps
    straight_bond_value[count] = bond_price(yield_to_maturity, coupon_rate, maturity - decrease_step, periods_per_year, face_value)
    count += 1

  #Value if bond was delivered now is just face value plus one coupon
  straight_bond_value[-1] = face_value + coupon_rate * face_value / periods_per_year
  return straight_bond_value

def run_all(n,maturity,initial_price,time_interval,interest_rate,periods_per_year,sigma,dividend_yield,conversion_ratio,coupon_rate,face_value,option_type,option_strike,exercise_period):
   # Stock simulation with GBM
    total_periods_simulated = maturity * periods_per_year
    stock_prices = generate_gbm(n, total_periods_simulated, initial_price, interest_rate, sigma, time_interval, dividend_yield)

    # Initiate the payoff array
    payoff = [[0.0] * (total_periods_simulated) for _ in range(n)]

    #Straight Bond calculation
    straight_bond_value = []
    straight_bond_value = bond_prices_over_time(interest_rate, coupon_rate, maturity, periods_per_year, face_value)

    #For each of the stock GBM paths, and each of the periods simulated, calculate payoff of convertible bond
    for path in range(0, n):
        for t in range(0, total_periods_simulated):
            if t < exercise_period:
                payoff[path][t] = (float(np.exp(-1 * interest_rate/periods_per_year * t) * np.maximum(straight_bond_value[t], (conversion_ratio*stock_prices[path:path + 1, t:t+1]))))
            else:
                if option_type == "Call":
                    payoff[path][t] = (float(np.exp(-1 * interest_rate/periods_per_year * t) * np.minimum(option_strike/100*face_value, np.maximum(straight_bond_value[t], (conversion_ratio*stock_prices[path:path + 1, t:t+1])))))
                else:
                    payoff[path][t] = (float(np.exp(-1 * interest_rate/periods_per_year * t) * np.maximum(option_strike/100*face_value, np.maximum(straight_bond_value[t], (conversion_ratio*stock_prices[path:path + 1, t:t+1])))))
                if payoff[path][t] == (option_strike/100*face_value):
                    payoff[path][t] = float(np.exp(-1 * interest_rate/periods_per_year * t) * payoff[path][t]) #If payoff is equal to option, discount it


    #######################    Results   #######################

    #Calculate convertible price
    Bond_Price = np.mean(payoff)
    print('Price: %s' % Bond_Price)
    return Bond_Price


# Streamlit app
st.title("Callable Convertible Bond Pricing Simulator")

n = st.number_input("Number of Simulations",value=100000)             # Number of simulations
maturity = st.number_input("Number of Years until Maturity",value=3)           # Number of years until maturity
initial_price = st.number_input("Initial Stock Price",value=22.75)  # Initial stock price
time_interval = st.number_input("Time Interval Per year",value=0.5)    # Time interval per year (semi-annual)
interest_rate = st.number_input("Annual Interest Rate",value=0.0526) # Annual interest rate
periods_per_year = np.int64(1/time_interval) # Semi-annual = 2 periods per year
sigma = 0.55598/np.sqrt(periods_per_year)    # Convert Annualized Volatility to period volatility
dividend_yield = st.number_input("Dividend Yield",value=0)   # Dividend yield needs to be inputted as a decimal
conversion_ratio = st.number_input("Conversion Ratio",value=37.037)
coupon_rate = st.number_input("Annual Coupon by Straight Bond",value=0.09)   # Annual coupon by straight bond
face_value = st.number_input("Face Value",value=1000)    # Face value of the bond
option_type = st.radio("Option Type",options=['Call','Put']) # Option type "Call" or "Put"
option_strike = st.number_input("Option Strike",value=100.00) #Needs to be inputted in 100 basis (% of face value).
exercise_period = st.number_input("Exercise Period",value=0) # This representes which step the embedded option starts to kick in.
                    # If we have 2 year maturity with 0.5 steps, then the inputs accepted should be [0,1,2,3,4]

if st.button("Calculate Bond Price"):
    bond_price = run_all(n,maturity,initial_price,time_interval,interest_rate,periods_per_year,sigma,dividend_yield,conversion_ratio,coupon_rate,face_value,option_type,option_strike,exercise_period)
    st.success(f"Estimated Bond Price: {bond_price:.2f}")


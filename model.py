import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import joblib

Clean_data = "power_consumption_cleaned.csv"

def train():
    df = pd.read_csv(Clean_data) 

    features = ['Global_reactive_power', 'Voltage', 'Global_intensity', 'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3', 'year', 'month', 'day', 'hour', 'minute'] 
    target = 'Global_active_power'

    X_train, X_test, y_train, y_test = train_test_split(
        df[features], df[target], test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = LinearRegression()
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    rmse = mean_squared_error(y_test, predictions, squared=False) 
    print(f"Root Mean Squared Error: {rmse}")

    joblib.dump(model, 'power_consumption_model.pkl')
    print("Model saved to power_consumption_model.pkl")
    return True

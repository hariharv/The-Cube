import pandas as pd
from tensorflow import keras

# 1) Load the trained model
model = keras.models.load_model("Mod8-16-8.h5")

# 2) Load your input data (replace with your actual input file)
X = pd.read_csv("input_features.csv")  # Must match model’s expected input shape
# TODO: Replace the above line with the correct path to your input_features.csv file or create the file with appropriate data.

# 3) Make predictions
y_pred = model.predict(X.values)

# 4) Save predictions to CSV
out = X.copy()
out["prediction"] = y_pred.ravel()
out.to_csv("predictions.csv", index=False)

print("✅ predictions.csv created — ready for Streamlit upload!")
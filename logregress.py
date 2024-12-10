import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import statsmodels.api as sm

# General logistic regression function
def binary_logistic_regression(X, y):
    """
    Fits a binary logistic regression model and analyzes results.

    Parameters:
        X: Features (2D array-like, pandas DataFrame, or numpy array)
        y: Target variable (binary, 1D array-like)

    Returns:
        model: Trained logistic regression model
        odds_ratios: Odds ratios for the predictors
    """
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Fit logistic regression model
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # Evaluate the model
    y_pred = model.predict(X_test)
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # Add constant for statsmodels
    X_const = sm.add_constant(X_train)
    sm_model = sm.Logit(y_train, X_const)
    results = sm_model.fit(disp=False)
    print("\nStatsmodels Summary:")
    print(results.summary())

    # Compute odds ratios
    odds_ratios = np.exp(results.params)
    print("\nOdds Ratios:")
    print(odds_ratios)

    return model, odds_ratios

# Example usage
if __name__ == "__main__":
    # Example dataset: Meme coin factors
    np.random.seed(42)
    n_samples = 100

    # Independent variables (features)
    market_cap = np.random.rand(n_samples) * 100  # Market capitalization in millions
    community_size = np.random.randint(1000, 50000, size=n_samples)  # Community size
    hype_index = np.random.rand(n_samples) * 10  # Hype index (e.g., social media activity)
    volatility = np.random.rand(n_samples)  # Volatility index

    # Dependent variable (graduated or not)
    graduated = np.random.choice([0, 1], size=n_samples, p=[0.7, 0.3])

    # Combine into a DataFrame
    data = pd.DataFrame({
        'MarketCap': market_cap,
        'CommunitySize': community_size,
        'HypeIndex': hype_index,
        'Volatility': volatility,
        'Graduated': graduated
    })

    # Split features and target
    X = data[['MarketCap', 'CommunitySize', 'HypeIndex', 'Volatility']]
    y = data['Graduated']

    # Fit and analyze logistic regression
    model, odds_ratios = binary_logistic_regression(X, y)

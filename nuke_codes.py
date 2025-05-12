def 
         Crosses  Long balls per game  Through balls per game  
count  911.000000           911.000000              911.000000  
mean     0.302195             2.253568                0.014380  
std      0.433584             1.369160                0.039802  
min      0.000000             0.000000                0.000000  
25%      0.000000             1.200000                0.000000  
50%      0.100000             2.000000                0.000000  
75%      0.400000             3.000000                0.000000  
max      3.000000             8.500000                0.300000  
    Aerials Won per game  Man of the match     Tackles  \
count            911.000000        911.000000  911.000000   
mean               1.685950          0.515917    1.544237   
std                1.040868          0.927906    0.683854   
min                0.000000          0.000000    0.000000   
25%                0.900000          0.000000    1.100000   
50%                1.600000          0.000000    1.500000   
75%                2.300000          1.000000    2.000000   
max                7.500000          8.000000    6.000000   


mid
        Crosses  Long balls per game  Through balls per game        
count  1405.00000          1405.000000             1405.000000  
mean      0.38911             1.429893                0.042491  
std       0.44860             1.225856                0.081271  
min       0.00000             0.000000                0.000000  
25%       0.10000             0.500000                0.000000  
50%       0.20000             1.100000                0.000000  
75%       0.60000             1.900000                0.100000  
max       2.90000             7.800000                1.000000  
       Aerials Won per game  Man of the match      Tackles  \
count           1405.000000       1405.000000  1405.000000   
mean               0.979502          0.828470     1.310890   
std                0.859408          1.515744     0.739265   
min                0.000000          0.000000     0.000000   
25%                0.400000          0.000000     0.800000   
50%                0.800000          0.000000     1.200000   
75%                1.300000          1.000000     1.800000   
max                7.500000         17.000000     6.500000 

att: 

     Aerials Won per game  Man of the match     Tackles  \
count            559.000000        559.000000  559.000000   
mean               1.248658          1.266547    0.633095   
std                1.320066          2.049252    0.525271   
min                0.000000          0.000000    0.000000   
25%                0.400000          0.000000    0.300000   
50%                0.800000          0.000000    0.500000   
75%                1.600000          2.000000    0.900000   
max                7.500000         17.000000    4.000000   
          Crosses  Long balls per game  Through balls per game  
count  559.000000           559.000000              559.000000  
mean     0.218784             0.526655                0.043828  
std      0.343325             0.660527                0.085049  
min      0.000000             0.000000                0.000000  
25%      0.000000             0.100000                0.000000  
50%      0.100000             0.300000                0.000000  
75%      0.300000             0.700000                0.100000  
max      2.900000             5.000000                0.900000  


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

def read_fifa_data(file_path):
    """Read all sheets from the FIFA Excel file"""
    try:
        excel_file = pd.ExcelFile(file_path)
        sheet_names = excel_file.sheet_names
        print(f"Available sheets: {sheet_names}")
        
        sheets_data = {}
        for sheet_name in sheet_names:
            sheets_data[sheet_name] = pd.read_excel(excel_file, sheet_name=sheet_name)
            print(f"\n{sheet_name} sheet loaded with {sheets_data[sheet_name].shape[0]} rows and {sheets_data[sheet_name].shape[1]} columns")
        
        return sheets_data
    except Exception as e:
        print(f"Error reading file: {e}")
        return None

def analyze_correlation_with_fifa_overall(df):
    """Analyze correlation between FIFA Overall rating and real-life metrics"""
    # Ensure "Fifa Ability Overall" is in the columns
    if "Fifa Ability Overall" not in df.columns:
        print("Error: 'Fifa Ability Overall' column not found in dataframe")
        return
    
    # Calculate correlation with all numeric columns
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    correlations = df[numeric_cols].corr()["Fifa Ability Overall"].sort_values(ascending=False)
    
    print("\n--- Correlation with FIFA Overall Rating ---")
    print(correlations)
    
    # Plot top correlations
    plt.figure(figsize=(12, 8))
    top_corr = correlations.drop("Fifa Ability Overall")  # Remove self-correlation
    top_corr = top_corr.iloc[np.argsort(np.abs(top_corr.values))[-15:]]  # Top 15 by absolute value
    
    sns.barplot(x=top_corr.values, y=top_corr.index)
    plt.title('Top 15 Correlations with FIFA Overall Rating')
    plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)
    plt.grid(axis='x', alpha=0.3)
    plt.show()
    
    # Return the metric most correlated with FIFA Overall
    most_correlated = correlations.drop("Fifa Ability Overall").idxmax()
    print(f"\nMetric most positively correlated with FIFA Overall: {most_correlated} ({correlations[most_correlated]:.4f})")
    
    return most_correlated

def scatter_plot_with_fifa_overall(df, target_variable):
    """Create scatter plot of FIFA Overall vs the target variable"""
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x="Fifa Ability Overall", y=target_variable, data=df, alpha=0.6)
    
    # Add regression line
    sns.regplot(x="Fifa Ability Overall", y=target_variable, data=df, 
                scatter=False, color='red', line_kws={"linewidth": 2})
    
    plt.title(f'FIFA Overall Rating vs {target_variable}')
    plt.xlabel('FIFA Overall Rating')
    plt.ylabel(target_variable)
    plt.grid(alpha=0.3)
    plt.show()

def predict_with_fifa_overall(df, target_variable):
    """Build a simple regression model using FIFA Overall to predict target variable"""
    X = df[["Fifa Ability Overall"]].values
    y = df[target_variable].values
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Fit linear regression model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"\n--- Predicting {target_variable} using FIFA Overall Rating ---")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"R²: {r2:.4f}")
    
    # Plot predictions vs actual
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--')
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title(f'Actual vs Predicted {target_variable}')
    plt.grid(alpha=0.3)
    plt.show()
    
    return model, r2

def compare_with_multivariate_model(df, target_variable):
    """Compare FIFA Overall prediction with a model using multiple real-life features"""
    # Drop the target variable and FIFA Overall from potential features
    potential_features = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    if target_variable in potential_features:
        potential_features.remove(target_variable)
    if "Fifa Ability Overall" in potential_features:
        potential_features.remove("Fifa Ability Overall")
    
    # Select features based on correlation with target
    correlations = df[potential_features + [target_variable]].corr()[target_variable].abs().sort_values(ascending=False)
    selected_features = correlations.head(5).index.tolist()  # Top 5 most correlated features
    
    print(f"\n--- Top 5 features correlated with {target_variable} ---")
    print(correlations.head(5))
    
    X = df[selected_features].values
    y = df[target_variable].values
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train models
    models = {
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(),
        'Random Forest': RandomForestRegressor(random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(random_state=42)
    }
    
    results = {}
    best_r2 = -float('inf')
    best_model = None
    best_model_name = None
    
    print(f"\n--- Predicting {target_variable} using top 5 correlated features ---")
    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        results[name] = {
            'RMSE': rmse,
            'MAE': mae,
            'R²': r2
        }
        
        if r2 > best_r2:
            best_r2 = r2
            best_model = model
            best_model_name = name
        
        print(f"{name} - RMSE: {rmse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}")
    
    # Plot best model predictions
    y_pred_best = best_model.predict(X_test_scaled)
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred_best, alpha=0.5)
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--')
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title(f'Best Model ({best_model_name}): Actual vs Predicted {target_variable}')
    plt.grid(alpha=0.3)
    plt.show()
    
    # Feature importance for the best model
    if hasattr(best_model, 'feature_importances_'):
        importances = best_model.feature_importances_
        feature_importance = pd.DataFrame({
            'Feature': selected_features,
            'Importance': importances
        }).sort_values('Importance', ascending=False)
        
        plt.figure(figsize=(10, 6))
        sns.barplot(x='Importance', y='Feature', data=feature_importance)
        plt.title(f'Feature Importance for {best_model_name}')
        plt.grid(axis='x', alpha=0.3)
        plt.show()
        
        print("\nFeature Importance:")
        for _, row in feature_importance.iterrows():
            print(f"{row['Feature']}: {row['Importance']:.4f}")
    
    return results, best_model_name, best_r2

def explore_position_specific_data(sheets_data, target_variable):
    """Explore position-specific data (DEF, MID, OFF)"""
    position_sheets = {}
    for name, df in sheets_data.items():
        if name.upper() in ["DEF", "MID", "OFF", "ATT"]:
            position_sheets[name] = df
    
    if not position_sheets:
        print("No position-specific sheets found")
        return
    
    print("\n--- Position-Specific Analysis ---")
    for position, df in position_sheets.items():
        print(f"\nAnalyzing {position} sheet:")
        print(f"Shape: {df.shape}")
        
        # Check if the target variable and FIFA Overall are in this sheet
        if target_variable in df.columns and "Fifa Ability Overall" in df.columns:
            # Calculate correlation
            corr = df[["Fifa Ability Overall", target_variable]].corr().iloc[0, 1]
            print(f"Correlation between FIFA Overall and {target_variable}: {corr:.4f}")
            
            # Scatter plot
            plt.figure(figsize=(8, 5))
            sns.scatterplot(x="Fifa Ability Overall", y=target_variable, data=df, alpha=0.6)
            sns.regplot(x="Fifa Ability Overall", y=target_variable, data=df, 
                        scatter=False, color='red', line_kws={"linewidth": 2})
            plt.title(f'{position}: FIFA Overall Rating vs {target_variable}')
            plt.grid(alpha=0.3)
            plt.show()
            
            # Simple regression
            X = df[["Fifa Ability Overall"]].values
            y = df[target_variable].values
            
            # Train-test split (if we have enough data)
            if len(df) >= 30:  # Only proceed if we have enough data
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                
                # Fit model
                model = LinearRegression()
                model.fit(X_train, y_train)
                
                # Make predictions
                y_pred = model.predict(X_test)
                
                # Calculate metrics
                r2 = r2_score(y_test, y_pred)
                print(f"Linear Regression R² for {position}: {r2:.4f}")
        else:
            print(f"Required columns not found in {position} sheet")

def main():
    # Replace with your actual file path
    file_path = 'Data.xlsx'
    
    # Read all sheets from the file
    sheets_data = read_fifa_data(file_path)
    
    if not sheets_data:
        print("Failed to read the Excel file")
        return
    
    # Get the main data sheet
    main_data = sheets_data.get('Data')
    
    if main_data is None:
        print("Main Data sheet not found in the Excel file")
        return
    
    # Step 1: Find which real-life metric is most correlated with FIFA Overall
    print("\n--- STEP 1: Finding which real-life metric is most correlated with FIFA Overall ---")
    most_correlated_metric = analyze_correlation_with_fifa_overall(main_data)
    
    # Step 2: Create scatter plot
    print("\n--- STEP 2: Creating scatter plot for visual analysis ---")
    scatter_plot_with_fifa_overall(main_data, most_correlated_metric)
    
    # Step 3: Predict using FIFA Overall only
    print("\n--- STEP 3: Predicting using FIFA Overall only ---")
    _, fifa_r2 = predict_with_fifa_overall(main_data, most_correlated_metric)
    
    # Step 4: Compare with multivariate model
    print("\n--- STEP 4: Comparing with multivariate model ---")
    _, best_model_name, best_r2 = compare_with_multivariate_model(main_data, most_correlated_metric)
    
    # Step 5: Position-specific analysis
    print("\n--- STEP 5: Position-specific analysis ---")
    explore_position_specific_data(sheets_data, most_correlated_metric)
    
    # Print conclusion
    print("\n--- CONCLUSION ---")
    print(f"Target variable: {most_correlated_metric}")
    print(f"- FIFA Overall Rating alone explains {fifa_r2:.4f} (R²) of the variance")
    print(f"- Best multivariate model ({best_model_name}) explains {best_r2:.4f} (R²) of the variance")
    
    if best_r2 > fifa_r2:
        improvement = ((best_r2 - fifa_r2) / fifa_r2) * 100
        print(f"- The multivariate model improves prediction by {improvement:.2f}%")
        print("- This suggests FIFA Overall Rating alone is NOT a comprehensive digital twin of real-life performance")
    else:
        print("- FIFA Overall Rating is surprisingly effective as a predictor")
        print("- This supports the idea that FIFA could serve as a simplified digital twin of real-life performance")

if __name__ == "__main__":
    main()
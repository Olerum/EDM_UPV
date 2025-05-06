# ---- Load data -----------------------------------------------------------
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

clean_train = autoclean(train)
clean_test = autoclean(test)

X = clean_train.drop("SalePrice", axis=1)
Y = clean_train["SalePrice"]

# Split data into training and validation sets
X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, random_state=55)

# Define models with pipelines for consistent preprocessing
def create_pipeline(model):
    return Pipeline([
        ('scaler', StandardScaler()),
        ('model', model)
    ])

# ---- Train data ----------------------------------------------------------------------------------------------------------------------

random_state_nr = 55
models_with_pipeline = {
    "Decision Tree": create_pipeline(DecisionTreeRegressor(max_depth=10, random_state=55)),
    "Random Forest": create_pipeline(RandomForestRegressor(n_estimators=100, random_state=55)),
    "Gradient Boosting": create_pipeline(GradientBoostingRegressor(n_estimators=50, learning_rate=0.1, max_depth=3)),
    "XGBoost": create_pipeline(XGBRegressor(n_estimators=50, learning_rate=0.1, max_depth=3)),
    "Bagging Regressor": create_pipeline(BaggingRegressor(n_estimators=10, random_state=55)),
}

models = {
    "Decision Tree": DecisionTreeRegressor(max_depth=10, random_state=random_state_nr ),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=random_state_nr ),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=50, learning_rate=0.1, max_depth=3),
    "XGBoost": XGBRegressor(n_estimators=50, learning_rate=0.1, max_depth=3),
    "Bagging Regressor": BaggingRegressor(n_estimators=10, random_state=random_state_nr ),
}

print("-"*30 + f" RMSE values for {len(models)} models:" + "-" * 60)

#Train all models
results = {}
for name, pipeline in models.items():
    pipeline.fit(X_train, Y_train)
    y_pred = pipeline.predict(X_val)
    rmse = np.sqrt(mean_squared_error(Y_val, y_pred))
    results[name] = {"RMSE": rmse}
    print(f"{name}: RMSE: {rmse:.2f}")

#Sort the results by RMSE
sorted_results = sorted(results.items(), key=lambda x: x[1]["RMSE"])

for name, metrics in sorted_results:
    print(f"{name}: RMSE: {metrics['RMSE']:.2f}")

# Get the best model
best_model_name = sorted_results[0][0]
best_model = models[best_model_name]

print(f"\nBest model: {best_model_name} with RMSE: {results[best_model_name]['RMSE']:.2f}")

# ---- Train model with stacking of top models -------------------------------------------------------------------------------

number_of_top_models = 3

sorted_results = sorted(results.items(), key=lambda x: x[1]["RMSE"])
top_models = sorted_results[:number_of_top_models]  

#Combine the models
estimators = [(name, models[name]) for name, _ in top_models]
stacking_regressor = StackingRegressor(
    estimators=estimators,
    final_estimator=Ridge(),
    cv=5  # using 5-fold cross-validation
)

# Train and evaluate
stacking_regressor.fit(X_train, Y_train)
y_predicted_stacked = stacking_regressor.predict(X_val)
rmse_stacked = np.sqrt(mean_squared_error(Y_val, y_predicted_stacked))
print(f"\nStacking Regressor: RMSE: {rmse_stacked:.2f}")

# Train on entire datas
stacking_regressor.fit(X,Y)

best_model = stacking_regressor
# ---- Predict data with best model -----------------------------------------------------------
predictions = best_model.predict(clean_test)

submission = pd.DataFrame({
    'Id': test['Id'],
    'SalePrice': predictions
})
submission.to_csv('kaggle_submission_UPV34.csv', index=False)

print("Submission file has been created successfully.")

# # ---- Upload data -----------------------------------------------------------

# Write this in terminal 
#kaggle competitions submit titanic -f submission_autoskl.csv -m "submission_autoskl"
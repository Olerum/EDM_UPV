# ---- Load data -----------------------------------------------------------
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

clean_train = autoclean(train)
clean_test = autoclean(test)


# ---- Clean data ----------------------------------------------------------------------------------------------------------------------

X_train = clean_train.drop("SalePrice", axis=1)
Y_train = clean_train["SalePrice"]

test = test.drop(worst_attributes, axis=1)

# ---- Train data ----------------------------------------------------------------------------------------------------------------------

random_state_nr = 55
models = {
    "Decision Tree": DecisionTreeRegressor(max_depth=10, random_state=random_state_nr ),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=random_state_nr ),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=50, learning_rate=0.1, max_depth=3),
    "XGBoost": XGBRegressor(n_estimators=50, learning_rate=0.1, max_depth=3),
    "Bagging Regressor": BaggingRegressor(n_estimators=10, random_state=random_state_nr ),
}

print("-"*30 + f" RMSE values for {len(models)} models:" + "-" * 60)

results = {}

for name, model in models.items():

    model.fit(X_train, Y_train)
    y_predicted = model.predict(X_train)
    rmse = np.sqrt(mean_squared_error(Y_train, y_predicted))
    r2 = r2_score(Y_train, y_predicted)

    results[name] = {"RMSE": rmse}

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
    final_estimator=Ridge() 
)

# Train and evaluate
stacking_regressor.fit(X_train, Y_train)
y_predicted_stacked = stacking_regressor.predict(X_train)
rmse_stacked = np.sqrt(mean_squared_error(Y_train, y_predicted_stacked))

print(f"\nStacking Regressor: RMSE: {rmse_stacked:.2f}")

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


# # Prepare data for incremental learning
# remaining_train2 = train2.copy()
# increment_size = 20  # Number of instances added per iteration
# rmse_values = []
# num_instances = []

# # Initial evaluation before adding train2 instances
# initial_preds = best_model.predict(test.drop(columns=[target_column]))
# initial_rmse = np.sqrt(mean_squared_error(test[target_column], initial_preds))

# rmse_values.append(initial_rmse)
# num_instances.append(len(train1))

# # Incremental learning loop
# current_training_set = train1.copy()


# while not remaining_train2.empty:
#     # Select instances WITHOUT labels (random selection here, could be replaced with other unsupervised methods)
#     selected_indices = remaining_train2.sample(min(increment_size, len(remaining_train2))).index
#     selected_instances = remaining_train2.loc[selected_indices]

#     # Reveal labels (simulate acquiring labels from a teacher)
#     current_training_set = pd.concat([current_training_set, selected_instances])

#     # Remove selected instances from remaining train2
#     remaining_train2 = remaining_train2.drop(selected_indices)

#     # Retrain model
#     best_model.fit(current_training_set.drop(columns=[target_column]), current_training_set[target_column])

#     # Evaluate on test set
#     preds = best_model.predict(test.drop(columns=[target_column]))
#     rmse = np.sqrt(mean_squared_error(test[target_column], preds))

#     # Record performance
#     rmse_values.append(rmse)
#     num_instances.append(len(current_training_set))

    
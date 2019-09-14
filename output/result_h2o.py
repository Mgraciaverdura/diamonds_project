# Packages to be imported
import numpy as np
import pandas as pd
import os
import pandas as pd
from h2o.automl import H2OAutoML
import h2o
h2o.init()

# Data loading

print("Data loading... Please wait !")

train = h2o.import_file('dataset/diamonds_train.csv')
test = h2o.import_file('dataset/diamonds_test.csv')
sub_data = h2o.import_file('dataset/sample_submission.csv')

# Set train-test dataframes

print("Ok ! Remember that here our y is the feature price")

y = "price"
x = train.columns
x.remove(y)

run_automl_for_seconds = 18000
balance_classes= True #looking for the best features
aml = H2OAutoML(max_runtime_secs = run_automl_for_seconds, #corremos el modelo
                #weights_column = weight_col,
                balance_classes = True,
                sort_metric = "RMSE", #metric
                include_algos = ["GBM", "XGBoost"] #GradientBoost #XGBoost
               )

# We make the split of training and validation
train_final, valid = train.split_frame(ratios=[0.8])

aml.train(x=x, y =y, training_frame=train_final, validation_frame=valid)

leader_model = aml.leader
pred = leader_model.predict(test_data=test)
print(pred)

# Setup prediction as dataframe
pred_pd = pred.as_data_frame() #our prediction
sub = sub_data.as_data_frame() #for our submission

perf = aml.leader.model_performance(valid)
print(perf)

sub['price'] = pred_pd
h2o_test = pd.DataFrame(sub.price)
h2o_test.index.name = "id"

display(h2o_test)

# Final setup and saved as .csv
print("Congratulations ! Your .csv is ready.")
h2o_test.to_csv('output/result.csv')

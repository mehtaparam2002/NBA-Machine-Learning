import numpy as np
import pandas as pd
import csv
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import tensorflow as tf
import matplotlib.pyplot as plt



dataset = pd.read_csv("NBA Per Month Data 1996-2015 - Full Year 1996-1997.csv")
dataset = dataset.dropna()
print(dataset)

x = dataset.drop('POM',axis=1)
y_labels = dataset['POM']

assert not np.any(np.isnan(x)) #makes sure there are no missing values in the data

#sizes = [0,.1,.2,.3,.4,.5,.6,.7,.8,.9,1.0]

X_train, X_test, y_train, y_test = train_test_split(x, y_labels, test_size=0.2, random_state=101)

age = tf.feature_column.numeric_column("AGE")
games_played = tf.feature_column.numeric_column("GP")
wins = tf.feature_column.numeric_column("W")
losses = tf.feature_column.numeric_column("L")
minutes = tf.feature_column.numeric_column("MIN")
points = tf.feature_column.numeric_column("PTS")
field_goals_made = tf.feature_column.numeric_column("FGM")
field_goals_attempted = tf.feature_column.numeric_column("FGA")
#field_goals_percentage = tf.feature_column.numeric_column("FGPercent")
three_point_made = tf.feature_column.numeric_column("3PM")
three_point_attempted = tf.feature_column.numeric_column("3PA")
#three_point_percentage = tf.feature_column.numeric_column("3PPercent")
free_throw_made = tf.feature_column.numeric_column("FTM")
#free_throw_attempted = tf.feature_column.numeric_column("FTA")
#free_throw_percentage = tf.feature_column.numeric_column("FTPercent")
offensive_rebound = tf.feature_column.numeric_column("OREB")
defensive_rebound = tf.feature_column.numeric_column("DREB")
rebounds= tf.feature_column.numeric_column("REB")
assists = tf.feature_column.numeric_column("AST")
turnovers = tf.feature_column.numeric_column("TOV")
steals = tf.feature_column.numeric_column("STL")
blocks = tf.feature_column.numeric_column("BLK")
personal_fouls = tf.feature_column.numeric_column("PF")
fantasy_points = tf.feature_column.numeric_column("FP")
double_doubles = tf.feature_column.numeric_column("DD2")
tripple_doubles = tf.feature_column.numeric_column("TD3")
plus_minus = tf.feature_column.numeric_column("Plus_Minus")


feat_cols = [age, games_played, wins, losses, minutes, points, field_goals_made, field_goals_attempted, three_point_made, three_point_attempted, free_throw_made, offensive_rebound,defensive_rebound, rebounds, assists,turnovers,steals, blocks, personal_fouls, fantasy_points,double_doubles,tripple_doubles]

input_func = tf.compat.v1.estimator.inputs.pandas_input_fn(x=X_train,y=y_train,batch_size=10,num_epochs=200,shuffle=True)
model = tf.estimator.DNNClassifier(feature_columns=feat_cols, hidden_units=[5,10], model_dir='stanford_models')
for i in range(300):
  model.train(input_fn=input_func,steps=10)

pred_fn = tf.compat.v1.estimator.inputs.pandas_input_fn(x=X_test,batch_size=len(X_test),shuffle=False)
predictions = list(model.predict(input_fn=pred_fn))
final_preds = []
for pred in predictions:
    final_preds.append(pred['class_ids'][0])

print(classification_report(y_test,final_preds))


#my_stats = [4,4.8,1600,0.99]

#data = pd.DataFrame({'UW': [my_stats[0]], 'W': [my_stats[1]], 'SAT': [my_stats[2]],
#                     'Rank': [my_stats[3]]})


pred_fn = tf.compat.v1.estimator.inputs.pandas_input_fn(x=X_test,num_epochs=1,shuffle=False)
pred_gen = list(model.predict(input_fn=pred_fn))
graph_data = []
for i in range(len(pred_gen)):
    graph_data.append(pred_gen[i-1]['logistic'])
    plt.hist(graph_data,density=1, bins=20)
    plt.axis([0, 1, 0, 0.03])
    plt.xlabel('Bins')
    plt.ylabel('Data')
    plt.savefig('tensorflowplot/')



plt.show()


#likelyhood = round(likelyhood[0]*100,2)
#print('\n')
#print ("You are "+str(likelyhood)+"% likely to get into Stanford")

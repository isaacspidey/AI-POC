# import necessary libraries
import pandas as pd
from sklearn.linear_model import LogisticRegression

# read in the csv file
data = pd.read_csv('patient_data.csv')

# select the features we want to use for classification
X = data[['weight', 'age', 'calories', 'exercise']]
y = data['heart_attack']

# create the Logistic Regression model
clf = LogisticRegression()

# fit the model to the data
clf.fit(X, y)

# set up a while loop to continuously ask for new patient data until the user chooses to stop
while True:
  # ask the user for the new patient's data and store it as a list of floats
  new_patient_input = input("Enter the new patient's data (weight, age, calories, exercise): ")
  new_patient_data = [float(x) for x in new_patient_input.split(',')]
  
  # predict the label of the new patient data
  prediction = clf.predict([new_patient_data])[0]

  # output the prediction (0 for no heart attack, 1 for heart attack)
  # output the prediction and probability (0 for no heart attack, 1 for heart attack)

  # predict the probability of the new patient data having a heart attack
  heart_attack_probability = round(clf.predict_proba([new_patient_data])[0][1] * 100)
  heart_attack_probability = min(100, heart_attack_probability)

  print('Prediction: Heart attack probability:', heart_attack_probability, '%')

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

import pandas as pd
import pickle

# Here is only a template for your reference, you only need to ensure the predict function can 
# receive the test dataset and return the prediction results.



class MachineLearningModel():
    def __init__(self):
        self.model = LogisticRegression(max_iter=1000)


    def train(self, data):
        #you can do your training here
        self.model = self.model.fit(data.drop('Label', axis=1), data['Label'])
        with open('final_bagged_0.915_stacking_model.pkl', 'rb') as model_file:
            self.model = pickle.load(model_file)

    # def preprocess(self, data):
        # you can do your preprocessing here

    def predict(self, data):
        df_train = pd.read_csv('../data/train.csv')
        # Apply any data preprocessing if you want, just to keep the test data in the same format as the training data
        self.train(df_train)
        return self.model.predict(data)
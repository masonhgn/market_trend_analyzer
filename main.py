import yfinance as yf
import pylab as p
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score
import pandas as pd


class TrendPredictionModel:
 	def __init__(self, ticker):
 		self.ticker = ticker
 		self.data = yf.Ticker(ticker)
 		self.data = self.data.history(period = "max")
 		del self.data["Dividends"]
 		del self.data["Stock Splits"]

 		self.data["Tomorrow"] = self.data["Close"].shift(-1)
 		self.data["Goal"] = (self.data["Tomorrow"] > self.data["Close"]).astype(int)
 		self.data = self.data.loc["1990-01-01":].copy()

 		self.model = RandomForestClassifier(n_estimators = 100, min_samples_split = 100, random_state = 1)
 		self.training_set = self.data.iloc[:-100]
 		self.test_set = self.data.iloc[-100:]

 		self.predictors = ["Close", "Volume", "Open", "Low", "High"]


 	def getData(self): return self.data

 	def getDataShapeSize(self): return self.data.shape[0]

 	def getModel(self): return self.model

 	def getPredictors(self): return self.predictors


 	def printHistory(self):
 		print(self.data)

 	def predict(self):
 		self.model.fit(self.training_set[self.predictors], self.training_set["Goal"])
 		self.predictions = self.model.predict(self.test_set[self.predictors])
 		self.predictions = pd.Series(self.predictions, index = self.test_set.index, name = "Predictions")
 		total = pd.concat([self.test_set["Goal"], self.predictions], axis = 1)
 		return total


 	def backtest(self, start = 2500, step = 250):
 		all_predictions = []

 		for i in range(start, self.data.shape[0], step):
 			train = self.data.iloc[0:i].copy()
 			test = self.data.iloc[i:(i+step)].copy()
 			new_predictions = self.predict()
 			all_predictions.append(new_predictions)
 		return pd.concat(all_predictions)


def main():
	print("Welcome to my stock market trend prediction model.")
	ticker = input("Please enter a ticker...")
	MyModel = TrendPredictionModel(ticker)

	predictions = MyModel.backtest()

	print(predictions["Predictions"].value_counts())

	print("precision score: ",precision_score(predictions["Goal"], predictions["Predictions"]))


if __name__ == "__main__":
	main()


# univariate lstm example
from numpy import array
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Bidirectional
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import pickle

# split a univariate sequence
def split_sequence(sequence, n_steps):
	X, y = list(), list()
	for i in range(len(sequence)):
		# find the end of this pattern
		end_ix = i + n_steps
		# check if we are beyond the sequence
		if end_ix > len(sequence)-1:
			break
		# gather input and output parts of the pattern
		seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
		X.append(seq_x)
		y.append(seq_y)
	return array(X), array(y)


dataset_train = pd.read_csv(r"C:\Users\CCM Laptop\Desktop\GHI-DATA\capstoneproject 10mins Joburg training dataset.csv")
training_set = dataset_train.iloc[:,4:5].values


# choose a number of time steps
n_steps = 3
# split into samples
X, y = split_sequence(training_set, n_steps)


# reshape from [samples, timesteps] into [samples, timesteps, features]
n_features = 1
X = X.reshape((X.shape[0], X.shape[1], n_features))


# define model
model = Sequential()
model.add(Bidirectional(LSTM(50, activation='relu'), input_shape=(n_steps, n_features)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')


# fit model
history = model.fit(X, y, validation_split = 0.2, epochs= 100, batch_size = 16)


#Saving the model with pickle

filename1 = "finalized_model_radiation2.sav"
pickle.dump(model, open(filename1, "wb"))



# demonstrate prediction
actual_solar_radiation = array([96,
123,
144,
])
actual_solar_radiation= actual_solar_radiation.reshape((1, n_steps, n_features))

predicted_solar_radiation = model.predict(actual_solar_radiation, verbose=0)

print(predicted_solar_radiation)


#Predicting the solar power that will be generated
capacity_of_one_panel = array([250])

N = number_of_solar_panels = array([3])

A = area_of_panel = array([12])


r = yield_of_one_panel = (capacity_of_one_panel/(area_of_panel * 10))

PR= default_performance_losses = array([0.75]) #performance ratio (pr) of a panel ranges from 0.5 to 0.9



predicted_electrical_power = ((predicted_solar_radiation * area_of_panel  * yield_of_one_panel * number_of_solar_panels *  default_performance_losses)/1000)

print(predicted_electrical_power)


#Visualizing the performance of the model
plt.plot(history.history['loss'], color = 'red', label = 'MSE loss')
plt.plot(history.history['val_loss'], color = 'blue', label = 'epochs')
plt.title('10mins Ahead Johannesburg Solar Radiation Prediction')
plt.ylabel('MSE loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')
plt.show()

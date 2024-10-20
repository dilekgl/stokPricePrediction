from tensorflow.keras.models import load_model
import numpy as np
# Kaydedilen modeli yükle
loaded_model = load_model('model_train.h5')

# Modelin özetini görmek için
#loaded_model.summary()

#features = [Open, High, Low, Adj Close, Volume]
features = np.array([[177.089996, 180.419998, 177.070007, 74919600]])
print(loaded_model.predict(features))

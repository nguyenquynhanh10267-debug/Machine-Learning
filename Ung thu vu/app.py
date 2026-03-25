from flask import Flask, request, render_template
import pandas as pd 
import numpy as np
import pickle
model = pickle.load(open('model.pkl', 'rb')) #chạy file model.pkl để load model đã được train
# flask app
app = Flask(__name__)
@app.route('/') #định nghĩa route cho đường dẫn gốc
def index():
    return render_template('index.html') #trả về file index.html khi truy cập vào đường dẫn gốc
@app.route('/predict', methods=['POST']) #định nghĩa route cho đường dẫn /predict và chỉ chấp nhận phương thức POST
def predict():
    features = request.form['feature']
    features = features.split(',')
    np_features = np.asarray(features, dtype=np.float32)

    # prediction
    pred = model.predict(np_features.reshape(1, -1))
    message = ['Cancrouse' if pred[0] == 1 else 'Not Cancrouse']
    # print(message[0])
    return render_template('index.html', message=message)



# python main function
if __name__ == '__main__':
    app.run(debug=True)

from flask import Flask,request,render_template
import dill
import numpy as np

app = Flask('__name__')
@app.route('/')
def read_main():
    return render_template('index.html')

@app.route('/predict',methods=['GET'])
def generate_output():
    json_data = False
    input_data = request.args.get('data')
    if input_data==None:
        input_data = request.get_json()
        json_data = True
    Output = process_and_predict(input_text=input_data,json_data=json_data)
    return {'predicted': str(Output)}

def process_and_predict(input_text,json_data):
    if(json_data==True):
        output_text = [int(item) for item in input_text['data'].split(',')]
    else:
        output_text = [int(item) for item in input_text.split(',')]
    with open('src/models/preprocessor.pkl','rb') as p:
        preprocessor = dill.load(p)
    output_text = np.array(output_text).reshape(1, -1)
    output_text_dims = preprocessor.transform(output_text)
    with open('src/models/model.pkl', 'rb') as m:
        model = dill.load(m)
    disease_prediction = model.predict(output_text_dims)
    with open('src/models/cat_preprocessor.pkl','rb') as f:
        cat_preprocessor = dill.load(f)
    disease_predicted = cat_preprocessor.classes_[int(disease_prediction)]
    return disease_predicted

if __name__=='__main__':
    app.run(host='0.0.0.0',port=5000)
from flask import Flask, render_template, request, flash
from wtforms import Form, FloatField, SubmitField, validators, ValidationError
import numpy as np
from sklearn.externals import joblib

def predict(parameters):
    # NNモデルの読み込み
    model = joblib.load('./nn.pkl')
    # params = parameters
    params = parameters.reshape(1, -1)
    pred = model.predict(params)
    return pred


def getName(label):
    print(label)
    if label == 0:
        return 'Sentosa'
    elif label == 1:
        return 'Versicolor'
    elif label == 2:
        return 'Verginia'
    else:
        return 'Error'


app = Flask(__name__)
app.config.from_object(__name__)


class IrisForm(Form):
    SepalLength = FloatField('Sepal Lenght(cm)(がくの長さ)',
                            [validators.InputRequired('この項目は入力必須です'),
                            validators.NumberRange(min=0, max=10)])

    SepalWidth = FloatField('Sepal Width(cm)(がくの幅)',
                            [validators.InputRequired('この項目は入力必須です'),
                            validators.NumberRange(min=0, max=10)])

    PetalLength = FloatField('Petal Lenght(cm)(花弁の長さ)',
                            [validators.InputRequired('この項目は入力必須です'),
                            validators.NumberRange(min=0, max=10)])

    PetalWidth = FloatField('Petal Width(cm)(花弁の幅)',
                            [validators.InputRequired('この項目は入力必須です'),
                            validators.NumberRange(min=0, max=10)])

    submit = SubmitField('判定')

@app.route('/', methods = ['GET', 'POST'])
def predict_model():
    form = IrisForm(request.form)
    if request.method == 'POST':
        if form.validate() == False:
            flash('全て入力する必要があります。')
            return render_template('index.html', form=form)
        else:
            SepalLength = float(request.form['SepalLength'])
            SepalWidth = float(request.form['SepalWidth'])
            PetalLength = float(request.form['PetalLength'])
            PetalWidth = float(request.form['PetalWidth'])

            x = np.array([SepalLength, SepalWidth, PetalLength, PetalWidth])
            pred = predict(x)
            irisName = getName(pred)

            return render_template('result.html', irisName=irisName)
    elif request.method == 'GET':

        return(render_template('index.html', form=form))

if __name__ == '__main__':
    app.run()

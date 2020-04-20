from flask import Flask,render_template, url_for, request, redirect, flash
import model
from helper import cmdRun
from werkzeug.utils import secure_filename
from importlib import reload

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = "\data"



@app.route('/')
def index():
        return render_template('index.html')


@app.route('/predictCase', methods=['GET', 'POST'])
def predictCase():
    if request.method == 'POST':
        selectedReg = request.form.get('country')
        totalpos = request.form.get('totalpos')
        currentPos = request.form.get('currentPos')
        fecha = request.form.get('fecha')
        info = {
            "selectedRegion": selectedReg,
            "totalPositive": totalpos,
            "currentPositive": currentPos,
            "date": fecha,
        }
        value = model.convertToCsv(info)
        linD = str(value[0])
        linD = linD.replace('[[', '')
        linD = linD.replace(']]', '')

        ridgeD = str(value[1])
        ridgeD = ridgeD.replace('[[', '')
        ridgeD = ridgeD.replace(']]', '')
        info['linD'] = linD
        info['ridgeD'] = ridgeD
        # newCasePieChart(info)

        return render_template('caseresults.html', deathsLin=linD, deathsRidge=ridgeD, info=info)

    else:
        return render_template('newcase.html', titulo="Registrar nuevo caso", btn="Evaluar")



@app.route('/newDataset', methods=['GET', 'POST'])
def newDataset():

    if request.method == 'POST' :
        f = request.files['file']
        f.save(secure_filename(f.filename))
        # training(f.filename)
        print("1",model.filename)
        # model.filename = f.filename
        model.filename = f.filename
        reload(model)
        # import model
        print("2",model.filename)
        return redirect('/')
    else: 
        return render_template('fileupload.html')




if __name__ == "__main__":
    app.run(debug=True)



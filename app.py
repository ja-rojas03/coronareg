from flask import Flask,render_template, url_for, request, redirect
from model import convertToCsv

app = Flask(__name__)


@app.route('/')
def index():
        return render_template('index.html')


@app.route('/newCase', methods=['GET', 'POST'])
def newCase():
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
        value = convertToCsv(info)
        value = str(value)
        value = value.replace('[[', '')
        value = value.replace(']]', '')

        return render_template('caseresults.html', deaths=value)

    else:
        return render_template('newcase.html', titulo="Registrar nuevo dia", btn="Registrar")


@app.route('/predictCase', methods=['GET', 'POST'])
def predictCase():
    return render_template('newcase.html', titulo="Registre datos del caso", btn="Evaluar")


if __name__ == "__main__":
    app.run(debug=True)



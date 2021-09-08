import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import os
from flask import Flask, render_template, request, redirect, url_for, session
from flask_mysqldb import MySQL
import MySQLdb.cursors
import re

app = Flask(__name__)
app.secret_key = 'your secret key'

app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = 'abc456'
app.config['MYSQL_DB'] = 'logindb'

mysql = MySQL(app)



model = pickle.load(open('model.pkl', 'rb'))
humid= pickle.load(open('humid.pkl','rb'))
heat = pickle.load(open('heat.pkl','rb'))
dew = pickle.load(open('dew.pkl','rb'))
AQI=pickle.load(open('AQI.pkl','rb'))
PM25=pickle.load(open('PM25.pkl','rb'))
PM10=pickle.load(open('PM10.pkl','rb'))
@app.route('/')
def home_try_a():
    return render_template('home_try_a.html')

@app.route('/login', methods =['GET', 'POST'])
def login():
	msg = ''
	if request.method == 'POST' and 'username' in request.form and 'password' in request.form:
		username = request.form['username']
		password = request.form['password']
		cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
		cursor.execute('SELECT * FROM accounts WHERE username = % s AND password = % s', (username, password, ))
		account = cursor.fetchone()
		if account:
			session['loggedin'] = True
			session['id'] = account['id']
			session['username'] = account['username']
			msg = 'Logged in successfully !'
			return render_template('/home_final.html', msg = msg)
		else:
			msg = 'Incorrect username / password !'
	return render_template('login.html', msg = msg)

@app.route('/logout')
def logout():
	session.pop('loggedin', None)
	session.pop('id', None)
	session.pop('username', None)
	return redirect(url_for('login'))

@app.route('/register', methods =['GET', 'POST'])
def register():
	msg = ''
	if request.method == 'POST' and 'username' in request.form and 'password' in request.form and 'email' in request.form :
		username = request.form['username']
		password = request.form['password']
		email = request.form['email']
		cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
		cursor.execute('SELECT * FROM accounts WHERE username = % s', (username, ))
		account = cursor.fetchone()
		if account:
			msg = 'Account already exists !'
		elif not re.match(r'[^@]+@[^@]+\.[^@]+', email):
			msg = 'Invalid email address !'
		elif not re.match(r'[A-Za-z0-9]+', username):
			msg = 'Username must contain only characters and numbers !'
		elif not username or not password or not email:
			msg = 'Please fill out the form !'
		else:
			cursor.execute('INSERT INTO accounts VALUES (NULL, % s, % s, % s)', (username, password, email, ))
			mysql.connection.commit()
			msg = 'You have successfully registered !'
	elif request.method == 'POST':
		msg = 'Please fill out the form !'
	return render_template('register.html', msg = msg)
@app.route('/home_final',methods=["GET",'POST'])
def home_final():
    return render_template('/home_final.html')

@app.route('/weather_main.html',methods=["GET",'POST']) 

def weather_main():  
    return render_template("/weather_main.html");  

@app.route('/weather_predict',methods=["GET",'POST'])
def weather_predict():
    if request.method == 'POST': 
        month = request.form.get("month")
        if(month=="August-2020"):
            prediction = model.Forecast.get(key='2020-08-01')
            output=prediction
            pred_hum=humid.Forecast.get(key='2020-08-01')
            pred_heat=heat.Forecast.get(key='2020-08-01')
            pred_dew=dew.Forecast.get(key='2020-08-01')
            return render_template('weather_predict.html',pred_temp='Temperature for August 2020= {}'.format(output),
            pred_humid='Humidity for August 2020= {}'.format(pred_hum),
            pred_heat='Heat Index for August 2020= {}'.format(pred_heat),
            pred_dew='Dew point for August 2020= {}'.format(pred_dew))
        elif(month=="Sept-2020"):
            prediction = model.Forecast.get(key='2020-09-01')
            output=prediction
            pred_hum=humid.Forecast.get(key='2020-09-01')
            pred_heat=heat.Forecast.get(key='2020-09-01')
            pred_dew=dew.Forecast.get(key='2020-09-01')
            return render_template('weather_predict.html',pred_temp='Temperature for Semptember 2020= {}'.format(output),
            pred_humid='Humidity for September 2020= {}'.format(pred_hum),
            pred_heat='Heat Index for September 2020= {}'.format(pred_heat),
            pred_dew='Dew point for September 2020= {}'.format(pred_dew))
        elif(month=="Oct-2020"):
            prediction = model.Forecast.get(key='2020-10-01')
            output=prediction
            pred_hum=humid.Forecast.get(key='2020-10-01')
            pred_heat=heat.Forecast.get(key='2020-10-01')
            pred_dew=dew.Forecast.get(key='2020-10-01')
            return render_template('weather_predict.html',pred_temp='Temperature for October 2020= {}'.format(output),
            pred_humid='Humidity for October 2020= {}'.format(pred_hum),
            pred_heat='Heat Index for October 2020= {}'.format(pred_heat),
            pred_dew='Dew point for October 2020= {}'.format(pred_dew))
        elif(month=="Nov-2020"):
            prediction = model.Forecast.get(key='2020-11-01')
            output=prediction
            pred_hum=humid.Forecast.get(key='2020-11-01')
            pred_heat=heat.Forecast.get(key='2020-11-01')
            pred_dew=dew.Forecast.get(key='2020-11-01')
            return render_template('weather_predict.html',pred_temp='Temperature for November 2020= {}'.format(output),
            pred_humid='Humidity for November 2020= {}'.format(pred_hum),
            pred_heat='Heat Index for November 2020= {}'.format(pred_heat),
            pred_dew='Dew point for November 2020= {}'.format(pred_dew))
        elif(month=="Dec-2020"):
            prediction = model.Forecast.get(key='2020-12-01')
            output=prediction
            pred_hum=humid.Forecast.get(key='2020-12-01')
            pred_heat=heat.Forecast.get(key='2020-12-01')
            pred_dew=dew.Forecast.get(key='2020-12-01')
            return render_template('weather_predict.html',pred_temp='Temperature for December 2020= {}'.format(output),
            pred_humid='Humidity for December 2020= {}'.format(pred_hum),
            pred_heat='Heat Index for December 2020= {}'.format(pred_heat),
            pred_dew='Dew point for December 2020= {}'.format(pred_dew))
        elif(month=="Jan-2021"):
            prediction = model.Forecast.get(key='2021-01-01')
            output=prediction
            pred_hum=humid.Forecast.get(key='2021-01-01')
            pred_heat=heat.Forecast.get(key='2021-01-01')
            pred_dew=dew.Forecast.get(key='2021-01-01')
            return render_template('weather_predict.html',pred_temp='Temperature for January 2021= {}'.format(output),
            pred_humid='Humidity for January 2021= {}'.format(pred_hum),
            pred_heat='Heat Index for January 2021= {}'.format(pred_heat),
            pred_dew='Dew point for January 2021 = {}'.format(pred_dew))
        elif(month=="Feb-2021"):
            prediction = model.Forecast.get(key='2021-02-01')
            output=prediction
            pred_hum=humid.Forecast.get(key='2021-02-01')
            pred_heat=heat.Forecast.get(key='2021-02-01')
            pred_dew=dew.Forecast.get(key='2021-02-01')
            return render_template('weather_predict.html',pred_temp='Temperature for February 2021= {}'.format(output),
            pred_humid='Humidity for February 2021= {}'.format(pred_hum),
            pred_heat='Heat Index for February 2021= {}'.format(pred_heat),
            pred_dew='Dew point for February 2021 = {}'.format(pred_dew))
        elif(month=="March-2021"):
            prediction = model.Forecast.get(key='2021-03-01')
            output=prediction
            pred_hum=humid.Forecast.get(key='2021-03-01')
            pred_heat=heat.Forecast.get(key='2021-03-01')
            pred_dew=dew.Forecast.get(key='2021-03-01')
            return render_template('weather_predict.html',pred_temp='Temperature for March 2021= {}'.format(output),
            pred_humid='Humidity for March 2021= {}'.format(pred_hum),
            pred_heat='Heat Index for March 2021= {}'.format(pred_heat),
            pred_dew='Dew point for March 2021 = {}'.format(pred_dew))
        elif(month=="April-2021"):
            prediction = model.Forecast.get(key='2021-04-01')
            output=prediction
            pred_hum=humid.Forecast.get(key='2021-04-01')
            pred_heat=heat.Forecast.get(key='2021-04-01')
            pred_dew=dew.Forecast.get(key='2021-04-01')
            return render_template('weather_predict.html',pred_temp='Temperature for April 2021= {}'.format(output),
            pred_humid='Humidity for April 2021= {}'.format(pred_hum),
            pred_heat='Heat Index for April 2021= {}'.format(pred_heat),
            pred_dew='Dew point for April 2021 = {}'.format(pred_dew))
        elif(month=="May-2021"):
            prediction = model.Forecast.get(key='2021-05-01')
            output=prediction
            pred_hum=humid.Forecast.get(key='2021-05-01')
            pred_heat=heat.Forecast.get(key='2021-05-01')
            pred_dew=dew.Forecast.get(key='2021-05-01')
            return render_template('weather_predict.html',pred_temp='Temperature for May 2021= {}'.format(output),
            pred_humid='Humidity for May 2021= {}'.format(pred_hum),
            pred_heat='Heat Index for May 2021= {}'.format(pred_heat),
            pred_dew='Dew point for May 2021 = {}'.format(pred_dew))
        elif(month=="June-2021"):
            prediction = model.Forecast.get(key='2021-06-01')
            output=prediction
            pred_hum=humid.Forecast.get(key='2021-06-01')
            pred_heat=heat.Forecast.get(key='2021-06-01')
            pred_dew=dew.Forecast.get(key='2021-06-01')
            return render_template('weather_predict.html',pred_temp='Temperature for June 2021= {}'.format(output),
            pred_humid='Humidity for June 2021= {}'.format(pred_hum),
            pred_heat='Heat Index for June 2021= {}'.format(pred_heat),
            pred_dew='Dew point for June 2021 = {}'.format(pred_dew))
        elif(month=="July-2021"):
            prediction = model.Forecast.get(key='2021-07-01')
            output=prediction
            pred_hum=humid.Forecast.get(key='2021-07-01')
            pred_heat=heat.Forecast.get(key='2021-07-01')
            pred_dew=dew.Forecast.get(key='2021-07-01')
            return render_template('weather_predict.html',pred_temp='Temperature for July 2021= {}'.format(output),
            pred_humid='Humidity for July 2021= {}'.format(pred_hum),
            pred_heat='Heat Index for July 2021= {}'.format(pred_heat),
            pred_dew='Dew point for July 2021 = {}'.format(pred_dew))
        
        
        return render_template('weather_main.html')
@app.route('/pollution_main.html',methods=["GET",'POST']) 
def pollution_main():  
    return render_template("pollution_main.html");  

@app.route('/predict_poll1',methods=["GET",'POST'])
def predict_poll1():
    if request.method == 'POST': 
        month = request.form.get("month")
        if(month=="August-2020"):
            pred_AQI = AQI.Forecast.get(key='2020-08-01')
            pred_PM25=PM25.Forecast.get(key='2020-08-01')
            pred_PM10=PM10.Forecast.get(key='2020-08-01')
            return render_template('predict_poll1.html',month='August 2020',
            pred_AQI='AQI = {}'.format(pred_AQI),
            pred_PM25='PM2.5 = {}'.format(pred_PM25),
            pred_PM10='PM10 = {}'.format(pred_PM10))
        elif(month=="Sept-2020"):
            pred_AQI = AQI.Forecast.get(key='2020-09-01')
            pred_PM25=PM25.Forecast.get(key='2020-09-01')
            pred_PM10=PM10.Forecast.get(key='2020-09-01')
            return render_template('predict_poll1.html',month='September 2020',
            pred_AQI='AQI = {}'.format(pred_AQI),
            pred_PM25='PM2.5 = {}'.format(pred_PM25),
            pred_PM10='PM10 = {}'.format(pred_PM10))
        elif(month=="Oct-2020"):
            pred_AQI = AQI.Forecast.get(key='2020-10-01')
            pred_PM25=PM25.Forecast.get(key='2020-10-01')
            pred_PM10=PM10.Forecast.get(key='2020-10-01')
            return render_template('predict_poll1.html',month='October 2020',
            pred_AQI='AQI = {}'.format(pred_AQI),
            pred_PM25='PM2.5 = {}'.format(pred_PM25),
            pred_PM10='PM10 = {}'.format(pred_PM10))
        elif(month=="Nov-2020"):
            pred_AQI = AQI.Forecast.get(key='2020-11-01')
            pred_PM25=PM25.Forecast.get(key='2020-11-01')
            pred_PM10=PM10.Forecast.get(key='2020-11-01')
            return render_template('predict_poll1.html',month='November 2020',
            pred_AQI='AQI = {}'.format(pred_AQI),
            pred_PM25='PM2.5 = {}'.format(pred_PM25),
            pred_PM10='PM10 = {}'.format(pred_PM10))
        elif(month=="Dec-2020"):
            pred_AQI = AQI.Forecast.get(key='2020-12-01')
            pred_PM25=PM25.Forecast.get(key='2020-12-01')
            pred_PM10=PM10.Forecast.get(key='2020-12-01')
            return render_template('predict_poll1.html',month='December 2020',
            pred_AQI='AQI = {}'.format(pred_AQI),
            pred_PM25='PM2.5 = {}'.format(pred_PM25),
            pred_PM10='PM10 = {}'.format(pred_PM10))
        elif(month=="Jan-2021"):
            pred_AQI = AQI.Forecast.get(key='2021-01-01')
            pred_PM25=PM25.Forecast.get(key='2021-01-01')
            pred_PM10=PM10.Forecast.get(key='2021-01-01')
            return render_template('predict_poll1.html',month='January 2021',
            pred_AQI='AQI = {}'.format(pred_AQI),
            pred_PM25='PM2.5 = {}'.format(pred_PM25),
            pred_PM10='PM10 = {}'.format(pred_PM10))
        elif(month=="Feb-2021"):
            pred_AQI = AQI.Forecast.get(key='2021-02-01')
            pred_PM25=PM25.Forecast.get(key='2021-02-01')
            pred_PM10=PM10.Forecast.get(key='2021-02-01')
            return render_template('predict_poll1.html',month='February 2021',
            pred_AQI='AQI = {}'.format(pred_AQI),
            pred_PM25='PM2.5 = {}'.format(pred_PM25),
            pred_PM10='PM10 = {}'.format(pred_PM10))
        elif(month=="March-2021"):
            pred_AQI = AQI.Forecast.get(key='2021-03-01')
            pred_PM25=PM25.Forecast.get(key='2021-03-01')
            pred_PM10=PM10.Forecast.get(key='2021-03-01')
            return render_template('predict_poll1.html',month='March 2021',
            pred_AQI='AQI = {}'.format(pred_AQI),
            pred_PM25='PM2.5 = {}'.format(pred_PM25),
            pred_PM10='PM10 = {}'.format(pred_PM10))
        elif(month=="April-2021"):
            pred_AQI = AQI.Forecast.get(key='2021-04-01')
            pred_PM25=PM25.Forecast.get(key='2021-04-01')
            pred_PM10=PM10.Forecast.get(key='2021-04-01')
            return render_template('predict_poll1.html',month='April 2021',
            pred_AQI='AQI = {}'.format(pred_AQI),
            pred_PM25='PM2.5 = {}'.format(pred_PM25),
            pred_PM10='PM10 = {}'.format(pred_PM10))
            
        elif(month=="May-2021"):
            pred_AQI = AQI.Forecast.get(key='2021-05-01')
            pred_PM25=PM25.Forecast.get(key='2021-05-01')
            pred_PM10=PM10.Forecast.get(key='2021-05-01')
            return render_template('predict_poll1.html',month='May 2021',
            pred_AQI='AQI = {}'.format(pred_AQI),
            pred_PM25='PM2.5 = {}'.format(pred_PM25),
            pred_PM10='PM10 = {}'.format(pred_PM10))
        elif(month=="June-2021"):
            pred_AQI = AQI.Forecast.get(key='2021-06-01')
            pred_PM25=PM25.Forecast.get(key='2021-06-01')
            pred_PM10=PM10.Forecast.get(key='2021-06-01')
            return render_template('predict_poll1.html',month='June 2021',
            pred_AQI='AQI = {}'.format(pred_AQI),
            pred_PM25='PM2.5 = {}'.format(pred_PM25),
            pred_PM10='PM10 = {}'.format(pred_PM10))
        elif(month=="July-2021"):
            pred_AQI = AQI.Forecast.get(key='2021-07-01')
            pred_PM25=PM25.Forecast.get(key='2021-07-01')
            pred_PM10=PM10.Forecast.get(key='2021-07-01')
            return render_template('predict_poll1.html',month='July 2021',
            pred_AQI='AQI = {}'.format(pred_AQI),
            pred_PM25='PM2.5 = {}'.format(pred_PM25),
            pred_PM10='PM10 = {}'.format(pred_PM10))
        
        
        return render_template('pollution_main.html')
@app.route('/ozone.html' ,methods=['GET', 'POST'])
def ozone():
    return render_template('ozone.html')
@app.route('/gallery.html' ,methods=['GET', 'POST'])
def gallery():
    return render_template('gallery.html')


@app.route('/faq.html' ,methods=['GET', 'POST'])
def faq():
    return render_template('faq.html')
@app.route('/about.html' ,methods=['GET', 'POST'])
def about():
    return render_template('about.html')

@app.route('/temperature.html' ,methods=['GET', 'POST'])
def temperature():
    return render_template('temperature.html')
@app.route('/humidity.html' ,methods=['GET', 'POST'])
def humidity():
    return render_template('humidity.html')
@app.route('/heatindex.html' ,methods=['GET', 'POST'])
def heatindex():
    return render_template('heatindex.html')
@app.route('/dewpoint.html' ,methods=['GET', 'POST'])
def dewpoint():
    return render_template('dewpoint.html')
@app.route('/AQIf.html' ,methods=['GET', 'POST'])
def AQIf():
    return render_template('AQIf.html')
@app.route('/PM25f.html' ,methods=['GET', 'POST'])
def PM25f():
    return render_template('PM25f.html')
@app.route('/PM10f.html' ,methods=['GET', 'POST'])
def PM10f():
    return render_template('PM10f.html')

@app.route('/contact_us', methods=['GET', 'POST'])
def contact_us():
    msg = ''
    if request.method == "POST": 
        details = request.form
        name = details['name']
        email = details['email']
        subject = details['subject']
        message = details['message']
        cursor = mysql.connection.cursor()
        cursor.execute("INSERT INTO contacts(name, email,subject,message) VALUES (%s, %s ,%s, %s)", (name, email, subject, message))
        mysql.connection.commit()
        cursor.close()
        msg = 'Submitted successfully !'
    return render_template('contact_us.html', msg = msg)





if __name__ == "__main__":
    app.run(debug=True)


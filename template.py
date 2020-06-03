std_template = '''
from flask import Flask, render_template
from flask_ask import Ask, statement, question, session

app = Flask(__name__)
ask = Ask(app,"/")

@ask.launch
def launched():
	welcome = "<WELCOME_MESSAGE>. Would you like to try this skill out?"
	return question(welcome)

@ask.intent('YesIntent')
def logic_builder():
	"""
	Enter logic of you choice that you want
	your skill to do
	"""

@ask.intent('NoIntent')
def NoIntent():
	"""
	If user says no, then the skill will quit
	and respond to the user saying bye
	"""
	text = "Okay no problem! Will hopefully serve you soon! Bye!"
	return statement(text)

@ask.intent('AMAZON.StopIntent')
def stop():
    return statement("Alright! Bye.")


@ask.intent('AMAZON.CancelIntent')
def cancel():
	return statement("Alright! Bye.")

@ask.intent('AMAZON.HelpIntent')
def help():
    return statement("<HELP_MESSAGE>")

@ask.session_ended
def session_ended():
    return "{}", 200


if __name__ == '__main__':
	app.run(debug=True)
'''


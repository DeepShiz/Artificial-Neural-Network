from wtforms import Form, TextField, TextAreaField, validators, StringField, SubmitField

class ReusableForm(Form):
	name = TextField('Full Name *', validators=[validators.DataRequired()])
	email = TextField('Email ID *', validators=[validators.DataRequired()])
	ph_num = TextField('Phone Number *', validators=[validators.DataRequired()])
	mentor_field = TextField('Which area do you want to be mentored on? *', validators=[validators.DataRequired()])
	expect = TextField('What do you expect from your mentor? *', validators=[validators.DataRequired()])

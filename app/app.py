from flask import Flask, render_template, request
from translate import translate_eng_to_nepali

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/translate', methods=['POST'])
def translate():
    if request.method == 'POST':
        text_to_translate = request.form['text_to_translate']
        # target_language = request.form['target_language']

        translated_text = translate_eng_to_nepali(text_to_translate)
        # translated_text = translation.text

        return render_template('index.html', translated_text=translated_text, original_text=text_to_translate)
 
if __name__ == '__main__':
    app.run(debug=True)

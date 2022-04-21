from flask import Flask, render_template, redirect, url_for,request
from flask import make_response,jsonify
from flask_cors import CORS, cross_origin
import re

# calling the summarizer file 
import summarize 

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

@app.route("/")
@cross_origin()
def home():
    return "hi"


@app.route('/login', methods=['POST'])
@cross_origin()
def login():
   print("login")
   if request.method == 'POST':
        # print(request.form.to_dict())
        keyList = list(request.form.to_dict().keys())
        
        corpus = "".join(keyList[0])
        # corpus = corpus.split("\r\n")
        
        # print(corpus[2:-2:1])
        # re.sub('[","]','',corpus)
        corpus = corpus[2:-2:1].split("\\r\\n")
        corpus = "".join(corpus)
        corpus = corpus.split('''","''')
        corpus = "".join(corpus)
        # print(corpus)

        summary = summarize.summary(corpus)
        print("------------------")
        print(summary)
        # for i in range(len(corpus)):
        #     # corpus[i] = corpus[i].split(":")[1]
        #     print(i,corpus[i])
        #     print()
        
        # completeText = []
        # for k,v in splitText:
        #     completeText.append(v)


        return str(summary)

if __name__ == "__main__":
    app.run(debug = True)
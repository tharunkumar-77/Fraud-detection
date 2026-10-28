from flask import Flask,jsonify

app=Flask(__name__)

@app.route("/")
def response():
    return "Home"

if __name__ =="__main__":
    app.run(debug=True,port=5001)
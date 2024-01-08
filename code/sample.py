from flask import Flask, request, jsonify, render_template

app = Flask(__name__,static_url_path='/static')


@app.route('/')
def index():
    return render_template('Home.html')


if __name__ == '__main__':
    app.run(debug = True)
from flask import Flask, request, jsonify

app = Flask(__name__)

books = [
    {
        'name': 'Myth of Sisyphus',
        'author':"Albert Camus"
    },
    {
        'name': 'The Trouble with Being Born',
        'author': 'E. M. Cioran'
    }
]

# @app.route('/')
# def home():
#     return "<h1>Hello Excelize</h1>"

@app.route('/', methods=['POST', 'GET'])
def result():
    if request.method == 'GET':
        return jsonify(books)
    elif request.method == 'POST':
        name = request.form['name']
        author = request.form['author']
        books.append({'name': name, 'author': author})
        return jsonify(books)
    
    else:
        return 'Nothing', 404


if __name__=="__main__":
    app.run(debug=True)

from flask import Flask
from PriceCheck import check_price

app = Flask(__name__)

@app.route('/search')
def Response():
    return check_price()

if __name__ == '__main__':
    app.run(host="0.0.0.0",port=5000,debug=True)

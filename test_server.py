"""
Simple test to check if Flask server can start
"""
from flask import Flask

app = Flask(__name__)

@app.route("/")
def test():
    return "Server is running! Flask is working correctly."

if __name__ == "__main__":
    print("Testing Flask server...")
    print("If you see this message and can access http://localhost:5000, Flask is working!")
    app.run(debug=True, host='127.0.0.1', port=5000)



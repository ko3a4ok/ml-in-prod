from flask import Flask
import os
app = Flask(__name__)


@app.get('/')
def main():
  return {
      'predictions': ['Ololo']
  }


if __name__ == '__main__':
  print('Start the app')
  app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))

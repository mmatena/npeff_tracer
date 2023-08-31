"""Server for the NPEFF trace viewer."""

import collections
import os

from flask import Flask, jsonify, render_template

###############################################################################

app = Flask(__name__)

###############################################################################


if __name__ == '__main__':
    app.run(debug=True)

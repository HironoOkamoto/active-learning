import json
import os
from glob import glob
from datetime import date
from os.path import basename, exists, join
from os import path, walk

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report

from flask import Flask, flash, redirect, render_template, request, url_for
from flask_bootstrap import Bootstrap

extra_dirs = ['templates', ]
extra_files = extra_dirs[:]
for extra_dir in extra_dirs:
    for dirname, dirs, files in walk(extra_dir):
        for filename in files:
            filename = path.join(dirname, filename)
            if path.isfile(filename):
                extra_files.append(filename)


plt.style.use("ggplot")
plt.switch_backend('agg')

app = Flask(__name__)
bootstrap = Bootstrap(app)
app.config["SECRET_KEY"] = "sample1203"


@app.context_processor
def override_url_for():
    return dict(url_for=dated_url_for)


def dated_url_for(endpoint, **values):
    if endpoint == 'static':
        filename = values.get('filename', None)
        if filename:
            file_path = os.path.join(app.root_path,
                                     endpoint, filename)
            values['q'] = int(os.stat(file_path).st_mtime)
    return url_for(endpoint, **values)


@app.route('/')
def index():
    expid_paths = sorted(glob("./static/*"))
    cfgs = []
    for p in expid_paths:
        cfgs.append(json.load(open(join(p, 'cfg.json'), "r")))

    expids = [basename(f) for f in expid_paths]
    return render_template("index.html", expids=expids, cfgs=cfgs)


@app.route('/result/<string:expid>')
def result(expid):
    cfg_json = f'./static/{expid}/cfg.json'
    cfg = json.load(open(cfg_json, "r"))
    return render_template("result.html",
                           expid=expid, cfg=cfg)


if __name__ == "__main__":
    app.run(debug=True, port=8081, extra_files=extra_files)

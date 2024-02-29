import os

import open3d as o3d

from flask import (
    Blueprint, flash, g, redirect, render_template, request, url_for, current_app
)
from website.db import get_db
from werkzeug.exceptions import abort
from website.auth import login_required
from werkzeug.utils import secure_filename

from . import utils

ALLOWED_EXTENSIONS = {'txt', 'npy', 'pcd', 'pth', 'ply'}

bp = Blueprint('blog', __name__)


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@bp.route('/', methods=('GET', 'POST'))
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('File not detected!')
            return redirect(request.url)

        file = request.files['file']
        if file.filename == '':
            flash('Please upload a file!')
            return redirect(request.url)

        if file and allowed_file(file.filename):
            app = current_app
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            return redirect(url_for('blog.process', filename=filename))

    return render_template('blog/main.html')


@bp.route('/process', methods=('GET', 'POST'))
@login_required
def process():
    if request.method == 'POST':
        app = current_app
        filename = request.args.get('filename')
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        if os.path.isfile(filepath):
            if request.form['dataset'] == 'scannetv2' and request.form['model'] == 'spformer':
                preprocessed_file = utils.preprocess(
                    filepath, 'spformer', 'scannetv2')
                processed_ply = utils.process(
                    preprocessed_file, 'spformer', 'scannetv2')
                return redirect(url_for('blog.visualize', output=True, filename=processed_ply.split('/')[-1]))
            elif request.form['dataset'] == 's3dis' and request.form['model'] == 'pnpp':
                flash("Not implemented yet!")
            else:
                flash("Not implemented yet!")
        else:
            flash("File does not exist!")

    return render_template('blog/process.html', filename=request.args.get('filename'))


@bp.route('/visualize', methods=('GET',))
@login_required
def visualize():
    app = current_app
    if request.args.get('output'):
        filepath = os.path.join('/', *app.config['UPLOAD_FOLDER'].split('/')[:-1], 'output', request.args.get('filename'), 'output.ply')
    else:
        # TODO: File format conversion
        filepath = os.path.join('/', app.config['UPLOAD_FOLDER'], request.args.get('filename'))
    
    point_cloud = o3d.io.read_point_cloud(filepath)
    aabb = point_cloud.get_axis_aligned_bounding_box()
    
    o3d.visualization.draw_geometries([point_cloud, aabb])
    
    return redirect(url_for('blog.index'))

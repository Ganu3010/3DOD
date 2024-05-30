import os
import json

import numpy as np
import open3d as o3d

from flask import (
    Blueprint, flash, g, redirect, render_template, request, url_for, current_app, send_file, session
)
from website.db import get_db
from werkzeug.exceptions import abort
from website.auth import login_required
from werkzeug.utils import secure_filename


from . import utils

ALLOWED_EXTENSIONS = {'txt', 'npy', 'pcd', 'ply'}

bp = Blueprint('blog', __name__)


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@bp.route('/', methods=('GET', 'POST'))
def index():
    '''
    Homepage of the website. File upload happens here.
    File upload only possible if user is logged in.
    Redirects to /process?filename=name after successful upload.
    '''

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
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            try:
                if filename.endswith('npy'):
                    filepath = utils.to_pcd(filepath)

                if filename.endswith('txt'):
                    file_format = 'xyz'
                else:
                    file_format = 'auto'

                tmp = o3d.io.read_point_cloud(filepath, format=file_format)
                if len(tmp.points) < 1:
                    flash("File is empty!")
                    raise KeyError
            except Exception as e:
                print(e)
                flash("Upload Valid File!")
                if os.path.exists(filepath):
                    os.remove(filepath)
                return render_template('blog/main.html')

            return redirect(url_for('blog.process', filename=filename))
        else:
            flash('Upload Valid File!')

    return render_template('blog/main.html')


@bp.route('/process', methods=('GET', 'POST'))
@login_required
def process():
    '''
    This page decides the model and dataset to use.
    Requires the filename to be passed from the homepage.
    Redirects to /visualize?filename=name after successful processing.
    '''
    

    app = current_app
    filename = request.args.get('filename')
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    
    if not os.path.exists(filepath):
        flash('File not found!')
        return redirect(url_for('index'))
    
    if request.method == 'POST':
        db = get_db()

        if os.path.isfile(filepath):
            if request.form['dataset'] == 'scannetv2' and request.form['model'] == 'spformer':

                try:
                    preprocessed_file = os.path.split(
                        utils.preprocess(filepath, 'spformer', 'scannetv2'))[-1]
                    processed_ply = os.path.split(utils.process(
                        preprocessed_file, 'spformer', 'scannetv2'))[-1]
                except Exception as e:
                    flash(f"ERROR! {e}")
                    return render_template('blog/process.html', filename=request.args.get('filename'))

                try:
                    db.execute(
                        """
                        INSERT INTO experiments (
                            input_file_path,
                            preprocessed_file_path,
                            output_file_path,
                            dataset,
                            model
                        ) VALUES (?, ?, ?, ?, ?)
                        ON CONFLICT(input_file_path, dataset, model) DO UPDATE SET
                            dataset=EXCLUDED.dataset,
                            model=EXCLUDED.model,
                            created_at=datetime('now', 'localtime')
                        """, (
                            os.path.split(filepath)[-1],
                            preprocessed_file,
                            processed_ply,
                            request.form['dataset'],
                            request.form['model']
                        ),
                    )
                    db.commit()
                except Exception as e:
                    print(e)
                    # flash(f"ERROR! {e}")

                return redirect(url_for('blog.visualize', output=True, filename=processed_ply))

            else:
                flash("Not implemented yet!")

        else:
            flash("File does not exist!")

    return render_template('blog/process.html', filename=request.args.get('filename'))


@bp.route('/visualize', methods=('GET',))
@login_required
def visualize():
    '''
    Page for visualization of point cloud.
    Requires the filename to be passed from /process.
    Redirects to homepage after closing open3d window.
    '''

    app = current_app
    vis = app.config['VISUALIZER']
    vis.create_window()
    filename = request.args.get('filename')

    if request.args.get('output'):
        op_path = os.path.join(
            app.config['UPLOAD_FOLDER'].replace('input', 'output'), filename)
        filepath = os.path.join(op_path, 'output.ply')
        filename += '.ply'

        corners = []
        bbs, labels = utils.get_bounding_boxes(op_path)

        obj_id = 0
        json_output = {}

        for in_points, label in zip(bbs, labels):
            if len(in_points) > 0:
                bb = o3d.geometry.AxisAlignedBoundingBox().create_from_points(
                    o3d.utility.Vector3dVector(in_points))
                bb_corners = np.asarray(bb.get_box_points())
                corners.append(bb_corners)
                bb.color = [1, 0, 0]
                vis.add_geometry(bb)

                json_output[obj_id] = {"class": utils.CLASS_MAPPING[label],
                                       "corners": bb_corners.tolist()}
                obj_id += 1

                label = o3d.t.geometry.TriangleMesh.create_text(
                    utils.CLASS_MAPPING[label], depth=5).to_legacy()
                label.paint_uniform_color((0.8, 1, 0.1))
                label.transform([[0.012, 0, 0, bb_corners[-3][0]],
                                 [0, 0.012, 0, bb_corners[-3][1]],
                                 [0, 0, 0.012, bb_corners[-3][2]],
                                 [0, 0, 0, 0.95]])
                vis.add_geometry(label)

        with open(os.path.join(op_path, "bb_" + filename.split(".")[0] + ".json"), "w") as outfile:
            outfile.write(json.dumps(json_output, indent=4))

    else:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

        if filepath.endswith('.pth'):
            # TODO: Add support for .pth visualization
            flash('.pth not supported for visualization yet!')
        elif filepath.endswith(('.txt', '.npy')):
            try:
                filepath = utils.to_pcd(filepath)
            except Exception as e:
                vis.destroy_window()
                flash(f"ERROR! {e}! Please make sure file is valid!")
                return redirect(url_for('blog.index'))

    try:
        point_cloud = o3d.io.read_point_cloud(filepath)
        aabb = point_cloud.get_axis_aligned_bounding_box()
        vis.add_geometry(point_cloud)
        vis.add_geometry(aabb)
        vis.run()
    except Exception as e:
        flash(f"ERROR! {e}")

    vis.clear_geometries()
    vis.destroy_window()

    return redirect(url_for('blog.process', filename=filename))


@bp.route('/experiments', methods=('GET',))
@login_required
def experiments():
    db = get_db()
    cursor = db.cursor()

    cursor.execute('SELECT * FROM experiments')
    rows = cursor.fetchall()

    return render_template('blog/list.html', items=rows)


@bp.route('/export', methods=('GET',))
@login_required
def export():
    '''
    Export predicted bounding boxes to txt file
    '''
    app = current_app

    filename = request.args.get('filename').split('.')[0]
    output_file = os.path.join(app.config['UPLOAD_FOLDER'].replace(
        'input', 'output'), filename, 'bb_' + filename + '.json')

    try:
        return send_file(output_file, as_attachment=True)
    except Exception as e:
        flash(str(e))
        flash("Please make sure to click predict before exporting bounding boxes!")
        return render_template('blog/process.html', filename=request.args.get('filename'))

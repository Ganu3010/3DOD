import os

import open3d as o3d

from flask import (
    Blueprint, flash, g, redirect, render_template, request, url_for, current_app,send_file
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
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            return redirect(url_for('blog.process', filename=filename))

    return render_template('blog/main.html')


@bp.route('/process', methods=('GET', 'POST'))
@login_required
def process():
    '''
    This page decides the model and dataset to use.
    Requires the filename to be passed from the homepage.
    Redirects to /visualize?filename=name after successful processing.
    '''
    
    if request.method == 'POST':
        
        db=get_db()
        filename = request.args.get('filename')
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        if os.path.isfile(filepath):
            if request.form['dataset'] == 'scannetv2' and request.form['model'] == 'spformer':
                
                # preprocessed_file = utils.preprocess(filepath, 'spformer', 'scannetv2')
                # processed_ply = utils.process(preprocessed_file, 'spformer', 'scannetv2')

                # saving in table-> experiments
                #TO-DO change preprocessed and processed filepath while saving in db
                db.execute(
                    "INSERT INTO experiments (input_file_path,preprocessed_file_path,output_file_path,dataset,model) VALUES (?, ?, ?, ?, ?)",
                    (filepath,filepath,filepath,request.form['dataset'],request.form['model']),
                )
                db.commit()

                # return redirect(url_for('blog.visualize', output=True, filename=processed_ply.split('/')[-1]))
                
        
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
    if request.args.get('output'):
        filepath = os.path.join('/', *app.config['UPLOAD_FOLDER'].split('/')[:-1], 'output', request.args.get('filename'), 'output.ply')
    else:
        filepath = os.path.join('/', app.config['UPLOAD_FOLDER'], request.args.get('filename'))
    
    if filepath.endswith('.pth'):
        # TODO: Add support for .pth visualization
        flash('.pth not supported for visualization yet!')
    elif filepath.endswith(('.txt', '.npy')):
        filepath = utils.to_pcd(filepath)
    
    point_cloud = o3d.io.read_point_cloud(filepath)
    aabb = point_cloud.get_axis_aligned_bounding_box()
    
    o3d.visualization.draw_geometries([point_cloud, aabb])
    
    return redirect(url_for('blog.index'))

    

@bp.route('/list', methods=('GET',))
@login_required
def list():
    app = current_app
    db=get_db()
    cursor = db.cursor()

    cursor.execute('SELECT * FROM experiments')
    rows = cursor.fetchall()

    return render_template('blog/list.html', items=rows)



@bp.route('/export', methods=('GET',))
@login_required
def export():
    
    app = current_app
    my_array = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10,11,12,13,14]
    
    filepath=os.path.join(app.config['UPLOAD_FOLDER'])
    output_path = filepath.replace('input', 'bounding_boxes')
    output_file=os.path.join(output_path,'bb.txt')

    f = open(output_file, "w")
    for item in my_array:
        f.write(str(item) + '\n')
       
    f.close()

    return send_file(output_file, as_attachment=True)

                
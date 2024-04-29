from utils import get_bounding_boxes
from flask import jsonify

if __name__=='__main__':
    data = get_bounding_boxes('static/output/scene0000_00_vh_clean_2')
    CLASS_MAPPING = {
    3: 'cabinet', 4: 'bed', 5: 'chair',
    6: 'sofa', 7: 'table', 8: 'door',
    9: 'window', 10: 'bookshelf', 11: 'picture',
    12: 'counter', 14: 'desk', 16: 'curtain',
    24: 'refrigerator', 28: 'shower curtain', 33: 'toilet',
    34: 'sink', 36: 'bathtub', 39: 'otherfurniture'
    }
    print(data)
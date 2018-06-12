from flask import Flask,request,render_template,redirect,url_for
from scipy import misc
from sklearn.externals import joblib
from werkzeug.utils import secure_filename
import os
import tensorflow as tf
import numpy as np
from sklearn import metrics
import TextClassifierPrepare as prepare
import argparse

app = Flask(__name__)

def load_graph(frozen_graph_pb):
    with tf.gfile.GFile(frozen_graph_pb,'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def,name='prefix')
    return graph


@app.route("/")
def index():
    return '''
    <!doctype html>
    <html>
        <body>
            <form action='/upload_text' method='post' enctype='multipart/form-data'>
                <input type='file' name='file_text'>
                <input type='submit' value='Start Text Classifier'>
            </form>
        </body>
    </html>
    '''

@app.route('/upload_text',methods=['POST'])
def upload_text():
    if request.method == 'POST':
        result = test_from_pb()
    return "failed"

base_dir = ''
vocab_dir_default = 'zyjtextclassifier/cnews.vocab.txt'  # 'models/cnews.vocab.txt'


def test_from_pb():

    categories, cat_to_id = prepare.read_category()

    vocab_dir = os.path.join(base_dir, vocab_dir_default)


    words, word_to_id = prepare.read_vocab(vocab_dir)
    print("test from pb...")

    test_dir = request.files['file_text']

    up_path = os.path.join(base_dir,'',secure_filename(test_dir.filename))
    test_dir.save(up_path)

    test_dir = os.path.join(base_dir, test_dir.filename)

    x_test, y_test = prepare.process_file(test_dir, word_to_id, cat_to_id, 600)

    session = tf.Session()
    session.run(tf.global_variables_initializer())

    parser = argparse.ArgumentParser()
    parser.add_argument("--frozen_model_filename", default='zyjtextclassifier/frozen_model_argmax.pb',
                        type=str, help='Frozen model file to import')
    args = parser.parse_args()
    graph = load_graph(args.frozen_model_filename)
    print(graph)

    arg_max = graph.get_tensor_by_name("prefix/score/ArgMax:0")
    session = tf.Session(graph=graph)
    x = graph.get_tensor_by_name("prefix/input_x:0")
    prob = graph.get_tensor_by_name("prefix/keep_prob:0")

    out = session.run(arg_max, feed_dict={x:x_test,prob:1})
    y_test_cls = np.argmax(y_test, 1)

    print(metrics.classification_report(y_test_cls, out, target_names=categories))

    return out


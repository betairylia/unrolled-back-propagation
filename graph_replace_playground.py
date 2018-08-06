import tensorflow as tf
from tensorflow.contrib import graph_editor
from collections import OrderedDict

def f():
    """適当に計算グラフを作成して入力テンソルと出力テンソルを返す"""
    with tf.variable_scope('f'):
        x = tf.placeholder(tf.float32, (None, 64), name='x')
        h = x
        h = tf.layers.dense(h, 100, name='dense_1')
        h = tf.layers.dense(h, 100, name='dense_2')
        y = h
        return x, y


with tf.Graph().as_default():
    # 計算グラフを作成
    x, y = f()

    # x と互換性のあるテンソル
    new_x = tf.placeholder(x.dtype, x.get_shape(), name='x_copy')

    t = OrderedDict()
    t2 = OrderedDict()
    t[x] = new_x
    t2[x] = y

    print(t)

    # x の代わりに x_copy を使って、x から y への計算をやりなおす
    t = graph_editor.graph_replace(
        t2,                  # 複製元の出力
        t,         # {複製元の入力: 新しい入力}
    )

    print(x)
    print(y)
    print(new_x)
    print(t)

    # graph_replace will keep the shape of target_ts.
    # replacement_ts should be a dict, contains replacement_ts[origin(key)] = replacement(value)
    # If target_ts is a tensor, then graph_replace will return a tensor
    # But if target_ts is a dict e.g. in unrolled target_ts[key] = y,
    # Then the return will keep the key and the structure of target_ts, returning:
    # target_ts[key] = replaced_y, with replaced_y calculated from replacement_ts.
    # holy crap

    # TensorBoardで確認
    tf.summary.FileWriter('logs/', graph=tf.get_default_graph()).close()
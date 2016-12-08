import os
import tensorflow as tf
import time

from tensorflow.contrib import slim
from tensorflow.python.ops import ctc_ops
from util.importers.ldc93s1 import read_data_sets
from util.text import sparse_tensor_value_to_texts

learning_rate = 0.001
beta1 = 0.9
beta2 = 0.999
epsilon = 1e-8
training_iters = 100
train_batch_size = 1
dev_batch_size = 1
test_batch_size = 1
dropout_rate = 0.00
relu_clip = 20
session_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)

n_input = 26
n_context = 9
n_hidden_1 = n_input + 2*n_input*n_context
n_hidden_2 = n_input + 2*n_input*n_context
n_hidden_5 = n_input + 2*n_input*n_context
n_cell_dim = n_input + 2*n_input*n_context
n_hidden_3 = 2 * n_cell_dim
n_character = 29
n_hidden_6 = n_character

def model(batch_x, seq_length, dropout):
    def clipped_relu(x):
        return tf.minimum(tf.nn.relu(x), relu_clip)
    
    with slim.arg_scope([slim.variable], device="/cpu:0"):
        with slim.arg_scope([slim.fully_connected], activation_fn=clipped_relu):
            with slim.arg_scope([slim.dropout], keep_prob=(1.0 - dropout)):
                fc_1 = slim.dropout(slim.fully_connected(batch_x, n_hidden_1))
                fc_2 = slim.dropout(slim.fully_connected(fc_1, n_hidden_2))
                fc_3 = slim.dropout(slim.fully_connected(fc_2, n_hidden_3))
                
                lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(n_cell_dim, forget_bias=1.0, state_is_tuple=True)
                
                rnn_4, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw=lstm_cell,
                                                           cell_bw=lstm_cell,
                                                           inputs=fc_3,
                                                           dtype=tf.float32,
                                                           sequence_length=seq_length)
                rnn_4 = tf.concat(2, rnn_4)
                
                fc_5 = slim.dropout(slim.fully_connected(rnn_4, n_hidden_5))
                logits = slim.fully_connected(fc_5, n_hidden_6)
                
                # Reshape final layer to be time major as CTC expects                
                logits = tf.transpose(logits, [1, 0, 2])
                
                return logits

datasets = read_data_sets("./data/ldc93s1",
                          train_batch_size,
                          dev_batch_size,
                          test_batch_size,
                          n_input,
                          n_context)

audio, audio_lengths, labels = datasets.train.next_batch()

logits = model(audio, audio_lengths, dropout_rate)
loss = ctc_ops.ctc_loss(logits, labels, audio_lengths)
avg_loss = tf.reduce_mean(loss)

decoded, _ = ctc_ops.ctc_beam_search_decoder(logits, audio_lengths, merge_repeated=False)

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate,
                                   beta1=beta1,
                                   beta2=beta2,
                                   epsilon=epsilon)
optimize_op = optimizer.minimize(avg_loss)

session = tf.Session(config=session_config)
with session.as_default():
    tf.initialize_all_variables().run()
    tf.train.start_queue_runners()
    datasets.start_queue_threads(session)

    for epoch in range(training_iters):
        epoch_duration = 0.0
        total_cost = 0.0
        for batch in range(datasets.train.total_batches):
            batch_start_time = time.time()
            cost, _, decoded_str = session.run([loss, optimize_op, decoded])
            batch_duration = time.time() - batch_start_time

            epoch_duration += batch_duration
            total_cost += cost

        avg_cost = total_cost/datasets.train.total_batches
        print("epoch {}, time {}, avg cost {}, decoded {}".format(epoch, epoch_duration, avg_cost, sparse_tensor_value_to_texts(decoded_str[0])))

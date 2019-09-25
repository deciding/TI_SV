# GENERALIZED END-TO-END LOSS FOR SPEAKER VERIFICATION
# https://arxiv.org/abs/1710.10467

import tensorflow as tf
#import argparse
#from tensorflow.python.layers import core as layers_core
import utils
import numpy as np
import re


# 되도록이면 GE2E 클래스를 test/infer하는 데에도 쓸 수 있도록 바꿀 것

class GE2E():
    def __init__(self, hparams):

        self.hparams = hparams
        self.batch_size = self.hparams.num_utt_per_batch * self.hparams.num_spk_per_batch

    def set_up_model(self):
        ge2e_graph = tf.Graph()
        with ge2e_graph.as_default():
            if self.hparams.mode == "train":
                # Input Batch of [N*M(batch_size), total_frames, 40(spectrogram_channel)]
                # Target Batch of [N*M(batch_size)]
                self.input_batch = tf.placeholder(dtype=tf.float32, shape=[None, None, self.hparams.spectrogram_scale], name="input_batch")
                self.target_batch = tf.placeholder(dtype=tf.int32, shape=[None], name="target_batch")
                self._create_embedding()
                self._cal_loss()
                self._optimize()

            elif self.hparams.mode == "infer":
                self.input_batch = tf.placeholder(dtype=tf.float32, shape=[640, None, self.hparams.spectrogram_scale], name="input_batch")
                if self.hparams.batch_inference:
                    self._create_embedding()
                else:
                    self._create_embedding_single()

            # test 부분 추가

            elif self.hparams.mode == "test":
                pass

            else:
                raise ValueError("mode not supported")

        return ge2e_graph

    def _create_embedding(self):
        #start_gpu=int(self.hparams.gpu[0])
        start_gpu=0
        gpu_num=int(self.hparams.gpu_num)
        tower_inputs=tf.split(self.input_batch, gpu_num)
        gpus = ["/gpu:{}".format(i) for i in range(start_gpu, start_gpu+gpu_num)]
        self.tower_norm_out=[]
        for i in range(gpu_num):
            with tf.device(tf.train.replica_device_setter(ps_tasks=1, ps_device="/cpu:0", worker_device=gpus[i])):
                with tf.variable_scope("lstm_embedding", reuse=tf.AUTO_REUSE):
                    # Create Embedding Using LSTM
                    stacked_lstm = tf.nn.rnn_cell.MultiRNNCell([tf.nn.rnn_cell.LSTMCell(self.hparams.num_lstm_cells, num_proj=self.hparams.dim_lstm_projection) for _ in range(self.hparams.num_lstm_stacks)])
                    # 
                    # Create Initial State
                    #init_state = stacked_lstm.zero_state(self.batch_size, dtype=tf.float32)
                    # Decode Using dynamic_rnn
                    # outputs is a tensor of [batch_size, total_frames, output_size]
                    # output_size is self.hparams.dim_lstm_projection if num_proj in LSTMCell is set
                    # state is a tensor of [batch_size, state_size of the cell]

                    outputs, state = tf.nn.dynamic_rnn(cell=stacked_lstm, inputs=tower_inputs[i], dtype=tf.float32)

                    # L2 Normalize the output of the last layer at the final frame
                    # norm_out is a tensor of [batch_size, output_size], by default, [640, 256(proj_nodes)]
                    norm_out = tf.nn.l2_normalize(outputs[:, -1, :], axis=-1)
                    self.tower_norm_out.append(norm_out)

    def _create_embedding_single(self):
        with tf.variable_scope("lstm_embedding"):
            # Create Embedding Using LSTM
            stacked_lstm = tf.nn.rnn_cell.MultiRNNCell([tf.nn.rnn_cell.LSTMCell(self.hparams.num_lstm_cells, num_proj=self.hparams.dim_lstm_projection) for _ in range(self.hparams.num_lstm_stacks)])
            # 
            # Create Initial State
            #init_state = stacked_lstm.zero_state(self.batch_size, dtype=tf.float32)
            # Decode Using dynamic_rnn
            # outputs is a tensor of [batch_size, total_frames, output_size]
            # output_size is self.hparams.dim_lstm_projection if num_proj in LSTMCell is set
            # state is a tensor of [batch_size, state_size of the cell]

            outputs, state = tf.nn.dynamic_rnn(cell=stacked_lstm, inputs=self.input_batch, dtype=tf.float32)

            # L2 Normalize the output of the last layer at the final frame
            # norm_out is a tensor of [batch_size, output_size], by default, [640, 256(proj_nodes)]
            self.norm_out = tf.nn.l2_normalize(outputs[:, -1, :], axis=-1)

    def _cal_centroid_matrix(self, utt_idx):
        # centroid_idx counts from 0 to 63
        def cal_centroid(centroid_idx):
            # utt_idx counts from 0 to 639
            # spk_id counts from 0 to 63
            spk_id = (utt_idx // self.hparams.num_utt_per_batch)
            utt_idx_in_group = utt_idx % self.hparams.num_utt_per_batch

            all_utts_for_spk = self.norm_out[centroid_idx * self.hparams.num_utt_per_batch : (centroid_idx+1) * self.hparams.num_utt_per_batch, :]
            if centroid_idx == spk_id:
                mask = np.array([False if utt == utt_idx_in_group else True for utt in range(self.hparams.num_utt_per_batch)])
                centroid = tf.reduce_mean(tf.boolean_mask(all_utts_for_spk, mask), 0)
            else:
                centroid = tf.reduce_mean(all_utts_for_spk, 0)

            return centroid

        # [64, 256], the centroid for utt_idx will not count utt_idx
        centroid_mat = tf.convert_to_tensor(tf.map_fn(cal_centroid, tf.range(self.hparams.num_spk_per_batch), dtype=tf.float32))
        return centroid_mat

    def _create_sim_per_utt(self, utt_idx):

        #utt_dvector is a tensor of shape [output_size]
        utt_dvector = self.norm_out[utt_idx, :]
        #centroids is a tensor of shape [num_spk_per_batch, output_size]
        #sim_per_utt is a tensor of shape [num_spk_per_batch]
        centroids = self._cal_centroid_matrix(utt_idx)
        sim_per_utt = utils.tf_scaled_cosine_similarity(utt_dvector, centroids)

        # [64]
        return sim_per_utt

    def _cal_loss(self):
        with tf.variable_scope("loss"):
            # utt_idx // num_utt_per_batch(10) => true idx of sim_mat columns
            # 텐플은 matrix 연산에 최적화되어있어서, 텐플에서 제공하는matrix 연산(c로 연산됨)을 파이썬 for loop으로 하면 대박느려짐
            # sim_mat 구할 때도 그런 문제가 있어서 코드를 전면 수정했고
            # cal_loss 도 수정해야하는데 약간 까다로움 
            # train 에서 옵션을 받아서 loss를 2가지로 받을 수 있도록 (현재는 contrast loss만 구현)

            if self.hparams.loss_type == "softmax":
                # sim_mat has shape of [batch_size, num_spk]
                self.sim_mat = tf.convert_to_tensor(tf.map_fn(self._create_sim_per_utt, tf.range(self.batch_size),dtype=tf.float32))
                self.sim_mat_summary = tf.summary.image("sim_mat", tf.reshape(self.sim_mat,[1, self.batch_size, self.hparams.num_spk_per_batch, 1]))
                self.eval_sim_mat_summary = tf.summary.image("eval_sim_mat", tf.reshape(self.sim_mat,[1, self.batch_size, self.hparams.num_spk_per_batch, 1]))
                self.total_loss = tf.divide(tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.sim_mat, labels=self.target_batch)), self.batch_size)
                self.eval_total_loss = tf.divide(tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.sim_mat, labels=self.target_batch)), self.batch_size)

                self.total_loss_summary = tf.summary.scalar("loss", self.total_loss)
                self.eval_total_loss_summary = tf.summary.scalar("eval_loss", self.eval_total_loss)

            elif self.hparams.loss_type == "contrast":
                pass

            else:
                print("Loss type not supported")

    def _optimize(self):
        with tf.variable_scope("optimize"):
            self.global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')
            learning_rate = tf.train.exponential_decay(self.hparams.learning_rate, self.global_step,
                                       30000000, 0.5, staircase=True)
            optimizer = tf.train.GradientDescentOptimizer(learning_rate)
            grads_and_vars = optimizer.compute_gradients(self.total_loss)

            clipped_grad_and_vars = []
            for grad, var in grads_and_vars:
                if re.search("cos_params", var.name):
                    grad = tf.clip_by_value(grad, -self.hparams.scale_clip, self.hparams.scale_clip)
                elif re.search("projection", var.name):
                    grad = tf.clip_by_value(grad, -self.hparams.lstm_proj_clip, self.hparams.lstm_proj_clip)
                else:
                    grad = tf.clip_by_norm(grad, self.hparams.l2_norm_clip)
                clipped_grad_and_vars.append((grad, var))

            self.optimize = optimizer.apply_gradients(clipped_grad_and_vars, global_step=self.global_step)


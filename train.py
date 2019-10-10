import tensorflow as tf
from feeder import Feeder
import argparse
import utils
from model import GE2E
import re
import os
import queue
import time

def main():

    # Hyperparameters

    parser = argparse.ArgumentParser()

    # in_dir = ~/wav
    parser.add_argument("--in_dir", type=str, required=True, help="input data(pickle) dir")
    parser.add_argument("--ckpt_dir", type=str, required=True, help="checkpoint to save/ start with for train/inference")
    parser.add_argument("--mode", default="train", choices=["train", "test", "infer"], help="setting mode for execution")
    parser.add_argument('--data_types', nargs='+', default=['libri', 'vox1', 'vox2'])

    # Saving Checkpoints, Data... etc
    parser.add_argument("--max_step", type=int, default=500000, help="maximum steps in training")
    parser.add_argument("--checkpoint_freq", type=int, default=100, help="how often save checkpoint")
    parser.add_argument("--eval_freq", type=int, default=1000, help="how often do the evaluation")

    # Data
    parser.add_argument("--segment_length", type=float, default=1.6, help="segment length in seconds")
    parser.add_argument("--spectrogram_scale", type=int, default=40,
                                           help="scale of the input spectrogram")

    # Ininitialization
    parser.add_argument("--init_type", type=str, default="uniform", help="type of initializer")
    parser.add_argument("--init_weight_range", type=float, default=0.1, help="initial weight ranges from -0.1 to 0.1")

    # Optimization
    parser.add_argument("--loss_type", default="softmax", choices=["softmax", "contrast"], help="loss type for optimization")
    parser.add_argument("--optimizer", type=str, default="sgd", help="type of optimizer")
    parser.add_argument("--learning_rate", type=float, default=0.01, help="learning rate")
    parser.add_argument("--l2_norm_clip", type=float, default=3.0, help="L2-norm of gradient is clipped at")

    # Train
    parser.add_argument("--num_spk_per_batch", type=int, default= 64,
                                           help="N speakers of batch size N*M")
    parser.add_argument("--num_utt_per_batch", type=int, default= 10,
                                           help="M utterances of batch size N*M")

    # LSTM
    parser.add_argument("--lstm_proj_clip", type=float, default=0.5, help="Gradient scale for projection node in LSTM")
    parser.add_argument("--num_lstm_stacks", type=int, default=3, help="number of LSTM stacks")
    parser.add_argument("--num_lstm_cells", type=int, default=768, help="number of LSTM cells")
    parser.add_argument("--dim_lstm_projection", type=int, default=256, help="dimension of LSTM projection")

    # Scaled Cosine similarity
    parser.add_argument("--scale_clip", type=float, default=0.01, help="Gradient scale for scale values in scaled cosine similarity")

    # Collect hparams
    args = parser.parse_args()

    # Set up Queue
    global_queue = queue.Queue()
    eval_queue = queue.Queue()
    # Set up Feeder
    libri_feeder = Feeder(args, "libri")
    libri_feeder.set_up_feeder(global_queue)

    #vox1_feeder = Feeder(args, "vox1")
    #vox1_feeder.set_up_feeder(global_queue)

    #vox2_feeder = Feeder(args, "vox2")
    #vox2_feeder.set_up_feeder(global_queue)

    eval_feeder = Feeder(args, "librispeech/libritest")
    eval_feeder.set_up_feeder(eval_queue)

    # Set up Model

    model = GE2E(args)
    graph = model.set_up_model()

    # Training
    with graph.as_default():
        saver = tf.train.Saver()

    with tf.Session(graph=graph, config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        train_writer = tf.summary.FileWriter(args.ckpt_dir, sess.graph)
        ckpt = tf.train.get_checkpoint_state(args.ckpt_dir)
        if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
            print('Restoring Variables from {}'.format(ckpt.model_checkpoint_path))
            saver.restore(sess, ckpt.model_checkpoint_path)
            start_step = sess.run(model.global_step)

        else:
            print('start from 0')
            init_op = tf.global_variables_initializer()
            sess.run(init_op)
            start_step = 1

        start = 0
        from tensorflow.python.client import timeline
        for num_step in range(start_step, args.max_step + 1):
        #for num_step in range(start_step, start_step + 1):
            print("current step: %dth step, time consumed: %f sec" % (num_step, time.time()-start))
            start=time.time()

            start1=time.time()
            batch = global_queue.get()
            print("get queue time: %f" % (time.time()-start1))

            #from tensorflow.python.profiler import model_analyzer, option_builder
            #my_profiler = model_analyzer.Profiler(graph=sess.graph)
            #run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            #run_metadata = tf.RunMetadata()

            start2=time.time()
            #sim_mat_summary, training_loss_summary, training_loss, _ = sess.run([model.sim_mat_summary, model.total_loss_summary, model.total_loss, model.optimize], feed_dict={model.input_batch: batch[0], model.target_batch : batch[1]},  options=run_options, run_metadata=run_metadata)
            import pdb;pdb.set_trace()
            sim_mat_summary, training_loss_summary, training_loss, up = sess.run([model.sim_mat_summary, model.total_loss_summary, model.total_loss, model.optimize], feed_dict={model.input_batch: batch[0], model.target_batch : batch[1]})
            #training_loss, _ = sess.run([model.total_loss, model.optimize], feed_dict={model.input_batch: batch[0], model.target_batch : batch[1]})
            #import pdb;pdb.set_trace()
            print("sess run time: %f" % (time.time()-start2))

            ## Create the Timeline object, and write it to a json
            #tl = timeline.Timeline(run_metadata.step_stats)
            #ctf = tl.generate_chrome_trace_format()
            #with open('timeline.json', 'w') as f:
            #    f.write(ctf)

            #my_profiler.add_step(step=num_step, run_meta=run_metadata)
            #profile_op_builder = option_builder.ProfileOptionBuilder( )
            ## sort by time taken
            #profile_op_builder.select(['micros', 'occurrence'])
            #profile_op_builder.order_by('micros')
            #profile_op_builder.with_max_depth(20) # can be any large number
            #profile_op_builder.with_file_output('profile.log') # can be any large number
            #my_profiler.profile_name_scope(profile_op_builder.build())

            train_writer.add_summary(sim_mat_summary, num_step)
            train_writer.add_summary(training_loss_summary, num_step)
            print("batch loss:" + str(training_loss))

            if num_step % args.checkpoint_freq == 0:
                save_path = saver.save(sess, args.ckpt_dir+"/model.ckpt", global_step=model.global_step)
                print("model saved in file: %s / %d th step" % (save_path, sess.run(model.global_step)))

            #if num_step % args.eval_freq == 0:
            #    batch = eval_queue.get()
            #    eval_summary, eval_loss_summary, eval_loss = sess.run([model.eval_sim_mat_summary, model.eval_total_loss_summary, model.eval_total_loss], feed_dict={model.input_batch: batch[0], model.target_batch : batch[1]})
            #    train_writer.add_summary(eval_sim_mat_summary, num_step)
            #    train_writer.add_summary(eval_loss_summary, num_step)
        print("last step elapsed time: %f" % (time.time()-start))

if __name__ == "__main__":
    main()


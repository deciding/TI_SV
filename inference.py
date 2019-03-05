import tensorflow as tf
from model import GE2E
import numpy as np
import argparse
import utils
from feeder import Feeder
from numpy import dot
from numpy.linalg import norm
import glob

def main():

    # Hyperparameters

    parser = argparse.ArgumentParser()

    # Path

    # wav name formatting: id_clip_uttnum.wav
    parser.add_argument("--in_dir", type=str, required=True, help="input dir")
    parser.add_argument("--in_wav1", type=str, help="input wav1 dir")
    parser.add_argument("--in_wav2", default="temp.wav", type=str, help="input wav2 dir")
    #/home/hdd2tb/ninas96211/dev_wav_set
    parser.add_argument("--mode", default="infer", choices=["train", "test", "infer"], help="setting mode for execution")

    parser.add_argument("--ckpt_file", type=str, required=True, help="checkpoint to start with for inference")

    # Data
    #parser.add_argument("--window_length", type=int, default=160, help="sliding window length(frames)")
    parser.add_argument("--segment_length", type=float, default=1.6, help="segment length in seconds")
    parser.add_argument("--overlap_ratio", type=float, default=0.5, help="overlaping percentage")
    parser.add_argument("--spectrogram_scale", type=int, default=40,
                                           help="scale of the input spectrogram")
    # Enrol
    parser.add_argument("--num_spk_per_batch", type=int, default= 5,
                                           help="N speakers of batch size N*M")
    parser.add_argument("--num_utt_per_batch", type=int, default= 10,
                                           help="M utterances of batch size N*M")

    # LSTM
    parser.add_argument("--num_lstm_stacks", type=int, default=3, help="number of LSTM stacks")
    parser.add_argument("--num_lstm_cells", type=int, default=768, help="number of LSTM cells")
    parser.add_argument("--dim_lstm_projection", type=int, default=256, help="dimension of LSTM projection")

    # Collect hparams
    args = parser.parse_args()

    feeder = Feeder(args)
    feeder.set_up_feeder()

    model = GE2E(args)
    graph = model.set_up_model()

    # Training

    with graph.as_default():
        saver = tf.train.Saver()

    with tf.Session(graph=graph) as sess:
        # restore from checkpoints

        saver.restore(sess, args.ckpt_file)

        get_dvector_of_dir(sess, feeder, model, args)

        #wav1_data, wav2_data, match = feeder.create_infer_batch()

        ## score 

        #wav1_out = sess.run(model.norm_out, feed_dict={model.input_batch:wav1_data})
        #wav2_out = sess.run(model.norm_out, feed_dict={model.input_batch:wav2_data})

        #wav1_dvector = np.mean(wav1_out, axis=0)
        #wav2_dvector = np.mean(wav2_out, axis=0)
        #np.save('a.npy', wav1_dvector)

        ##print(wav1_dvector)
        ##print(wav2_dvector)

        #final_score = dot(wav1_dvector, wav2_dvector)/(norm(wav1_dvector)*norm(wav2_dvector))

        #print("final score:" + str(final_score))
        #print("same? :" + str(match))

def rmse(predictions, targets):
    return np.sqrt(np.mean((predictions-targets)**2))

def get_dvector_of_dir(sess, feeder, model, args):
    in_wavs=glob.glob(args.in_dir + '/*.wav')
    total_vectors=None

    #print(in_wavs)
    for in_wav1 in in_wavs:
        args.in_wav1=in_wav1
        wav1_data, wav2_data, match = feeder.create_infer_batch()
        wav1_out = sess.run(model.norm_out, feed_dict={model.input_batch:wav1_data})
        new_total_vectors=np.concatenate((total_vectors, wav1_out), axis=0) if total_vectors is not None else wav1_out
        if total_vectors is not None:
            dvec1=np.mean(new_total_vectors, axis=0)
            dvec2=np.mean(total_vectors,axis=0)
            cossim=dot(dvec1, dvec2)/(norm(dvec1)*norm(dvec2))
            print("=================================================================================================================" + str(rmse(dvec1,dvec2)) + "   " + str(cossim) + "   " + str(np.size(total_vectors, 0)))
            if cossim > 0.999:
                np.save('a.npy', dvec1)
                return

            #print(np.mean(total_vectors,axis=0))
            #print(np.mean(new_total_vectors,axis=0))

        total_vectors=new_total_vectors
    wav1_dvector = np.mean(total_vectors, axis=0)
    np.save('a.npy', wav1_dvector)

if __name__ == "__main__":
    main()

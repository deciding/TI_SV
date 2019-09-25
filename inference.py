import time
import tensorflow as tf
from model import GE2E
import numpy as np
import argparse
#import utils
from feeder import Feeder
#from numpy import dot
#from numpy.linalg import norm
import glob
import os
from tqdm import tqdm
from queue import Queue
from threading import Thread

q=Queue()

def main():

    # Hyperparameters

    parser = argparse.ArgumentParser()

    # Path

    # wav name formatting: id_clip_uttnum.wav
    parser.add_argument("--in_dir", type=str, required=True, help="input dir")
    parser.add_argument("--out_dir", type=str, required=True, help="out dir")
    parser.add_argument("--batch_inference", action="store_true", help="set whether to use the batch inference")
    parser.add_argument("--dataset", type=str, default="libri", help="out dir")
    parser.add_argument("--in_wav1", type=str, help="input wav1 dir")
    parser.add_argument("--in_wav2", default="temp.wav", type=str, help="input wav2 dir")
    #/home/hdd2tb/ninas96211/dev_wav_set
    parser.add_argument("--mode", default="infer", choices=["train", "test", "infer"], help="setting mode for execution")

    parser.add_argument("--ckpt_file", type=str, default='./xckpt/model.ckpt-58100', help="checkpoint to start with for inference")

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
    parser.add_argument('--gpu', default='0',
                        help='Path to model checkpoint')
    parser.add_argument('--gpu_num', default=4,
                        help='Path to model checkpoint')

    # Collect hparams
    args = parser.parse_args()

    import os
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpu)

    feeder = Feeder(args)
    feeder.set_up_feeder()

    model = GE2E(args)
    graph = model.set_up_model()

    #Training
    with graph.as_default():
        saver = tf.train.Saver()

    #num_gpu=4
    #sess_arr=[]
    #for i in range(num_gpu):
    #    gpu_options = tf.GPUOptions(visible_device_list=str(i))
    #    sess_arr.append(tf.Session(graph=graph, config=tf.ConfigProto(gpu_options=gpu_options)))
    #    saver.restore(sess_arr[i], args.ckpt_file)

    #    t = Thread(target=worker)
    #    t.daemon = True
    #    t.start()

    #    save_dvector_of_dir_parallel(sess_arr, feeder, model, args)

    with tf.Session(graph=graph) as sess:
        # restore from checkpoints

        saver.restore(sess, args.ckpt_file)

        #get_dvector_of_dir(sess, feeder, model, args)

        t = Thread(target=worker)
        t.daemon = True
        t.start()

        #save_dvector_of_dir_parallel(sess, feeder, model, args)
        save_dvector_of_dir_parallel_arr(sess, feeder, model, args)

        #save_dvector_of_dir_libri(sess, feeder, model, args)

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

def save_dvector_of_dir_parallel_arr(sess, feeder, model, args):
    #import pdb;pdb.set_trace()
    if args.in_dir.endswith('.wav'):
        in_wavs=[args.in_dir]
    else:
        in_wavs=glob.glob(args.in_dir + '/*.wav')
    args.in_wav1=in_wavs

    total_len=len(in_wavs)
    cur_len=0

    prev_path = '-'
    start=time.time()
    model_time=0
    for fix_len_mel_path_batch in feeder.infer_batch_generator():
        fix_len_mel_batch=[x[0] for x in fix_len_mel_path_batch]
        path_batch=[x[1] for x in fix_len_mel_path_batch]
        ss=time.time()
        norm_out_batch_arr = sess.run(model.tower_norm_out, feed_dict={model.input_batch:fix_len_mel_batch})
        norm_out_batch = []
        for norm in norm_out_batch_arr:
            norm_out_batch+=list(norm)
        model_time+=(time.time()-ss)
        for output in zip(norm_out_batch, path_batch):
            cur_path=output[1]
            if cur_path != prev_path:
                cur_len += 1
            prev_path=cur_path
            q.put(output)
        print("Progress: %.2f%%" % (cur_len/total_len*100), end='\r')

    q.join()
    print("\nElapsed: %.2f" % (time.time()-start))
    print("\nModel: %.2f" % (model_time))
    print(feeder.get_bad_rate())

def save_dvector_of_dir_parallel(sess, feeder, model, args):
    #import pdb;pdb.set_trace()
    if args.in_dir.endswith('.wav'):
        in_wavs=[args.in_dir]
    else:
        in_wavs=glob.glob(args.in_dir + '/*.wav')
    args.in_wav1=in_wavs

    total_len=len(in_wavs)
    cur_len=0

    prev_path = '-'
    start=time.time()
    model_time=0
    generate_time=0
    post_process_time=0
    gg=time.time()
    for fix_len_mel_path_batch in feeder.infer_batch_generator():
        generate_time+=(time.time()-gg)
        fix_len_mel_batch=[x[0] for x in fix_len_mel_path_batch]
        path_batch=[x[1] for x in fix_len_mel_path_batch]
        ss=time.time()
        norm_out_batch = sess.run(model.norm_out, feed_dict={model.input_batch:fix_len_mel_batch})
        model_time+=(time.time()-ss)
        pp=time.time()
        for output in zip(norm_out_batch, path_batch):
            cur_path=output[1]
            if cur_path != prev_path:
                cur_len += 1
            prev_path=cur_path
            q.put(output)
        post_process_time+=(time.time()-pp)
        print("Progress: %.2f%%" % (cur_len/total_len*100), end='\r')
        gg=time.time()

    q.join()
    print("\nElapsed: %.2f" % (time.time()-start))
    print("Model: %.2f" % (model_time))
    print("Generate: %.2f" % (generate_time))
    print("Post Process: %.2f" % (post_process_time))
    print(feeder.get_bad_rate())

def flush_dvector_buf(dvec_buf, path):
    dvector=np.mean(dvec_buf, axis=0)
    np.save(path, dvector)

def worker():
    buf=[]
    prev_path='-'# '-' means start, '' means end
    while True:
        item=q.get()
        path=item[1]
        if prev_path != '-' and path != prev_path:
            flush_dvector_buf(buf, prev_path)
        elif path != '':
            buf.append(item[0])
        else:
            pass# mark task done for this item directly
        q.task_done()

def save_dvector_of_dir_libri(sess, feeder, model, args):
    if args.in_dir.endswith('.wav'):
        in_wavs=[args.in_dir]
    else:
        in_wavs=glob.glob(args.in_dir + '/*.wav')
    #total_vectors=None

    #print(in_wavs)
    #small_sim_cnt=0
    bad_cnt = 0
    cnt = 0
    for in_wav1 in tqdm(in_wavs):
        args.in_wav1=in_wav1
        filename=os.path.splitext((in_wav1))[0]
        #print(filename)
        #import pdb;pdb.set_trace()
        names=filename.split('/')[-3::2]
        out_suffix=names[0]
        out_dir='%s/%s' % (args.out_dir, out_suffix)
        filename=names[1]
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        wav1_data, wav2_data, match = feeder.create_infer_batch()
        cnt += 1
        if len(wav1_data)==0:
            bad_cnt += 1
            #print(in_wav1)
            continue
        #else:
        #    continue
        wav1_out = sess.run(model.norm_out, feed_dict={model.input_batch:wav1_data})
        wav1_dvector = np.mean(wav1_out, axis=0)

        np.save('%s/%s.npy' % (out_dir, filename), wav1_dvector)
    print(bad_cnt/cnt, bad_cnt, cnt)

def get_dvector_of_dir(sess, feeder, model, args):
    if args.in_dir.endswith('.wav'):
        in_wavs=[args.in_dir]
    else:
        in_wavs=glob.glob(args.in_dir + '/*.wav')
    total_vectors=None

    #print(in_wavs)
    #small_sim_cnt=0
    for in_wav1 in in_wavs:
        args.in_wav1=in_wav1
        wav1_data, wav2_data, match = feeder.create_infer_batch()
        if len(wav1_data)==0:
            continue
        wav1_out = sess.run(model.norm_out, feed_dict={model.input_batch:wav1_data})
        new_total_vectors=np.concatenate((total_vectors, wav1_out), axis=0) if total_vectors is not None else wav1_out

        #if total_vectors is not None:
        #    #dvec1=np.mean(new_total_vectors, axis=0)
        #    #dvec2=np.mean(total_vectors,axis=0)
        #    dvec1=np.mean(total_vectors, axis=0)
        #    dvec2=np.mean(wav1_out,axis=0)
        #    cossim=dot(dvec1, dvec2)/(norm(dvec1)*norm(dvec2))
        #    if cossim < 0.5:
        #        small_sim_cnt+=1
        #        print("=================================================================================================================" + str(rmse(dvec1,dvec2)) + "   " + str(cossim) + "   " + str(np.size(new_total_vectors, 0)) + "   " + str(small_sim_cnt))
        #    #if cossim > 0.999:
        #    #    np.save('a.npy', dvec1)
        #    #    return

        #    #print(np.mean(total_vectors,axis=0))
        #    #print(np.mean(new_total_vectors,axis=0))

        total_vectors=new_total_vectors
    wav1_dvector = np.mean(total_vectors, axis=0)
    np.save('a.npy', wav1_dvector)

if __name__ == "__main__":
    main()

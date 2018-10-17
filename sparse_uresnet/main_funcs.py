from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
import os
import time
import argparse

from sparse_uresnet import io_factory
from sparse_uresnet import trainval
from sparse_uresnet import CSVData

torch.backends.cudnn.benchmark = True

def compute_accuracy(io,idx_v,pred_v):
    start,end=(0,0)
    acc_v = np.zeros(shape=[len(idx_v)],dtype=np.float32)
    for i,idx in enumerate(idx_v):
        voxel = io.voxel()[idx]
        label = io.label()[idx]
        end   = start + len(voxel)
        pred  = pred_v[start:end]
        acc_v[i] = (label == pred).astype(np.int32).sum() / float(len(label))
        start = end
    return acc_v

def store_softmax(io,idx_v,softmax_chunk):
    start,end=(0,0)
    acc_v = np.zeros(shape=[len(idx_v)],dtype=np.float32)
    softmax_v = []
    for i,idx in enumerate(idx_v):
        voxel = io.voxel()[idx]
        label = io.label()[idx]
        end   = start + len(voxel)
        softmax = softmax_chunk[start:end]
        io.store(idx,softmax)
        pred = np.argmax(softmax,axis=1)
        acc_v[i] = (label == pred).astype(np.int32).sum() / float(len(label))
        softmax_v.append(softmax)
        start = end
    return acc_v,softmax_v

def train(flags):
    flags.TRAIN = True
    io = io_factory(flags)
    io.initialize()
    io.start_threads()

    trainer = trainval(flags)
    trainer.initialize()

    csv = CSVData(os.path.join(flags.LOG_DIR,'train_log.csv'))
    
    iteration_to_epoch = float(flags.BATCH_SIZE) / io.num_entries()
    iteration = trainer.initialize()

    loss_v = np.zeros(shape=[flags.REPORT_STEP],dtype=np.float32)
    acc_v  = np.zeros(shape=[flags.REPORT_STEP],dtype=np.float32)

    while iteration < flags.ITERATION:
        
        tstart = time.time()
        report_step  = flags.REPORT_STEP     and ((iteration+1) % flags.REPORT_STEP == 0)
        checkpt_step = flags.CHECKPOINT_STEP and ((iteration+1) % flags.CHECKPOINT_STEP == 0)
            
        voxel,feature,label,idx = io.next()
        loss,pred = trainer.train_step(voxel,feature,label)

        acc = compute_accuracy(io,idx,pred)
        
        tspent = time.time() - tstart
        mem = torch.cuda.memory_allocated()
        csv.record(('iter','titer','ttrain','tsave','mem','loss','acc'),
                   (iteration,tspent,trainer.tspent_train,0.0,mem,loss,acc.mean()))

        loss_v [iteration % flags.REPORT_STEP] = loss
        acc_v  [iteration % flags.REPORT_STEP] = acc.mean()
        
        if report_step:
            epoch = iteration * iteration_to_epoch
            msg = 'Iteration %d (epoch %g) ... Mem %g ... Loss/Acc = %g/%g'
            msg = msg % (iteration,epoch,torch.cuda.memory_allocated(),loss_v.mean(),acc_v.mean())
            print(msg)

        if checkpt_step:
            trainer.save_state(iteration)
            csv.record(['tsave'],[trainer.tspent_save])
            
        csv.write()

        iteration +=1
    print('Done training...')
    csv.close()
    io.finalize()
    
def inference(flags):
    flags.TRAIN = False
    io = io_factory(flags)
    io.initialize()
    io.start_threads()

    trainer = trainval(flags)
    trainer.initialize()

    csv_batch = CSVData(os.path.join(flags.LOG_DIR,'inference_log.csv'))
    csv_event = CSVData(os.path.join(flags.LOG_DIR,'data.csv'))
    
    trainer.initialize()

    acc_v  = np.zeros(shape=[flags.REPORT_STEP],dtype=np.float32)

    event_ctr = 0
    iteration = 0
    while iteration < flags.ITERATION and event_ctr < io.num_entries():

        tstart = time.time()
        report_step  = flags.REPORT_STEP     and ((iteration+1) % flags.REPORT_STEP == 0)
        checkpt_step = flags.CHECKPOINT_STEP and ((iteration+1) % flags.CHECKPOINT_STEP == 0)
            
        voxel,feature,label,idx = io.next()
        softmax = trainer.inference_step(voxel,feature)
        #acc = (np.argmax(softmax,axis=1) == label).astype(np.int32).sum() / float(len(label))
        acc,softmax = store_softmax(io,idx,softmax)

        tspent = time.time() - tstart
        mem = torch.cuda.memory_allocated()
        csv_batch.record(('iteration','titer','tinference','mem','acc'),
                         (iteration,tspent,trainer.tspent_inference,mem,acc_v.mean()))

        acc_v[iteration % flags.REPORT_STEP] = acc.mean()
        
        if report_step:
            msg = 'Iteration %d ... Mem %g ... Acc = %g'
            msg = msg % (iteration,mem,acc_v.mean())
            print(msg)

        csv_batch.write()
        for i,event_index in enumerate(idx):
            csv_event.record(('index','acc'),(event_index,acc[i]))
            csv_event.write()
            event_ctr += 1
            if event_ctr >= io.num_entries():
                break
        iteration +=1
        
    print('Done running inference...')
    csv_batch.close()
    csv_event.close()
    io.finalize()

def io_test(flags):
    flags.TRAIN = False
    io = io_factory(flags)
    io.initialize()
    io.start_threads()
    num_entries = io.num_entries()
    ctr = 0
    data_check = 0
    nfailures = 0
    tspent_v = []
    while ctr < num_entries:
        tstart = time.time()
        voxel,feature,label,idx=io.next()
        tspent = time.time() - tstart
        tspent_v.append(tspent)
        msg = 'Read count {:d}/{:d} time {:g} index start={:d} end={:d} ({:d} entries) shape {:s}'
        msg = msg.format(ctr,num_entries,tspent,idx[0],idx[-1],len(idx),voxel.shape)
        ctr+=len(idx)
        print(msg)
        data_check += 1
        if data_check % 20 == 0:
            buf_start = voxel[0][0:3]
            buf_end   = voxel[-1][0:3]
            chk_start = io._voxel[idx[0]][0]
            chk_end   = io._voxel[idx[-1]][-1]
            good_start = (buf_start == chk_start).astype(np.int32).sum() == len(buf_start)
            good_end   = (buf_end   == chk_end  ).astype(np.int32).sum() == len(buf_end)

            print(buf_start,buf_end)
            print(chk_start,chk_end)
            print("Pass start/end? {:s}/{:s}".format(str(good_start),str(good_end)))
            if not good_start or not good_end:
                nfailures += 1
    io.finalize()
    tspent_v=np.array(tspent_v)
    print('Number of data check failures:',nfailures)
    print('Total time: {:g} [s] ... mean/std time-per-batch {:g}/{:g} [s]'.format(tspent_v.sum(),tspent_v.mean(),tspent_v.std()))
    

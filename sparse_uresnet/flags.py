from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torch
import numpy as np
import argparse, os, sys
from distutils.util import strtobool
from sparse_uresnet import train,inference,io_test
    
class FLAGS:
    # flags for IO
    IO_TYPE     = 'larcv'
    INPUT_FILE  = '/gpfs/slac/staas/fs1/g/neutrino/kterao/data/dlprod_ppn_v08_p02_test.root'
    OUTPUT_FILE = ''
    DATA_DIM    = 3
    BATCH_SIZE  = 10
    DATA_KEY    = 'data'
    LABEL_KEY   = 'segment'
    IMAGE_SIZE  = 512
    SHUFFLE     = 1
    NUM_THREADS = 1

    # flags for model
    NUM_CLASS  = 2
    NUM_STRIDE = 6
    NUM_BASE_FILTER=64
    TRAIN      = False
    KVALUE     = 20
    DEBUG      = True
    
    # flags for train/inference
    SEED           = -1
    LEARNING_RATE  = 0.001
    GPUS           = '1'
    LOG_DIR        = './'
    #MINIBATCH_SIZE = 1
    SAVE_WEIGHT    = './weight'
    INPUT_WEIGHT   = ''
    KVALUE         = 20
    NUM_POINT      = 2048
    NUM_CHANNEL    = -1
    ITERATION      = 10000
    REPORT_STEP    = 100
    SUMMARY_STEP     = 20
    CHECKPOINT_STEP  = 500
    CHECKPOINT_NUM   = 10
    CHECKPOINT_HOUR = 0.4

    def __init__(self):
        self._create_parser()

    def _common_arguments(self,parser):

        parser.add_argument('-ns','--num_stride',type=int,default=self.NUM_STRIDE,
                            help='The stride depth of the network [default %d]' % self.NUM_STRIDE)
        parser.add_argument('-nc','--num_class',type=int,default=self.NUM_CLASS,
                            help='The number of classes [default %d]' % self.NUM_CLASS)
        parser.add_argument('-nf','--num_base_filter',type=int,default=self.NUM_BASE_FILTER,
                            help='The number of base filter count of the network [default %d]' % self.NUM_BASE_FILTER)
        parser.add_argument('-ld','--log_dir', type=str,default=self.LOG_DIR,
                            help='Log dir [default: %s]' % self.LOG_DIR)
        parser.add_argument('--gpus', type=str, default=self.GPUS,
                            help='GPUs to utilize (comma-separated integers')
        parser.add_argument('-it','--iteration', type=int, default=self.ITERATION,
                            help='Iteration to run [default: %s]' % self.ITERATION)
        #parser.add_argument('-mbs','--minibatch_size', type=int, default=self.MINIBATCH_SIZE,
        #                    help='Mini-Batch Size during training for each GPU [default: %s]' % self.MINIBATCH_SIZE)
        parser.add_argument('-rs','--report_step', type=int, default=self.REPORT_STEP,
                            help='Period (in steps) to print out loss and accuracy [default: %s]' % self.REPORT_STEP)
        parser.add_argument('-iw','--input_weight', default=self.INPUT_WEIGHT,
                            help='Input weight file to be loaded [default: %s]' % self.INPUT_WEIGHT)
        ### IO ###
        parser.add_argument('-io','--io_type',type=str,default=self.IO_TYPE,
                            help='IO handler type [default: %s]' % self.IO_TYPE)
        parser.add_argument('-dd','--data_dim',type=int,default=self.DATA_DIM,
                            help='Input data dimension [default: %s]' % self.DATA_DIM)
        parser.add_argument('-is','--image_size',type=int,default=self.IMAGE_SIZE,
                            help='Image size [default: %s]' % self.IMAGE_SIZE)
        parser.add_argument('-if','--input_file',type=str,default=self.INPUT_FILE,
                            help='comma-separated input file list [default: %s]' % self.INPUT_FILE)
        parser.add_argument('-of','--output_file',type=str,default=self.OUTPUT_FILE,
                            help='output file name [default: %s]' % self.OUTPUT_FILE)
        parser.add_argument('-bs','--batch_size', type=int, default=self.BATCH_SIZE,
                            help='Batch Size during training for updating weights [default: %s]' % self.BATCH_SIZE)
        parser.add_argument('-dkey','--data_key',type=str,default=self.DATA_KEY,
                            help='A keyword to fetch data from file [default: %s]' % self.DATA_KEY)
        parser.add_argument('-lkey','--label_key',type=str,default=self.LABEL_KEY,
                            help='A keyword to fetch label from file [default: %s]' % self.LABEL_KEY)
        parser.add_argument('-sh','--shuffle',type=strtobool,default=self.SHUFFLE,
                            help='Shuffle the data entries [default: %s]' % self.SHUFFLE)
        parser.add_argument('-nt','--num_threads',type=int,default=self.NUM_THREADS,
                            help='Number of threads to generate batch data [default: %s]' % self.NUM_THREADS)
        
    def _create_parser(self):

        self.parser = argparse.ArgumentParser(description="SparseUResNet Flags")
        subparsers = self.parser.add_subparsers(title="Modules", description="Valid subcommands", dest='script',
                                                help="Script options")
        # train parser
        train_parser = subparsers.add_parser("train", help="Train SparseUResNet")
        # inference parser
        inference_parser = subparsers.add_parser("inference",help="Inference SparseUResNet")
        # io test parser
        iotest_parser = subparsers.add_parser("iotest", help="Test iotools")
        
        # attach common parsers
        self._common_arguments(train_parser)
        self._common_arguments(inference_parser)
        self._common_arguments(iotest_parser)

        train_parser.add_argument('-sd','--seed', default=self.SEED,
                                  help='Seed for random number generators [default: %s]' % self.SEED)
        train_parser.add_argument('-sw','--save_weight', default=self.SAVE_WEIGHT,
                                  help='Prefix for snapshots of weights [default: %s]' % self.SAVE_WEIGHT)
        train_parser.add_argument('-lr','--learning_rate', type=float, default=self.LEARNING_RATE,
                                  help='Initial learning rate [default: %s]' % self.LEARNING_RATE)
        train_parser.add_argument('-chks','--checkpoint_step', type=int, default=self.CHECKPOINT_STEP,
                                  help='Period (in steps) to store snapshot [default: %s]' % self.CHECKPOINT_STEP)
        
        # attach executables
        train_parser.set_defaults(func=train)
        inference_parser.set_defaults(func=inference)
        iotest_parser.set_defaults(func=io_test)
        
    def parse_args(self):
        args = self.parser.parse_args()
        self._update(vars(args))
        print("\n\n-- CONFIG --")
        for name in vars(self):
            attribute = getattr(self,name)
            if type(attribute) == type(self.parser): continue
            print("%s = %r" % (name, getattr(self, name)))
            
        # Set random seed for reproducibility
        torch.manual_seed(self.SEED)
        torch.cuda.manual_seed(self.SEED)
        np.random.seed(self.SEED)

        if not os.path.isdir(self.LOG_DIR):
            os.makedirs(self.LOG_DIR)
        if self.SAVE_WEIGHT.find('/')>=0:
            d = self.SAVE_WEIGHT[0:self.SAVE_WEIGHT.rfind('/')]
            if not os.path.isdir(d):
                os.makedirs(d)
        args.func(self)
        
    def _update(self, args):
        for name,value in args.iteritems():
            if name in ['func','script']: continue
            setattr(self, name.upper(), args[name])
        os.environ['CUDA_VISIBLE_DEVICES']=self.GPUS
        self.GPUS=[int(gpu) for gpu in self.GPUS.split(',')]
        self.INPUT_FILE=[str(f) for f in self.INPUT_FILE.split(',')]
        if self.SEED < 0:
            import time
            self.SEED = int(time.time())

if __name__ == '__main__':
    parse_args()

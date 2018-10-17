from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import sys
import time
import threading

def threadio_func(io_handle, thread_id):

    while 1:
        time.sleep(0.000001)
        while not io_handle._locks[thread_id]:
            idx_v     = []
            voxel_v   = []
            feature_v = []
            label_v   = []
            if io_handle._flags.SHUFFLE:
                idx_v = np.random.random([io_handle.batch_size()])*io_handle.num_entries()
                idx_v = idx_v.astype(np.int32)
            else:
                start = io_handle._start_idx[thread_id]
                end   = start + io_handle.batch_size()
                if end < io_handle.num_entries():
                    idx_v = np.arange(start,end)
                else:
                    idx_v = np.arange(start,io_handle.num_entries())
                    idx_v = np.concatenate([idx_v,np.arange(0,end-io_handle.num_entries())])
                next_start = start + len(io_handle._threads) * io_handle.batch_size()
                if next_start >= io_handle.num_entries():
                    next_start -= io_handle.num_entries()
                io_handle._start_idx[thread_id] = next_start
            
            for data_id, idx in enumerate(idx_v):
                voxel   = io_handle._voxel[idx]
                voxel_v.append(np.pad(voxel, [(0,0),(0,1)],'constant',constant_values=data_id))
                feature_v.append(io_handle._feature[idx])
                if len(io_handle._label):
                    label_v.append(io_handle._label[idx])
            voxel_v   = np.vstack(voxel_v)
            feature_v = np.vstack(feature_v)
            if len(label_v): label_v = np.hstack(label_v)
            io_handle._buffs[thread_id] = (voxel_v,feature_v,label_v,idx_v)
            io_handle._locks[thread_id] = True
    return

class io_base(object):

    def __init__(self,flags):
        self._batch_size   = flags.BATCH_SIZE
        self._num_entries  = -1
        self._num_channels = -1
        self._voxel        = [] # should be a list of numpy arrays
        self._feature      = [] # should be a list of numpy arrays, same length as self._voxel
        self._label        = [] # should be a list of numpy arrays, same length as self._voxel
        # For circular buffer / thread function controls
        self._locks   = [False] * flags.NUM_THREADS
        self._buffs   = [None ] * flags.NUM_THREADS
        self._threads = [None ] * flags.NUM_THREADS
        self._start_idx = [-1 ] * flags.NUM_THREADS
        self._last_buffer_id = -1
        self.set_index_start(0)

    def voxel   (self): return self._voxel
    def feature (self): return self._feature
    def label   (self): return self._label
    def num_entries(self): return self._num_entries
    def num_channels(self): return self._num_channels
        
    def stop_threads(self):
        if self._threads[0] is None:
            return
        for i in range(len(self._threads)):
            while self._locks[buffer_id]:
                time.sleep(0.000001)
            self._buffs[i] = None
            self._start_idx[i] = -1

    def set_index_start(self,idx):
        self.stop_threads()
        for i in range(len(self._threads)):
            self._start_idx[i] = idx + i*self._batch_size

    def start_threads(self):
        if self._threads[0] is not None:
            return
        for thread_id in range(len(self._threads)):
            print('Starting thread',thread_id)
            self._threads[thread_id] = threading.Thread(target = threadio_func, args=[self,thread_id])
            self._threads[thread_id].daemon = True
            self._threads[thread_id].start()
            
    def next(self,buffer_id=-1,release=True):
        if buffer_id >= len(self._locks):
            sys.stderr.write('Invalid buffer id requested: {:d}\n'.format(buffer_id))
            raise ValueError
        if buffer_id < 0: buffer_id = self._last_buffer_id + 1
        if buffer_id >= len(self._locks):
            buffer_id = 0
        if self._threads[buffer_id] is None:
            sys.stderr.write('Read-thread does not exist (did you initialize?)\n')
            raise ValueError
        while not self._locks[buffer_id]:
            time.sleep(0.000001)
        res = self._buffs[buffer_id]
        if release:
            self._buffs[buffer_id] = None
            self._locks[buffer_id] = False
            self._last_buffer_id   = buffer_id
        return res

    def batch_size(self,size=None):
        if size is None: return self._batch_size
        self._batch_size = int(size)

    def initialize(self):
        raise NotImplementedError

    def store(self,idx,softmax):
        raise NotImplementedError

    def finalize(self):
        raise NotImplementedError

class io_larcv(io_base):

    def __init__(self,flags):
        super(io_larcv,self).__init__(flags=flags)
        self._flags   = flags
        self._voxel   = None
        self._feature = None
        self._label   = None
        self._fout    = None
        self._last_entry = -1
        self._event_keys = []
        self._metas      = []

    def initialize(self):
        self._last_entry = -1
        self._event_keys = []
        self._metas = []
        # configure the input
        from larcv import larcv
        from ROOT import TChain
        ch_data   = TChain('sparse3d_%s_tree' % self._flags.DATA_KEY)
        ch_label  = None
        if self._flags.LABEL_KEY:
            ch_label  = TChain('sparse3d_%s_tree' % self._flags.LABEL_KEY)
        for f in self._flags.INPUT_FILE:
            ch_data.AddFile(f)
            if ch_label:  ch_label.AddFile(f)
        
        self._voxel   = []
        self._feature = []
        self._label   = []
        br_data,br_label=(None,None)
        event_fraction = 1./ch_data.GetEntries() * 100.
        total_point = 0.
        for i in range(ch_data.GetEntries()):
            ch_data.GetEntry(i)
            if ch_label:  ch_label.GetEntry(i)
            if br_data is None:
                br_data  = getattr(ch_data, 'sparse3d_%s_branch' % self._flags.DATA_KEY)
                if ch_label:  br_label  = getattr(ch_label, 'sparse3d_%s_branch' % self._flags.LABEL_KEY)
            num_point = br_data.as_vector().size()
            if num_point < 256: continue

            np_voxel   = np.zeros(shape=(num_point,3),dtype=np.int32)
            larcv.fill_3d_voxels(br_data, np_voxel)
            self._voxel.append(np_voxel)
            
            np_feature = np.zeros(shape=(num_point,1),dtype=np.float32)
            larcv.fill_3d_pcloud(br_data,  np_feature)
            self._feature.append(np_feature)
            
            self._event_keys.append((br_data.run(),br_data.subrun(),br_data.event()))
            self._metas.append(larcv.Voxel3DMeta(br_data.meta()))
            if ch_label:
                np_label = np.zeros(shape=(num_point,1),dtype=np.float32)
                larcv.fill_3d_pcloud(br_label, np_label)
                np_label = np_label.reshape([num_point]) - 1.
                self._label.append(np_label)
            total_point += num_point
            sys.stdout.write('Processed %d%% ... %d MB\r' % (int(event_fraction*i),int(total_point*4*2/1.e6)))
            sys.stdout.flush()

        sys.stdout.write('\n')
        sys.stdout.flush()
        self._num_channels = self._voxel[-1].shape[-1]
        self._num_entries = len(self._voxel)
        # Output
        if self._flags.OUTPUT_FILE:
            import tempfile
            cfg = '''
IOManager: {
      Verbosity:   2
      Name:        "IOManager"
      IOMode:      1
      OutFileName: "%s"
      InputFiles:  []
      InputDirs:   []
      StoreOnlyType: []
      StoreOnlyName: []
    }
                  '''
            cfg = cfg % self._flags.OUTPUT_FILE
            cfg_file = tempfile.NamedTemporaryFile('w')
            cfg_file.write(cfg)
            cfg_file.flush()
            self._fout = larcv.IOManager(cfg_file.name)
            self._fout.initialize()
            
    def store(self,idx,softmax):
        from larcv import larcv
        if self._fout is None:
            return
        idx=int(idx)
        if idx >= self.num_entries():
            raise ValueError
        keys = self._event_keys[idx]
        meta = self._metas[idx]
        
        larcv_data = self._fout.get_data('sparse3d',self._flags.DATA_KEY)
        voxel   = self._voxel[idx]
        feature = self._feature[idx].reshape([-1])
        vs = larcv.as_tensor3d(voxel,feature,meta,0.)
        larcv_data.set(vs,meta)

        score = np.max(softmax,axis=1).reshape([-1])
        prediction = np.argmax(softmax,axis=1).astype(np.float32).reshape([-1])
        
        larcv_softmax = self._fout.get_data('sparse3d','softmax')
        vs = larcv.as_tensor3d(voxel,score,meta,-1.)
        larcv_softmax.set(vs,meta)

        larcv_prediction = self._fout.get_data('sparse3d','prediction')
        vs = larcv.as_tensor3d(voxel,prediction,meta,-1.)
        larcv_prediction.set(vs,meta)
        
        if len(self._label) > 0:
            label = self._label[idx]
            label = label.astype(np.float32).reshape([-1])
            larcv_label = self._fout.get_data('sparse3d','label')
            vs = larcv.as_tensor3d(voxel,label,meta,-1.)
            larcv_label.set(vs,meta)

        self._fout.set_id(keys[0],keys[1],keys[2])
        self._fout.save_entry()
        
    def finalize(self):
        if self._fout:
            self._fout.finalize()
'''            
class io_h5(io_base):

    def __init__(self,flags):
        super(io_h5,self).__init__(flags=flags)
        self._flags  = flags
        self._data   = None
        self._label  = None
        self._fout   = None
        self._ohandler_data = None
        self._ohandler_label = None
        self._ohandler_softmax = None
        self._has_label = False

    def initialize(self):
        self._last_entry = -1
        # Prepare input
        import h5py as h5
        self._data   = None
        self._label  = None
        for f in self._flags.INPUT_FILE:
            f = h5.File(f,'r')
            if self._data is None:
                self._data  = np.array(f[self._flags.DATA_KEY ])
                if self._flags.LABEL_KEY : self._label  = np.array(f[self._flags.LABEL_KEY])
            else:
                self._data  = np.concatenate(self._data, np.array(f[self._flags.DATA_KEY ]))
                if self._label  : self._label  = np.concatenate(self._label, np.array(f[self._flags.LABEL_KEY ]))
        self._num_channels = self._data[-1].shape[-1]
        self._num_entries = len(self._data)
        # Prepare output
        if self._flags.OUTPUT_FILE:
            import tables
            FILTERS = tables.Filters(complib='zlib', complevel=5)
            self._fout = tables.open_file(self._flags.OUTPUT_FILE,mode='w', filters=FILTERS)
            data_shape = list(self._data[0].shape)
            data_shape.insert(0,0)
            self._ohandler_data = self._fout.create_earray(self._fout.root,self._flags.DATA_KEY,tables.Float32Atom(),shape=data_shape)
            self._ohandler_softmax = self._fout.create_earray(self._fout.root,'softmax',tables.Float32Atom(),shape=data_shape)
            if self._label:
                data_shape = list(self._label[0].shape)
                data_shape.insert(0,0)
                self._ohandler_label = self._fout.create_earray(self._fout.root,self._flags.LABEL_KEY,tables.Float32Atom(),shape=data_shape)
    def store(self,idx,softmax):
        if self._fout is None:
            raise NotImplementedError
        idx=int(idx)
        if idx >= self.num_entries():
            raise ValueError
        data = self._data[idx]
        self._ohandler_data.append(data[None])
        self._ohandler_softmax.append(softmax[None])
        if self._label is not None:
            label = self._label[idx]
            self._ohandler_label.append(label[None])

    def finalize(self):
        if self._fout:
            self._fout.close()
'''

def io_factory(flags):
    #if flags.IO_TYPE == 'h5':
    #    return io_h5(flags)
    if flags.IO_TYPE == 'larcv':
        return io_larcv(flags)
    raise NotImplementedError


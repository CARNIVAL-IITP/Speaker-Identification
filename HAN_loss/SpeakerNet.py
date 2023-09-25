#!/usr/bin/python
#-*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, pdb, sys, random, time, os, itertools, shutil, importlib
import numpy as np
from tuneThreshold import tuneThresholdfromScore
from DatasetLoader import test_dataset_loader
from torch.cuda.amp import autocast, GradScaler
#from hyperion.hyperion.score_norm import AdaptSNorm

class WrappedModel(nn.Module):
    def __init__(self, model):
        super(WrappedModel, self).__init__()
        self.module = model

    def forward(self, x, label=None):
        return self.module(x, label)


class SpeakerNet(nn.Module):
    def __init__(self, model, optimizer, trainfunc, num_utt, **kwargs):
        super(SpeakerNet, self).__init__()
        SpeakerNetModel = importlib.import_module('models.'+model).__getattribute__('MainModel')
        self.__S__ = SpeakerNetModel(**kwargs)
        LossFunction = importlib.import_module('loss.'+trainfunc).__getattribute__('LossFunction')
        self.__L__ = LossFunction(**kwargs)
        self.num_utt = num_utt
        self.num_pooling = kwargs.pop('num_pooling')
        self.dur_grouping = kwargs.pop('dur_grouping')

    def forward(self, data, label=None):
        if label == None:
            return self.__S__.forward(data.reshape(-1,data.size()[-1]).cuda(), aug=False) # from exp07
        else:
            if self.dur_grouping:
                outp = []
                for d in data: # 2022.03.16
                    outp += [self.__S__.forward(d.cuda(), aug=True)]
                outp = torch.stack(outp).transpose(0,1)                
            else:
                data = data.reshape(-1, data.size()[-1]).cuda() # from exp07
                outp = self.__S__.forward(data, aug=True)
                outp = outp.reshape(self.num_utt, -1, outp.size()[-1]).transpose(1,0).squeeze(1)
            nloss, prec1, prec2 = self.__L__.forward(outp, label)
            return nloss, prec1, prec2

class ModelTrainer(object):
    def __init__(self, speaker_model, optimizer, scheduler, gpu, **kwargs):
        self.__model__  = speaker_model
        Optimizer = importlib.import_module('optimizer.'+optimizer).__getattribute__('Optimizer')
        self.__optimizer__ = Optimizer(self.__model__.parameters(), **kwargs)
        Scheduler = importlib.import_module('scheduler.'+scheduler).__getattribute__('Scheduler')
        self.__scheduler__ = Scheduler(self.__optimizer__, **kwargs)
        self.scaler = GradScaler() 
        self.gpu = gpu
        self.ngpu = int(torch.cuda.device_count())
        self.ndistfactor = int(kwargs.pop('num_utt') * self.ngpu)
        self.dur_grouping = kwargs.pop('dur_grouping')
        #self.asnorm = AdaptSNorm(nbest=10)

    def train_network(self, loader, epoch, verbose):
        self.__model__.train()
        self.__scheduler__.step(epoch-1)
        bs = loader.batch_size
        df = self.ndistfactor
        cnt, idx, loss, top1, top2 = 0, 0, 0, 0, 0
        tstart = time.time()
        for data, data_label in loader:
            self.__model__.zero_grad()

            if self.dur_grouping: # 2022.03.16
                data = self.grouping(data)
            else:
                data = data.transpose(1,0) # from exp07
            label = torch.LongTensor(data_label).cuda()

            with autocast():
                nloss, prec1, prec2 = self.__model__(data, label)
            self.scaler.scale(nloss).backward()
            self.scaler.step(self.__optimizer__)
            self.scaler.update()

            loss    += nloss.detach().cpu().item()
            top1    += prec1.detach().cpu().item()
            top2    += prec2.detach().cpu().item()
            cnt += 1
            idx     += bs
            lr = self.__optimizer__.param_groups[0]['lr']
            telapsed = time.time() - tstart
            tstart = time.time()
            if verbose:
                sys.stdout.write("\rProcessing {:d} of {:d}: Loss {:f}, ACC1 {:2.3f}%, ACC2 {:2.3f}%, LR {:.6f} - {:.2f} Hz ".format(idx*df, loader.__len__()*bs*df, loss/cnt, top1/cnt, top2/cnt, lr, bs*df/telapsed))
                sys.stdout.flush()
        return (loss/cnt, top1/cnt, top2/cnt, lr)

    def grouping(self, data): # from exp14
        dr = [150, 300]
        data_g = []
        for i in range(data.size()[1]):
            data_g += [data[:, i, 0:(dr[i] * 160)]] #+ 240
        return data_g

    def evaluateFromList_with_snorm(self, test_list, test_path, train_list, train_path, score_norm, tta, num_thread, distributed, top_coh_size, eval_frames=0, num_eval=1, **kwargs):
        if distributed:
            rank = torch.distributed.get_rank()
        else:
            rank = 0
        self.__model__.eval()

        ## Eval loader ##
        feats_eval = {}
        tstart = time.time()
        with open(test_list) as f:
            lines_eval = f.readlines()
        files = list(itertools.chain(*[x.strip().split()[-2:] for x in lines_eval]))
        setfiles = list(set(files))
        setfiles.sort()
        test_dataset = test_dataset_loader(setfiles, test_path, eval_frames=eval_frames, num_eval=num_eval, **kwargs)
        if distributed:
            sampler = torch.utils.data.distributed.DistributedSampler(test_dataset, shuffle=False)
        else:
            sampler = None
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=num_thread, drop_last=False, sampler=sampler)
        ds = test_loader.__len__()
        gs = self.ngpu
        for idx, data in enumerate(test_loader):
            inp1 = data[0][0].cuda()
            with torch.no_grad():
                ref_feat = self.__model__(inp1).detach().cpu()
            feats_eval[data[1][0]] = ref_feat
            telapsed = time.time() - tstart
            if rank == 0:
                sys.stdout.write("\r Reading {:d} of {:d}: {:.2f} Hz, embedding size {:d}".format(idx*gs, ds*gs, idx*gs/telapsed,ref_feat.size()[1]))
                sys.stdout.flush()

        ## Cohort loader if using score normalization ##
        if score_norm:
            feats_coh = {}
            tstart = time.time()
            with open(train_list) as f:
                lines_coh = f.readlines()
            setfiles = list(set([x.split()[0] for x in lines_coh]))
            setfiles.sort()
            cohort_dataset = test_dataset_loader(setfiles, train_path, eval_frames=0, num_eval=1, **kwargs)
            if distributed:
                sampler = torch.utils.data.distributed.DistributedSampler(cohort_dataset, shuffle=False)
            else:
                sampler = None
            cohort_loader = torch.utils.data.DataLoader(cohort_dataset, batch_size=1, shuffle=False, num_workers=num_thread, drop_last=False, sampler=sampler)
            ds = cohort_loader.__len__()
            for idx, data in enumerate(cohort_loader):
                inp1 = data[0][0].cuda()
                with torch.no_grad():
                    ref_feat = self.__model__(inp1).detach().cpu()
                feats_coh[data[1][0]] = ref_feat
                telapsed = time.time() - tstart
                if rank == 0:
                    if idx==0: print('')
                    sys.stdout.write("\r Reading {:d} of {:d}: {:.2f} Hz, embedding size {:d}".format(idx*gs, ds*gs, idx*gs/telapsed,ref_feat.size()[1]))
                    sys.stdout.flush()
            coh_feat = torch.stack(list(feats_coh.values())).squeeze(1).cuda()
            if self.__model__.module.__L__.test_normalize:
                coh_feat = F.normalize(coh_feat, p=2, dim=1)

        ## Compute verification scores ##
        all_scores, all_labels = [], []
        if distributed:
            ## Gather features from all GPUs
            feats_eval_all = [None for _ in range(0,torch.distributed.get_world_size())]
            torch.distributed.all_gather_object(feats_eval_all, feats_eval)
            if score_norm:
                feats_coh_all = [None for _ in range(0,torch.distributed.get_world_size())]
                torch.distributed.all_gather_object(feats_coh_all, feats_coh)
        if rank == 0:
            tstart = time.time()
            print('')
            ## Combine gathered features
            if distributed:
                feats_eval = feats_eval_all[0]
                for feats_batch in feats_eval_all[1:]:
                    feats_eval.update(feats_batch)
                if score_norm:
                    feats_coh = feats_coh_all[0]
                    for feats_batch in feats_coh_all[1:]:
                        feats_coh.update(feats_batch)

            ## Read files and compute all scores
            for idx, line in enumerate(lines_eval):
                data = line.split()
                enr_feat = feats_eval[data[1]].cuda()
                tst_feat = feats_eval[data[2]].cuda()
                if self.__model__.module.__L__.test_normalize:
                    enr_feat = F.normalize(enr_feat, p=2, dim=1)
                    tst_feat = F.normalize(tst_feat, p=2, dim=1)

                if tta==True and score_norm==True:
                    print('Not considered condition')
                    exit()
                if tta == False:
                    score = F.cosine_similarity(enr_feat, tst_feat)

                if score_norm:
                    score_e_c = F.cosine_similarity(enr_feat, coh_feat)
                    score_c_t = F.cosine_similarity(coh_feat, tst_feat)

                    if top_coh_size == 0: top_coh_size = len(coh_feat)
                    score_e_c = torch.topk(score_e_c, k=top_coh_size, dim=0)[0]
                    score_c_t = torch.topk(score_c_t, k=top_coh_size, dim=0)[0]
                    score_e = (score - torch.mean(score_e_c, dim=0)) / torch.std(score_e_c, dim=0)
                    score_t = (score - torch.mean(score_c_t, dim=0)) / torch.std(score_c_t, dim=0)
                    score = 0.5 * (score_e + score_t)

                elif tta:
                    score = torch.mean(F.cosine_similarity(enr_feat.unsqueeze(-1), tst_feat.unsqueeze(-1).transpose(0,2)))

                all_scores.append(score.detach().cpu().numpy())
                all_labels.append(int(data[0]))
                telapsed = time.time() - tstart
                sys.stdout.write("\r Computing {:d} of {:d}: {:.2f} Hz".format(idx, len(lines_eval), idx/telapsed))
                sys.stdout.flush()
        return (all_scores, all_labels)


    def evaluateFromList_full_seg_with_snorm(self, test_list, test_path, train_list, train_path, score_norm, tta, num_thread, distributed, eval_frames=0, num_eval=1, type_coh='utt', top_coh_size=20000, **kwargs):
        if distributed:
            rank = torch.distributed.get_rank()
        else:
            rank = 0
        self.__model__.eval()
        print("score_norm")
        print(score_norm)
        ## Eval loader ##
        feats_eval  = {}
        tstart = time.time()
        with open(test_list) as f:
            lines_eval = f.readlines()
        files    = list(itertools.chain(*[x.strip().split()[-2:] for x in lines_eval]))
        setfiles = list(set(files))
        setfiles.sort()
        test_dataset = test_dataset_loader(setfiles, test_path, eval_frames=0, num_eval=1, **kwargs)
        if distributed:
            sampler = torch.utils.data.distributed.DistributedSampler(test_dataset, shuffle=False)
        else:
            sampler = None
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=num_thread, drop_last=False, sampler=sampler)
        ds = test_loader.__len__()
        gs = self.ngpu
        for idx, data in enumerate(test_loader):
            audio_1     = data[0][0].cuda()
            audio_fsize = audio_1
            audios = []
            audios += [audio_1]
            for dur in [500, 300, 150]:
                audio_dur = dur * 160 #+ 240
                # If full lenth is shorter than selecting duration, duplicate it
                if audio_1.shape[1] <= audio_dur:
                    audio = torch.cat((audio_1, audio_1), axis=1)
                    audio_size = audio.shape[1]
                else:
                    audio = audio_1
                    audio_size = audio_1.shape[1]
                audio = audio[:, int(audio_size/2) - int(audio_dur/2) : int(audio_size/2) + int(audio_dur/2)]
                audios += [audio]
            with torch.no_grad():
                ref_feat_1 = self.__model__(audios[0]).detach().cpu() # full
                ref_feat_2 = self.__model__(audios[1]).detach().cpu() # 5.0sec
                ref_feat_3 = self.__model__(audios[2]).detach().cpu() # 3.0sec
                ref_feat_4 = self.__model__(audios[3]).detach().cpu() # 1.5sec
            feats_eval[data[1][0]] = [ref_feat_1, ref_feat_2, ref_feat_3, ref_feat_4]
            telapsed = time.time() - tstart
            if rank == 0:
                sys.stdout.write("\r Reading {:d} of {:d}: {:.2f} Hz, embedding size {:d}".format(idx*gs, ds*gs, idx*gs/telapsed,ref_feat_1.size()[1]))
                sys.stdout.flush()

        ## Cohort loader if using score normalization ##
        if score_norm:
            feats_coh, labels_coh = {}, {}
            tstart = time.time()
            with open(train_list) as f:
                lines_coh = f.readlines() 
            files  = [x.split()[0] for x in lines_coh]
            labels = [x.split()[1] for x in lines_coh]
            cohort_dataset = test_dataset_loader(files, train_path, eval_frames=0, num_eval=1, label=labels, **kwargs)
            if distributed:
                sampler = torch.utils.data.distributed.DistributedSampler(cohort_dataset, shuffle=False)
            else:
                sampler = None
            cohort_loader = torch.utils.data.DataLoader(cohort_dataset, batch_size=1, shuffle=False, num_workers=num_thread, drop_last=False, sampler=sampler)
            ds = cohort_loader.__len__()
            for idx, data in enumerate(cohort_loader):
                inp1 = data[0][0].cuda()
                spk  = data[2]
                with torch.no_grad():
                    ref_feat = self.__model__(inp1).detach().cpu()
                feats_coh[data[1][0]]  = ref_feat
                labels_coh[data[1][0]] = spk
                telapsed = time.time() - tstart
                if rank == 0:
                    if idx==0: print('')
                    sys.stdout.write("\r Reading {:d} of {:d}: {:.2f} Hz, embedding size {:d}".format(idx*gs, ds*gs, idx*gs/telapsed,ref_feat.size()[1]))
                    sys.stdout.flush()

        ## Compute verification scores ##
        all_scores_1, all_scores_2, all_scores_3, all_scores_4, all_labels = [], [], [], [], []
        if distributed:
            ## Gather features from all GPUs
            feats_eval_all = [None for _ in range(0,torch.distributed.get_world_size())]
            torch.distributed.all_gather_object(feats_eval_all, feats_eval)
            if score_norm:
                feats_coh_all = [None for _ in range(0,torch.distributed.get_world_size())]
                torch.distributed.all_gather_object(feats_coh_all, feats_coh)
                if type_coh == 'spk':
                   labels_coh_all = [None for _ in range(0,torch.distributed.get_world_size())]
                   torch.distributed.all_gather_object(labels_coh_all, labels_coh)
        if rank == 0:
            tstart = time.time()
            print('')
            ## Combine gathered features
            if distributed:
                feats_eval = feats_eval_all[0]
                for feats_batch in feats_eval_all[1:]:
                    feats_eval.update(feats_batch)
                if score_norm:
                    feats_coh = feats_coh_all[0]
                    for feats_batch in feats_coh_all[1:]:
                        feats_coh.update(feats_batch)
                    if type_coh=='spk':
                        for labels_batch in labels_coh_all[1:]:
                            labels_coh.update(labels_batch)
            if score_norm:
                coh_feat = torch.stack(list(feats_coh.values())).squeeze(1).cuda()
                if self.__model__.module.__L__.test_normalize:
                    coh_feat  = F.normalize(coh_feat, p=2, dim=1)
                if type_coh=='spk':
                    coh_label = np.array(list(labels_coh.values()))
                    coh_feat_ = []
                    ## Speaker-level average, e.g., vox2-dev: 5994
                    for i in range(len(np.unique(coh_label))):
                        idx = [k for k, x in enumerate(coh_label) if x == str(i)]
                        coh_feat_ += [torch.mean(coh_feat[idx], axis=0)]
                    coh_feat = torch.stack(coh_feat_)

            ## Read files and compute all scores
            for idx, line in enumerate(lines_eval):
                data = line.split()
                enr_feat_1 = feats_eval[data[1]][0].cuda() # enr - full
                tst_feat_1 = feats_eval[data[2]][0].cuda() # tst - full
                tst_feat_2 = feats_eval[data[2]][1].cuda() # tst - 5.0sec
                tst_feat_3 = feats_eval[data[2]][2].cuda() # tst - 3.0sec
                tst_feat_4 = feats_eval[data[2]][3].cuda() # tst - 1.5sec
                if self.__model__.module.__L__.test_normalize:
                    enr_feat_1 = F.normalize(enr_feat_1, p=2, dim=1)
                    tst_feat_1 = F.normalize(tst_feat_1, p=2, dim=1)
                    tst_feat_2 = F.normalize(tst_feat_2, p=2, dim=1)
                    tst_feat_3 = F.normalize(tst_feat_3, p=2, dim=1)
                    tst_feat_4 = F.normalize(tst_feat_4, p=2, dim=1)

                if tta==True and score_norm==True:
                    print('Not considered condition')
                    exit()
                if tta == False:
                    score_1 = F.cosine_similarity(enr_feat_1, tst_feat_1)
                    score_2 = F.cosine_similarity(enr_feat_1, tst_feat_2)
                    score_3 = F.cosine_similarity(enr_feat_1, tst_feat_3)
                    score_4 = F.cosine_similarity(enr_feat_1, tst_feat_4)

                if score_norm:
                    score_e1_c = F.cosine_similarity(enr_feat_1, coh_feat)
                    score_c_t1 = F.cosine_similarity(coh_feat, tst_feat_1)
                    score_c_t2 = F.cosine_similarity(coh_feat, tst_feat_2)
                    score_c_t3 = F.cosine_similarity(coh_feat, tst_feat_3)
                    score_c_t4 = F.cosine_similarity(coh_feat, tst_feat_4)

                    if top_coh_size == 0: top_coh_size = len(coh_feat)
                    score_e1_c = torch.topk(score_e1_c, k=top_coh_size, dim=0)[0]
                    score_c_t1 = torch.topk(score_c_t1, k=top_coh_size, dim=0)[0]
                    score_c_t2 = torch.topk(score_c_t2, k=top_coh_size, dim=0)[0]
                    score_c_t3 = torch.topk(score_c_t3, k=top_coh_size, dim=0)[0]
                    score_c_t4 = torch.topk(score_c_t4, k=top_coh_size, dim=0)[0]

                    # full - full
                    score_e = (score_1 - torch.mean(score_e1_c, dim=0)) / torch.std(score_e1_c, dim=0)
                    score_t = (score_1 - torch.mean(score_c_t1, dim=0)) / torch.std(score_c_t1, dim=0)
                    score_1 = 0.5 * (score_e + score_t)

                    # full - 5.0sec                   
                    score_e = (score_2 - torch.mean(score_e1_c, dim=0)) / torch.std(score_e1_c, dim=0)
                    score_t = (score_2 - torch.mean(score_c_t2, dim=0)) / torch.std(score_c_t2, dim=0)
                    score_2 = 0.5 * (score_e + score_t)

                    # full - 3.0sec
                    score_e = (score_3 - torch.mean(score_e1_c, dim=0)) / torch.std(score_e1_c, dim=0)
                    score_t = (score_3 - torch.mean(score_c_t3, dim=0)) / torch.std(score_c_t3, dim=0)
                    score_3 = 0.5 * (score_e + score_t)

                    # full - 1.5sec
                    score_e = (score_4 - torch.mean(score_e1_c, dim=0)) / torch.std(score_e1_c, dim=0)
                    score_t = (score_4 - torch.mean(score_c_t4, dim=0)) / torch.std(score_c_t4, dim=0)
                    score_4 = 0.5 * (score_e + score_t)

                elif tta:
                    score_1 = np.mean(F.cosine_similarity(enr_feat_1.unsqueeze(-1), tst_feat_1.unsqueeze(-1).transpose(0,2)))
                    score_2 = np.mean(F.cosine_similarity(enr_feat_1.unsqueeze(-1), tst_feat_2.unsqueeze(-1).transpose(0,2)))
                    score_3 = np.mean(F.cosine_similarity(enr_feat_1.unsqueeze(-1), tst_feat_3.unsqueeze(-1).transpose(0,2)))
                    score_4 = np.mean(F.cosine_similarity(enr_feat_1.unsqueeze(-1), tst_feat_4.unsqueeze(-1).transpose(0,2)))

                all_scores_1.append(score_1.detach().cpu().numpy())
                all_scores_2.append(score_2.detach().cpu().numpy())
                all_scores_3.append(score_3.detach().cpu().numpy())
                all_scores_4.append(score_4.detach().cpu().numpy())
                all_labels.append(int(data[0]))
                telapsed = time.time() - tstart
                sys.stdout.write("\r Computing {:d} of {:d}: {:.2f} Hz".format(idx, len(lines_eval), idx/telapsed))
                sys.stdout.flush()
        return (all_scores_1, all_scores_2, all_scores_3, all_scores_4, all_labels)

    def evaluateFromList_H(self, test_list, test_path, train_list, train_path, num_thread, distributed, eval_frames=0, num_eval=1, **kwargs):
        if distributed:
            rank = torch.distributed.get_rank()
        else:
            rank = 0
        self.__model__.eval()
        ## Eval loader ##
        feats_eval  = {}
        tstart = time.time()
        with open(test_list) as f:
            lines_eval = f.readlines()
        files    = list(itertools.chain(*[x.strip().split()[-2:] for x in lines_eval]))
        setfiles = list(set(files))
        setfiles.sort()
        test_dataset = test_dataset_loader(setfiles, test_path, eval_frames=0, num_eval=1, **kwargs)
        if distributed:
            sampler = torch.utils.data.distributed.DistributedSampler(test_dataset, shuffle=False)
        else:
            sampler = None
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=num_thread, drop_last=False, sampler=sampler)
        ds = test_loader.__len__()
        gs = self.ngpu
        for idx, data in enumerate(test_loader):
            audio     = data[0][0].cuda()
            with torch.no_grad():
                ref_feat_1 = self.__model__(audio).detach().cpu() # full
            feats_eval[data[1][0]] = ref_feat_1.unsqueeze(0)
            telapsed = time.time() - tstart
            if rank == 0:
                sys.stdout.write("\r Reading {:d} of {:d}: {:.2f} Hz, embedding size {:d}".format(idx*gs, ds*gs, idx*gs/telapsed,ref_feat_1.size()[1]))
                sys.stdout.flush()


        ## Compute verification scores ##
        all_scores_1, all_labels = [], []
        if distributed:
            ## Gather features from all GPUs
            feats_eval_all = [None for _ in range(0,torch.distributed.get_world_size())]
            torch.distributed.all_gather_object(feats_eval_all, feats_eval)
            
        if rank == 0:
            tstart = time.time()
            print('')
            ## Combine gathered features
            if distributed:
                feats_eval = feats_eval_all[0]
                for feats_batch in feats_eval_all[1:]:
                    feats_eval.update(feats_batch)

            ## Read files and compute all scores
            for idx, line in enumerate(lines_eval):
                data = line.split()
                enr_feat_1 = feats_eval[data[1]][0].cuda() # enr - full
                tst_feat_1 = feats_eval[data[2]][0].cuda() # tst - full
                if self.__model__.module.__L__.test_normalize:

                    enr_feat_1 = F.normalize(enr_feat_1, p=2, dim=1)
                    tst_feat_1 = F.normalize(tst_feat_1, p=2, dim=1)

                score_1 = F.cosine_similarity(enr_feat_1, tst_feat_1)
                
                all_scores_1.append(score_1.detach().cpu().numpy())
                all_labels.append(int(data[0]))
                telapsed = time.time() - tstart
                #sys.stdout.write("\r Computing {:d} of {:d}: {:.2f} Hz".format(idx, len(lines_eval), idx/telapsed))
                #sys.stdout.flush()
        return (all_scores_1, all_labels)

    def saveParameters(self, path):
        torch.save(self.__model__.module.state_dict(), path)

    def loadParameters(self, path):
        self_state = self.__model__.module.state_dict()
        loaded_state = torch.load(path, map_location="cuda:%d"%self.gpu)
        for name, param in loaded_state.items():
            origname = name
            if name not in self_state:
                name = name.replace("module.", "")
                if name not in self_state:
                    print("{} is not in the model.".format(origname))
                    continue
            if self_state[name].size() != loaded_state[origname].size():
                print("Wrong parameter length: {}, model: {}, loaded: {}".format(origname, self_state[name].size(), loaded_state[origname].size()))
                continue
            self_state[name].copy_(param)
            
    def evaluateFromList(self, test_list, test_path, train_list, train_path, num_thread, distributed, eval_frames=0, num_eval=1, **kwargs):
        if distributed:
            rank = torch.distributed.get_rank()
        else:
            rank = 0
        self.__model__.eval()
        ## Eval loader ##
        feats_eval  = {}
        tstart = time.time()
        with open(test_list) as f:
            lines_eval = f.readlines()
        files    = list(itertools.chain(*[x.strip().split()[-2:] for x in lines_eval]))
        setfiles = list(set(files))
        setfiles.sort()
        test_dataset = test_dataset_loader(setfiles, test_path, eval_frames=0, num_eval=1, **kwargs)
        if distributed:
            sampler = torch.utils.data.distributed.DistributedSampler(test_dataset, shuffle=False)
        else:
            sampler = None
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=num_thread, drop_last=False, sampler=sampler)
        ds = test_loader.__len__()
        gs = self.ngpu
        for idx, data in enumerate(test_loader):
            audio     = data[0][0].cuda()
            with torch.no_grad():
                ref_feat_1 = self.__model__(audio).detach().cpu() # full
            feats_eval[data[1][0]] = ref_feat_1.unsqueeze(0)
            telapsed = time.time() - tstart
            if rank == 0:
                sys.stdout.write("\r Reading {:d} of {:d}: {:.2f} Hz, embedding size {:d}".format(idx*gs, ds*gs, idx*gs/telapsed,ref_feat_1.size()[1]))
                sys.stdout.flush()


        ## Compute verification scores ##
        all_scores_1, all_labels = [], []
        if distributed:
            ## Gather features from all GPUs
            feats_eval_all = [None for _ in range(0,torch.distributed.get_world_size())]
            torch.distributed.all_gather_object(feats_eval_all, feats_eval)
            
        if rank == 0:
            tstart = time.time()
            print('')
            ## Combine gathered features
            if distributed:
                feats_eval = feats_eval_all[0]
                for feats_batch in feats_eval_all[1:]:
                    feats_eval.update(feats_batch)

            ## Read files and compute all scores
            for idx, line in enumerate(lines_eval):
                data = line.split()
                enr_feat_1 = feats_eval[data[1]][0].cuda() # enr - full
                tst_feat_1 = feats_eval[data[2]][0].cuda() # tst - full
                if self.__model__.module.__L__.test_normalize:

                    enr_feat_1 = F.normalize(enr_feat_1, p=2, dim=1)
                    tst_feat_1 = F.normalize(tst_feat_1, p=2, dim=1)

                score_1 = F.cosine_similarity(enr_feat_1, tst_feat_1)
                
                all_scores_1.append(score_1.detach().cpu().numpy())
                all_labels.append(int(data[0]))
                telapsed = time.time() - tstart
                sys.stdout.write("\r Computing {:d} of {:d}: {:.2f} Hz".format(idx, len(lines_eval), idx/telapsed))
                sys.stdout.flush()
        return (all_scores_1, all_labels)

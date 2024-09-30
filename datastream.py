import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.linalg import vector_norm

import zipfile 
import itertools
import os
import json
import pandas as pd
import numpy as np
import traceback

import matplotlib
matplotlib.use('nbagg') # chart zooming
matplotlib.rcParams['agg.path.chunksize'] = 10000


import matplotlib.pyplot as plt
from sklearn.metrics import *
import gc
import copy
import math

import dataclasses
from collections import defaultdict
from tqdm import tqdm

from river.drift import PageHinkley
from river.drift.binary import HDDM_W
import time
from statistics import mean
from math import *

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score



CUDA = torch.cuda.is_available()
device = torch.device("cuda" if CUDA else "cpu")
print('device =', device)

plt.style.use(["seaborn-v0_8-paper"])

TITLE_SIZE=48
TICK_SIZE=38
TICK_LABEL_SIZE=42
TEXT_SIZE=40

tex_fonts = {
    # Use LaTeX to write all text
    # "text.usetex": True,
    "font.family": "serif",
    # Use 10pt font in plots, to match 10pt font in document
    "axes.labelsize": TICK_LABEL_SIZE,
    "font.size": TEXT_SIZE,
    # Make the legend/label fonts a little smaller
    "legend.fontsize": TICK_SIZE,
    "xtick.labelsize": TICK_SIZE,
    "ytick.labelsize": TICK_SIZE, 
    "axes.titlesize": TITLE_SIZE
}

plt.rcParams.update(tex_fonts)

batch_size = 2000
epochs = 20
skiprows = 0  # 1100000
# category='MSNews'
window_size=20
lr = 1e-3

ENABLE_SAMPLING = True
ENABLE_HITRATE_STAT = True
DISABLE_TEMP_CHART = True
step = 20
negative_sampling_processes=4
#negative_sampling_processes = 10


# # Evaluation
# 
# Sample code to use evaluator object
# 
#     acc = MetricAccumulator()
#     t_user = torch.abs(torch.randn(dataLoader.batch_size, 1)* 10).long().view(-1, 1)
#     t_pos = torch.abs(torch.randn(dataLoader.batch_size, 1)* 10).long().view(-1, 1)
#     t_neg = 10 + torch.arange(100).repeat(dataLoader.batch_size, 1)
# 
#     t_score = torch.abs(torch.randn(dataLoader.batch_size, t_pos.shape[1] + t_neg.shape[1]))
#     t_score /= t_score.max()
# 
#     print(t_user.shape, t_pos.shape, t_neg.shape)
# 
#     acc.compute_metric(t_user, t_pos, t_neg, t_score, top_k=1)
#     acc.compute_metric(t_user, t_pos, t_neg, t_score, top_k=10)
# 
#     print(acc.get_metrics())
# 
#     for k, values in acc.get_metrics().items():
#         for v in values.metric_names():
#             print(f'{v}@{k} = {values[v]}')
# 

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)
        

class UserHistoryRecorder:
    def __init__(self):
        self._max_item_id = 0
        self._user_history = defaultdict(set)
    
    def update(self, users_tensor, pos_items_tensor, items_max_id):
        """Update positives seen by users
        
        :param users_tensor: bsx1 with user id (repeated)
        :param pos_items_tensor: bsx1 with positive id for user at position i in users_tensor
        :param items_max_id: max items' id at batch i
        """
        for u, p in zip(users_tensor, pos_items_tensor):
            u, p = u.item(), p.item()            
            self._user_history[u].add(p)
            
        self._max_item_id = max(self._max_item_id, items_max_id)
        
    def sample(self, users_tensor, pos_items_tensor, negatives_number=100):
        """Create a test set sample as:
        t_user is the array with distinct user in user_tensor, 
        t_pos is the array with positives of users in t_user provided by pos_items_tensor, 
        t_neg contains the number of negatives to generate for each positive in t_pos 
        
        !Alert: call update(users_tensor, pos_items_tensor) before this method
        
        :param users_tensor: bsx1 with user id (repeated)
        :param pos_items_tensor: bsx1 with positive id for user at position i in users_tensor
        :param negatives_number: number of negatives to generate for each postive in t_pos
        :return: t_user, t_pos, t_neg 
        """
        items_set = set(range(self._max_item_id + 1))
        t_neg = []
        
        for u, p in zip(users_tensor, pos_items_tensor):
            u, p = u.item(), p.item()
            negative_candidates = list(items_set - self._user_history[u])
            sample_with_replacement = len(negative_candidates) < negatives_number
            
            neg = np.random.choice(negative_candidates, size=negatives_number, replace=sample_with_replacement)
            t_neg.append(neg)
        
        return users_tensor, pos_items_tensor, torch.tensor(t_neg, device=users_tensor.device, dtype=torch.int)
    
    def __str__(self):
        return f'max_item_id = {self._max_item_id}, #users {len(self._user_history)}'
    
# ONLY for debug
# history_recorder = UserHistoryRecorder()
# a = torch.abs(torch.randn(dataLoader.batch_size, 1)* 10).long()
# b = torch.abs(torch.randn(dataLoader.batch_size, 1)* 10).long()
# history_recorder.update(a, b, b.max().item())
# print(history_recorder)

# t_user, t_pos, t_neg  = history_recorder.sample(a, b)
# print(t_user.shape, t_pos.shape, t_neg.shape)

# a= None 
# b = None 
# batch = None
# history_recorder = None
# t_user, t_pos, t_neg = [None] * 3


import collections
import numpy as np


class SimpleMetric:
    def __init__(self):
        self.recall = 0
        self.recall_den = 0
        self.hitrate = 0
        self.hitrate_den = 0
        self._num_users = 0

    def metric_names(self):
        return ('recall',
                'hitrate')

    def __getitem__(self, item):
        v = getattr(self, item)
        if isinstance(v, np.ndarray):
            return ','.join([f'{j:.2f}' for j in v])
        return v

    def __str__(self):
        return f'recall = {self.recall:.2f}, ' \
               f'hitrate = {self.hitrate:.2f} '

    def __repr__(self):
        return self.__str__()


class MetricAccumulator:
    def __init__(self):
        self.data = collections.defaultdict(SimpleMetric)

    def reset(self):
        self.data.clear()

    def get_top_k(self):
        return sorted(self.data.keys())

    def get_metrics(self, top_k=None):
        result = {}

        with np.errstate(divide='ignore', invalid='ignore'):
            k_list = [top_k] if top_k else self.data.keys()
            for k in k_list:
                acc = self.data[k]

                computed_acc = SimpleMetric()
                #computed_acc.recall = acc.recall / acc.recall_den
                #computed_acc.hitrate = acc.hitrate / acc._num_users
                computed_acc.hitrate = acc.hitrate / acc.hitrate_den

                result[k] = computed_acc

        return result[top_k] if top_k else result

    def compute_metric(self, t_user, t_pos, t_neg, t_score, top_k=10):
        """
        Compute metric:
        - hitrate

        :param t_user: #user, BSx1 with user id 
        :param t_pos: #positives, BSx1 with positive id 
        :param t_neg: #negatives, BSx#negs with negative id 
        :param t_score: score for positives and negatives, BSx(1+negs)
        :param top_k: top k items to select ranked by score
        """
        assert t_user.shape[0] == t_pos.shape[0] == t_neg.shape[0] == t_score.shape[0] \
               and 1 + t_neg.shape[1] == t_score.shape[1] and top_k > 0, \
               f"fail with top_k = {top_k} and t_score = {t_score.shape}"
        
        
        accumulator = self.data[top_k]
        
        pos_score = t_score[:, 0].view(-1, 1)
        neg_score = t_score[:, 1:]
        
        # find k-th highest score of the negatives    
        _, indices = torch.kthvalue(-neg_score, k=top_k, dim=-1, keepdim=True)
        
        # compare with positive
        k_th_prob = torch.gather(neg_score, 1, indices)
        best_positives = pos_score >= k_th_prob

        # update counter
        accumulator.hitrate += best_positives.sum().item()
        accumulator.hitrate_den += best_positives.shape[0]


# EXAMPLE, only for test

# acc = MetricAccumulator()
# t_user = torch.abs(torch.randn(dataLoader.batch_size, 1)* 10).long().view(-1, 1)
# t_pos = torch.abs(torch.randn(dataLoader.batch_size, 1)* 10).long().view(-1, 1)
# t_neg = 10 + torch.arange(100).repeat(dataLoader.batch_size, 1)

# t_score = torch.abs(torch.randn(dataLoader.batch_size, t_pos.shape[1] + t_neg.shape[1]))
# t_score /= t_score.max()

# print(t_user.shape, t_pos.shape, t_neg.shape)

# acc.compute_metric(t_user, t_pos, t_neg, t_score, top_k=1)
# acc.compute_metric(t_user, t_pos, t_neg, t_score, top_k=10)

# print(acc.get_metrics())

# for k, values in acc.get_metrics().items():
    # for v in values.metric_names():
        # print(f'{v}@{k} = {values[v]}')



# # Matrix Factorization Models

class BPR_MatrixFactorization(torch.nn.Module):
    def __init__(self, M, N, K=20, model=None):
        super().__init__()
        self.latent_size = K

        self.P = torch.nn.Embedding(M, K)
        self.Q = torch.nn.Embedding(N, K)

        if model is None:
          nn.init.normal_(self.P.weight, std=0.01)
          nn.init.normal_(self.Q.weight, std=0.01)
        else:
          self.load_state_dict(model.state_dict())

    def forward(self, users, items, neg_items=None):  

        '''
        When we use neg_items, for each tuple (user, pos_item) we have to provide the following input:

        user pos_item neg_item_1
        user pos_item neg_item_2
        user pos_item neg_item_3
        user pos_item neg_item_4
        '''

        users_e = self.P(users)
        items_e = self.Q(items)

        pos_prediction = (users_e * items_e).sum(dim=-1)

        if neg_items is not None:
            neg_items_e = self.Q(neg_items)

            neg_prediction = (users_e * neg_items_e).sum(dim=-1)
            return pos_prediction, neg_prediction

        return pos_prediction
    

def resize_matrices(model: BPR_MatrixFactorization, new_K):
    P_size = model.P.weight.shape
    P_weight = torch.zeros((P_size[0], new_K))
    
    Q_size = model.Q.weight.shape
    Q_weight = torch.zeros((Q_size[0], new_K))
    
    P_weight[:, :P_size[1]] = model.P.weight
    Q_weight[:, :Q_size[1]] = model.Q.weight
    
    model.P = torch.nn.Embedding.from_pretrained(P_weight,freeze=False)
    model.Q = torch.nn.Embedding.from_pretrained(Q_weight,freeze=False)
    
    return model


# ## Losses
class BPRLoss(torch.nn.Module):

  def __init__(self):
    super().__init__()
    self.logSigmoid = nn.LogSigmoid()

  def forward(self, pos_scores, neg_scores):
    return torch.mean(-self.logSigmoid(pos_scores - neg_scores))


class DG_BPRLoss(torch.nn.Module):

  def __init__(self, alpha=1, beta=1):
    super().__init__()
    self.logSigmoid = nn.LogSigmoid()
    self._alpha = alpha
    self._beta = beta

  def forward(self, users, pos_items, neg_items, 
              pos_scores=None, neg_scores=None, 
              model=None, old_model=None, delta_mask=None):
    """
      Compute losses as L = L^bpr + L^sim + L^reg

      :param old_model: if None, the simple loss (L^bpr + L^reg) is returned

      :return: loss, loss_bpr, loss_sim, loss_reg
    """

    Lbpr = torch.mean(-self.logSigmoid(pos_scores - neg_scores))

    if model:
      Lreg = self._beta * (vector_norm(model.P.weight, ord=np.inf) \
             + vector_norm(model.Q.weight, ord=np.inf))
    else:
      Lreg = torch.tensor(0)

    if old_model is None:
      return Lbpr + Lreg, Lbpr, torch.tensor(0), Lreg
      
    users_e = model.P(users)
    pos_items_e = model.Q(pos_items)
    neg_items_e = model.Q(neg_items)

    old_users_e = old_model.P(users).detach()
    old_pos_items_e = old_model.Q(pos_items).detach()
    old_neg_items_e = old_model.Q(neg_items).detach()

    #sim1 = torch.mean(vector_norm(delta_mask * (users_e - old_users_e), ord=1))
    #sim2 = torch.mean(vector_norm(delta_mask * (pos_items_e - old_pos_items_e), ord=1))
    #sim3 = torch.mean(vector_norm(delta_mask * (neg_items_e - old_neg_items_e), ord=1))

    sim1 = vector_norm(delta_mask * (users_e - old_users_e), ord=1)
    sim2 = vector_norm(delta_mask * (pos_items_e - old_pos_items_e), ord=1)
    sim3 = vector_norm(delta_mask * (neg_items_e - old_neg_items_e), ord=1)

    #print('SIM1-', delta_mask.sum(), (users_e - old_users_e).shape, sim1.shape, sim1)

    loss_sim = self._alpha * (sim1 + sim2 + sim3)
    loss = Lbpr + Lreg + loss_sim

    return loss, Lbpr, loss_sim, Lreg
    

def save_charts(folder, windows_length=5):
    """
    Save charts on files
    """
    if DISABLE_TEMP_CHART:
        return
    
    if not os.path.exists(folder):
        os.mkdir(folder)
    
    # loss windowed 
    fig = plt.figure(figsize=(20, 8))
    plt.xlabel('batches')
    plt.ylabel('bpr loss')
    #plt.yscale('log') 
    plt.plot(ma(resultBean.bpr_loss_list, windows_length),linewidth=0.8)
    for d in resultBean.drift_points:
        plt.axvline(d, color='red',linewidth=0.8,alpha=0.7)
    for w in resultBean.warning_points:
        plt.axvline(w, color='orange',linewidth=0.8,alpha=0.4)

    fig.savefig(os.path.join(folder, 'train_loss_windowed.pdf'))
    plt.close(fig)
    
    
    # loss per batch
    fig = plt.figure(figsize=(20, 8))
    plt.xlabel('batches')
    plt.ylabel('bpr loss')
    #plt.yscale('log') 
    plt.plot(resultBean.bpr_loss_list,linewidth=0.8)
    for d in resultBean.drift_points:
        plt.axvline(d, color='red',linewidth=0.8,alpha=0.7)
    for w in resultBean.warning_points:
        plt.axvline(w, color='orange',linewidth=0.8,alpha=0.4)

    fig.savefig(os.path.join(folder, 'train_loss_per_batch.pdf'))
    plt.close(fig)
    
    
    # test loss windowed
    fig = plt.figure(figsize=(20, 8))
    plt.xlabel('batches')
    plt.ylabel('bpr loss')
    #plt.yscale('log') 
    plt.plot(ma(resultBean.testset_loss_list, windows_length),linewidth=0.8)
    for d in resultBean.drift_points:
        plt.axvline(d, color='red',linewidth=0.8,alpha=0.7)
    for w in resultBean.warning_points:
        plt.axvline(w, color='orange',linewidth=0.8,alpha=0.4)

    fig.savefig(os.path.join(folder, 'test_loss_windowed.pdf'))
    plt.close(fig)
    
    
    # test loss 
    fig = plt.figure(figsize=(20, 8))
    plt.xlabel('batches')
    plt.ylabel('bpr loss')
    #plt.yscale('log') 
    plt.plot(resultBean.testset_loss_list,linewidth=0.8)
    for d in resultBean.drift_points:
        plt.axvline(d, color='red',linewidth=0.8,alpha=0.7)
    for w in resultBean.warning_points:
        plt.axvline(w, color='orange',linewidth=0.8,alpha=0.4)

    fig.savefig(os.path.join(folder, 'test_loss_per_batch.pdf'))
    plt.close(fig)
    
    # HITRATE
    if sum([len(x) for x in resultBean.metrics_stream]) > 0:
        fig, axs = plt.subplots(len(resultBean.metrics_stream), 1, figsize=(20, 8))

        axs = axs.ravel()

        for k, ax in zip(sorted(resultBean.metrics_stream.keys()), axs):

            x = resultBean.metrics_stream[k]

            ax.plot(x,linewidth=0.8)
            ax.set_title(f'Hitrate@{k}')

            for d in resultBean.drift_points:
                ax.axvline(d, color='red',linewidth=0.8,alpha=0.7)
            for w in resultBean.warning_points:
                ax.axvline(w, color='orange',linewidth=0.8,alpha=0.4)

        plt.xlabel('batches')
        fig.savefig(os.path.join(folder, 'test_hitrate.pdf'))
        plt.close(fig)


# # Automaton
# 
# 


def ma(l,window_size):
  numbers_series = pd.Series(l)
  windows = numbers_series.rolling(window_size)
  moving_averages = windows.mean()

  moving_averages_list = moving_averages.tolist()
  without_nans = moving_averages_list[window_size - 1:]

  return without_nans


# ## Train model procedure


def train_with_batch(resultBean, model,old_model,users,pos_items,neg_items,optimizer,
                     loss_fn, mask_delta, epochs=epochs, train_batch_size=1024): 
  model.train()

  idx_train_dataset = list(range(users.shape[0]))

  for epoch in range(epochs):

    for i in range(0, len(idx_train_dataset), train_batch_size):
      mask_delta_batch = mask_delta[i: i+train_batch_size] if mask_delta is not None else None

      optimizer.zero_grad()

      # Predict and calculate loss
      pos_scores, neg_scores = model(users[i: i+train_batch_size], 
                                     pos_items[i: i+train_batch_size], 
                                     neg_items[i: i+train_batch_size])
      loss, Lbpr, loss_sim, Lreg = loss_fn(users[i: i+train_batch_size], 
                     pos_items[i: i+train_batch_size], 
                     neg_items[i: i+train_batch_size], 
                     pos_scores, 
                     neg_scores, 
                     model, old_model, mask_delta_batch)
      
      # Backpropagate
      loss.backward()

      resultBean.train_batch_losses.append(loss.item())
      resultBean.train_losses_bpr.append(Lbpr.item())
      resultBean.train_losses_loss_sim.append(loss_sim.item())
      resultBean.train_losses_reg.append(Lreg.item())

      # Update the parameters
      optimizer.step()


def train(resultBean, model,old_model,users,pos_items,neg_items,optimizer,loss_fn,mask_delta,epochs=epochs):
  train_with_batch(resultBean, model,old_model,users,pos_items,neg_items,optimizer,
                     loss_fn, mask_delta, epochs=epochs, train_batch_size=64)
  if True:
      return
    
  model.train()

  #print(f'Train on {users.shape}, {pos_items.shape}, {neg_items.shape}')

  for epoch in range(epochs):

    optimizer.zero_grad()

    # Predict and calculate loss
    pos_scores, neg_scores = model(users, pos_items, neg_items)
    loss, Lbpr, loss_sim, Lreg = loss_fn(users, 
                  pos_items, 
                  neg_items, 
                  pos_scores, 
                  neg_scores, 
                  model, old_model, mask_delta)
    
    # Backpropagate
    loss.backward()

    if np.isnan(Lbpr.item()):
      raise RuntimeError(f'loss bpr is nan at epoch', epoch)

    resultBean.train_batch_losses.append(loss.item())
    resultBean.train_losses_bpr.append(Lbpr.item())
    resultBean.train_losses_loss_sim.append(loss_sim.item())
    resultBean.train_losses_reg.append(Lreg.item())

    # Update the parameters
    optimizer.step()


# ## Sampling procedure
# 
# We train the model on the current batch and a sample representing the entire history of the stream.
# 
# The procedure updates some variable while processing each batch to store:
#     
# 1. the number of times a user is seen by the model. It is quivalent to the number of pairs of the user *u* (A)
# 2. the positive items processed by the model at $batch_i$
# 3. the number of users in each batch
# 
# **To build a sample**
# 
# 1. Sampling users according to its frequencies (A)
# 2. Compute the number of positives *k* to generate for each user related to the *importance* of the user in the batch. More times a user is seen in the stream by the model more probably the user is sampled in the current batch.
# 3. Generate the triplet (u, p, n) for each user u sampled at step 1, sampling $p_1, ..., p_k$ positives with *k* defined at step 2, and generate random negatives for each positives *p*. The positive are sampled according to the score predicted by the model


from operator import pos
# PARAM: negative_sampling_processes
#from scipy.special import softmax


class DataStreamSampler:
  def __init__(self, batch_size, debug=False):
    self._batch_size = batch_size
    self._users_per_batch = []
    self._latent_sizes = []
    self._positive_items_per_user = defaultdict(set)  # dict user -> set positives
    self._items_max_id = 0
    self._user_preferences = defaultdict(lambda: 0)
    self._user_preferences_per_regime = defaultdict(lambda: defaultdict(list))
    self._current_regime = 0
    self._debug = debug
    
  def start_new_regime(self):
    self._current_regime += 1
    
  def get_user_regime_data(self):
    return self._user_preferences_per_regime
    
  def get_new_latent_size(self, default_value=50):
    """
    Return the new value of latent size or the default value
    """
    
    if len(self._latent_sizes) > 0:
        k = self._latent_sizes[-1] + 50
    else:
        k = default_value
    
    self._latent_sizes.append(k)
    return k
    
  def get_registered_batch(self):
    return len(self._users_per_batch)

  def __str__(self):
    return f'BS = {self._batch_size}, ' \
           f'#batch = {len(self._users_per_batch)}, ' \
           f'#users = {len(self._user_preferences)}'

  def update(self, users_tensor, pos_items_tensor, items_max_id):
    self._items_max_id = max(items_max_id, pos_items_tensor.max().item(), self._items_max_id)

    user_item_in_batch = set()
    user_in_batch = set()
    for i, u in enumerate(users_tensor):
      self._user_preferences[u.item()] += 1
      self._positive_items_per_user[u.item()].add(pos_items_tensor[i].item())
      user_in_batch.add(u.item())
      
      pair = (u.item(), pos_items_tensor[i].item())
      if pair not in user_item_in_batch:
        self._user_preferences_per_regime[self._current_regime][pair[0]].append(pair[1])
        user_item_in_batch.add(pair)

    self._users_per_batch.append(len(user_in_batch))

  def get_kwnown_item_mask(self, pos_items):
    """
    Generate a 1-D mask for already seen items. The mask has 1 when the item
    in the corresponding position in pos_items is already seen by the model
    """
    mask_delta = torch.zeros_like(pos_items, device=pos_items.device)

    if self._positive_items_per_user:
      all_pos = set(itertools.chain.from_iterable(self._positive_items_per_user.values()))
      for i, p in enumerate(pos_items):
        if p.item() in all_pos:
          mask_delta[i] = 1

    return mask_delta

  def generate_batch(self, model, device):
    d_users = list(sorted(self._user_preferences.keys()))
    #d_freq = softmax([self._user_preferences[u] for u in self._user_preferences])
    d_freq = [self._user_preferences[u] for u in d_users]
    d_freq = np.array(d_freq) / np.sum(d_freq)
    
    if self._debug:
      print(f'Sample k={int(np.mean(self._users_per_batch))} users')
    distinct_users = np.random.choice(d_users, 
                                      size=int(np.mean(self._users_per_batch)), 
                                      replace=False, 
                                      p=d_freq)

    if self._debug:
      print(f'Generate data for {len(distinct_users)} users')

    #total_pairs = np.sum([self._user_preferences[u] for u in distinct_users])
    #(self._batch_size * d_freq[u] / total_pairs) \
    total_users_weight = np.sum([self._user_preferences[u] for u in distinct_users])
    relative_batch_size = self._batch_size / negative_sampling_processes
    if self._debug:
      print('total_users_weight', total_users_weight)
    pairs_per_user = [math.ceil(
        relative_batch_size * (self._user_preferences[u] / total_users_weight) )  \
        for u in distinct_users]
    
    assert np.sum(np.array(pairs_per_user) == 0) == 0, f'no items for users in {pairs_per_user}'
    if self._debug:
      print(f'#pairs_per_user = {len(pairs_per_user)}. ' +
          f'Total positives in current batch {sum(pairs_per_user)} '+
          f'resulting in BS = {sum(pairs_per_user) * negative_sampling_processes}')

    # sampling negatives from unseen items (for each users)
    negatives_id_per_user = {}
    for u in distinct_users:
        if u in self._positive_items_per_user:
            negatives_id_per_user[u] = list(set(range(max(self._positive_items_per_user[u])+1)) - self._positive_items_per_user[u])
            if len(negatives_id_per_user[u]) == 0:
              if self._debug:
                print(f'no negatives generated for user {u}')
                
        if u not in negatives_id_per_user or not negatives_id_per_user[u]:
            if self._debug:
              print('WARNING: unknown user', u)
            negatives_id_per_user[u] = list(range(self._items_max_id + 1))
        
    if self._debug:
      print('negatives ids length:', [len(negatives_id_per_user[u]) for u in negatives_id_per_user])
    
    t_items = torch.tensor(range(self._items_max_id + 1), 
                           device=device, dtype=torch.int)

    t_distinct_users = torch.tensor(distinct_users,
                                    device=device, dtype=torch.int).view(-1, 1)
    
    #batch_size_all_positives = 10    
    batch_size_all_positives = 3
    t_items = t_items.view(1, -1).repeat(batch_size_all_positives, 1)
    if self._debug:
      print(f't_distinct_users = {t_distinct_users.shape}, t_items = {t_items.shape}')

    pos_per_user = []
    batch = []    
    for batch_i in range(0, t_distinct_users.shape[0], batch_size_all_positives):
        b_i, b_j = batch_i, batch_i + batch_size_all_positives

        t_u = t_distinct_users[b_i:b_j]
        t_p = t_items[:t_u.shape[0]]
        pos_score = model(t_u, t_p)
        if self._debug:
          print(f'pos score shape is {pos_score.shape}')

        for i, u, num_pos in zip(range(distinct_users[b_i:b_j].shape[0]), 
                                 distinct_users[b_i:b_j], 
                                 pairs_per_user[b_i:b_j]):
          #print(u, num_pos * negative_sampling_processes)

          k = min(num_pos, pos_score.shape[1])
          _, item_idx = torch.topk(pos_score[i], k=k)
          pos_per_user.append((u, item_idx.shape[0]))

          for p in item_idx:
            batch.append([u, t_items[0, p]])
    
    pairs = torch.tensor(batch, device=device, dtype=torch.int)
    #print(f'pairs = {pairs.shape}')
    pairs = pairs.repeat(1, negative_sampling_processes).view(-1, 2)
    #print(f'pairs2 = {pairs.shape}')

    neg = []
    for u, num_pos in pos_per_user:
        neg.extend(np.random.choice(negatives_id_per_user[u], 
                                    num_pos*negative_sampling_processes)
                   .tolist())
    t_neg = torch.tensor(neg, device=device, dtype=torch.int)
    if self._debug:
      print(f't_neg = {t_neg.shape}')
    t_batch = torch.cat((pairs, t_neg.view(-1, 1)), dim=-1)
    if self._debug:
      print(f't_batch = {t_batch.shape}')

    return t_batch.long()


# ONLY for debug
# user_sampler = DataStreamSampler(dataLoader.batch_size, debug=True)
# a = torch.abs(torch.randn(dataLoader.batch_size, 1)* 10).long()
# b = torch.abs(torch.randn(dataLoader.batch_size, 1)* 10).long()
# user_sampler.update(a, b, b.max().item())
# print(user_sampler)
# batch = None

# a_mask = user_sampler.get_kwnown_item_mask(a)
# print(a_mask.shape, a_mask.sum())

# batch = user_sampler.generate_batch(resultBean.model_list[-1], device)

# if batch is not None:
  # print(batch.shape)
# print(batch[-24:, :].tolist())

# a= None 
# b = None 
# batch = None
# user_sampler = None


# ## Train procedure def.


@dataclasses.dataclass
class ResultDataClass:
  model_list: list = dataclasses.field(default_factory=list)
  bpr_loss_list: list = dataclasses.field(default_factory=list)
  drift_points: list = dataclasses.field(default_factory=list)
  warning_points: list = dataclasses.field(default_factory=list)
  testset_loss_list: list = dataclasses.field(default_factory=list)
  train_batch_losses: list = dataclasses.field(default_factory=list)
  train_losses_bpr: list = dataclasses.field(default_factory=list)
  train_losses_loss_sim: list = dataclasses.field(default_factory=list)
  train_losses_reg: list = dataclasses.field(default_factory=list)
  latent_sizes: list = dataclasses.field(default_factory=list)
  loss_detailed: dict = dataclasses.field(default_factory=lambda: defaultdict(list))
  metrics_stream: dict = dataclasses.field(default_factory=lambda: defaultdict(list))
  user_regimes: dict = None


def evalutate_batch(resultBean, user_recorder, model, 
                    users_test, pos_items_test, 
                    top_k=[5, 10, 100]):
  if not ENABLE_HITRATE_STAT:
    return
  t_user, t_pos, t_neg  = user_recorder.sample(users_test, pos_items_test)

  acc = MetricAccumulator()
    
  x = torch.cat((t_pos.view(-1, 1), t_neg), -1)
  t_score = model(t_user.view(-1, 1), x)

  for k in top_k:
    acc.compute_metric(t_user, t_pos, t_neg, t_score, top_k=k)

  result = acc.get_metrics()
  for k in result:
    resultBean.metrics_stream[k].append(result[k]['hitrate'])

    
def train_automaton(resultBean, dataLoader, device, loss_alpha_param, loss_beta_param):
  user_sampler = DataStreamSampler(int(dataLoader.batch_size * .7))  # as train
  user_recorder = UserHistoryRecorder()

  cache=[]

  #K=int(math.sqrt(dataLoader.N)/2)
  K = user_sampler.get_new_latent_size()
  print('Latent K', K)
  model = BPR_MatrixFactorization(dataLoader.M,dataLoader.N, K)
  model = model.to(device)
  old_model=None

  optimizer = torch.optim.Adam(model.parameters(), lr=lr)

  n_regime_batches=0
  n_batches=0

  status='training'

  warning=False
  drift=False

  drift_detector=HDDM_W()  

  regime = 0

  bpr_loss_fn = BPRLoss()
  #bpr_loss_fn = BPRPosNegLoss()  
  dg_bpr_loss_fn = DG_BPRLoss(loss_alpha_param, loss_beta_param)

  n_user_samples = np.zeros(dataLoader.M)
  n_item_samples = np.zeros(dataLoader.N)

  user_processing_threshold = 10
  item_processing_threshold = 0

  prev_batch = None
  prev_bpr_loss = None

  number_of_batch = dataLoader.number_of_samples // dataLoader.batch_size + 1

  viewed_items = set()
    
  # START TRAIN
  n_nan_batches = 0
  try:
    for idx_number, batch in tqdm(enumerate(dataLoader.iter()), total=number_of_batch):
      if idx_number > 2:
        continue
      model.eval()
      users = torch.LongTensor(batch[0]).to(device)
      pos_items = torch.LongTensor(batch[1]).to(device)
      neg_items = torch.LongTensor(batch[2]).to(device)
        
      user_recorder.update(users, pos_items, max(pos_items.max().item(), neg_items.max().item()))

      np_users = batch[0].astype(int)
      np_pos_items = batch[1].astype(int)
      np_neg_items = batch[2].astype(int)

      users_train, users_test, \
      pos_items_train, pos_items_test, \
      neg_items_train, neg_items_test = train_test_split(
          users, pos_items, neg_items,
          test_size=.3
      )
    
      # PREQUENTIAL PROTOCOL FIX
      #users_train = users; users_test = users
      #pos_items_train = pos_items; pos_items_test = pos_items
      #neg_items_train = neg_items; neg_items_test = neg_items

      # Computing losses =======================================================
      model.eval()
      pos_scores, neg_scores = model(users_test, pos_items_test, neg_items_test)

      bpr_loss = bpr_loss_fn(pos_scores, neg_scores)
    
      ## TEST loss on old and new items
      idx_viewed_items = torch.tensor([pos_items_test[i].item() in viewed_items
                or neg_items_test[i].item() in viewed_items 
                for i in range(pos_items_test.shape[0])])

      for i in torch.cat((pos_items_test, neg_items_test)).flatten():
        viewed_items.add(i.item())

      resultBean.loss_detailed['old'].append(bpr_loss_fn(pos_scores[idx_viewed_items], neg_scores[idx_viewed_items]).item())
      resultBean.loss_detailed['new'].append(bpr_loss_fn(pos_scores[~idx_viewed_items], neg_scores[~idx_viewed_items]).item())

      if not np.isnan(bpr_loss.item()):
        resultBean.bpr_loss_list.append(bpr_loss.item())
        drift_detector.update(bpr_loss.item())
        in_drift, in_warning = drift_detector.drift_detected, drift_detector.warning_detected
      else:
        in_drift, in_warning = False, False
        n_nan_batches += 1

      # ==========================================================================
      
      # copy the model before a warning
      #if status == 'warning' and old_model is None:
      #  old_model = copy.deepcopy(model)
      #  print('caching old model at batch ', idx_number)

      # TRAIN ON BATCH
      n_train_epoch = epochs
      
      # build a training set as sample U *_train
      if ENABLE_SAMPLING and user_sampler.get_registered_batch() > 0:
        model.eval()    
        sample_batch = user_sampler.generate_batch(model, device)
        prev_batch = (sample_batch[:, 0], sample_batch[:, 1], sample_batch[:, 2])
        #print(f'u = {prev_batch[0].shape}, {users_train.shape}. pos = {prev_batch[1].shape}, {pos_items_train.shape}. neg = {prev_batch[1].shape}, {neg_items_train.shape}')

        a = torch.cat((prev_batch[0], users_train))
        b = torch.cat((prev_batch[1], pos_items_train))
        c = torch.cat((prev_batch[2], neg_items_train))
      else:
        a = users_train
        b = pos_items_train 
        c = neg_items_train

      if not (in_drift and status == 'warning'):
        train(resultBean, model, None, a,b,c, optimizer, dg_bpr_loss_fn, None, epochs=n_train_epoch)
    
        # compute loss on test set after train
        model.eval()
        evalutate_batch(resultBean, user_recorder, model, users_test, pos_items_test)
        
        pos_scores_test, neg_scores_test = model(users_test, pos_items_test, neg_items_test)
        bpr_loss_test_set = bpr_loss_fn(pos_scores_test, neg_scores_test)
        resultBean.testset_loss_list.append(bpr_loss_test_set.item())
      
      user_sampler.update(users_train, pos_items_train, neg_items_train.max().item())

      # DETECT STREAM STATE
      if status=='training':

        if in_warning: # and not drift_detector.change_detected:
          resultBean.warning_points.append(n_batches)
          status='warning'
          print(status + ' @ batch=',n_batches)

      elif status=='warning':
        print('IN WARNING PHASE')
        cache.append((users_train, pos_items_train, neg_items_train)) # store current batch in cache
        
        if not in_warning and not in_drift:
          cache.clear()
          print('False warning...')
          status='training'
          print(status + ' @ batch =',n_batches)
          old_model = None

        elif in_drift:
          resultBean.drift_points.append(n_batches)

          print('drift @ batch=',n_batches)
          print('> Adding a regime model to the list @',n_batches)
          old_model = copy.deepcopy(model.cpu())
          resultBean.model_list.append(old_model)
          user_sampler.start_new_regime()

          regime+=1
            
          newK = user_sampler.get_new_latent_size()
          if newK != old_model.latent_size:
            old_model = resize_matrices(copy.deepcopy(model.cpu()), newK)
          model = BPR_MatrixFactorization(dataLoader.M, dataLoader.N, newK, old_model)
          model = model.to(device)

          optimizer = torch.optim.Adam(model.parameters(), lr=lr)

          # Use cache to train model
          print('> Training a new regime model with batches collected in the warning phase...')
          print('> Batches collected in the warning phase:', len(cache))
        
          old_model = old_model.to(device)
            
          # viewed item as mask
          mask_delta = user_sampler.get_kwnown_item_mask(pos_items_train)
          mask_delta = mask_delta.repeat(1, model.latent_size).view(-1, model.latent_size)

          a = users_train
          b = pos_items_train 
          c = neg_items_train
          train(model, old_model, a,b,c, optimizer, dg_bpr_loss_fn, mask_delta, epochs=n_train_epoch)
    
          # compute loss on test set after train
          model.eval()
          evalutate_batch(user_recorder, model, users_test, pos_items_test)
          
          pos_scores_test, neg_scores_test = model(users_test, pos_items_test, neg_items_test)
          bpr_loss_test_set = bpr_loss_fn(pos_scores_test, neg_scores_test)
          resultBean.testset_loss_list.append(bpr_loss_test_set.item())

          n_regime_batches=0

          #for cache_users, cache_pos_items, cache_neg_items in cache:
          #  train(model, old_model, cache_users, cache_pos_items, cache_neg_items, optimizer, dg_bpr_loss_fn)
          #  n_regime_batches+=1
          
          cache.clear()
          status='training'
          print(status + ' @ batch=',n_batches)
      
      if n_batches % (number_of_batch // 10) == 0:
        print('\n' + '='*80)
        print('n_batches:',n_batches,'ratings:',n_batches*batch_size,'/', dataLoader.number_of_samples)
        print(user_sampler)
        
      if n_batches > 0 and n_batches % 20 == 0:
        save_charts(f'partial_result_at_{n_batches}', windows_length=5)

      n_regime_batches+=1
      n_batches+=1

    # end for
    resultBean.latent_sizes = list(user_sampler._latent_sizes)
    print('Done!')
  except KeyboardInterrupt:
    print('Interrupted')

  except Exception:
    traceback.print_exc()

  print(f'\n\nn_nan_batches = {n_nan_batches}')
  print(user_sampler)
  resultBean.user_regimes = user_sampler.get_user_regime_data()
  resultBean.model_list.append(model)


# # Train

# ALPHA = .005
# BETA = .2
# resultBean = ResultDataClass()
# train_automaton(dataLoader,device, ALPHA, BETA)

# print(f'Sampling: {ENABLE_SAMPLING}, Hitrate: {ENABLE_HITRATE_STAT}')
# print(f'models: {len(resultBean.model_list)}')
# print(f'bpr loss items: {len(resultBean.bpr_loss_list)}')
# print(f'train batch losses: {len(resultBean.train_batch_losses)}')
# print(f'drift points: {len(resultBean.drift_points)}')
# print(f'warning points: {len(resultBean.warning_points)}')


# ## Loss on old (blue) and new (red) items

def plot_items(resultBean):
    plt.rcParams['figure.figsize'] = 20, 8
    plt.xlabel('batches')
    plt.ylabel('bpr loss')
    #plt.yscale('log') 
    plt.plot(resultBean.loss_detailed['old'],linewidth=0.8, color='b')
    plt.plot(resultBean.loss_detailed['new'],linewidth=0.8, color='r')


    for d in resultBean.drift_points:
        plt.axvline(d, color='red',linewidth=0.8,alpha=0.7)
    for w in resultBean.warning_points:
        plt.axvline(w, color='orange',linewidth=0.8,alpha=0.4)
        
    plt.savefig('train_loss_components.pdf')
    plt.show()


# ## Latent sizes

def plot_latent_size(resultBean):
    y = resultBean.latent_sizes
    x = list([str(i) for i in range(1, len(y) + 1)])

    plt.bar(x, y)
    plt.savefig('latent_sizes.pdf')
    plt.show()


# ## Plot BPR loss windowed
# 
# Computed on a mobile window

def plot_train_loss(resultBean):
    windows_length = 5

    plt.rcParams['figure.figsize'] = 20, 8
    plt.xlabel('batches')
    plt.ylabel('bpr loss')
    #plt.yscale('log') 
    plt.plot(ma(resultBean.bpr_loss_list, windows_length),linewidth=0.8)
    for d in resultBean.drift_points:
        plt.axvline(d, color='red',linewidth=0.8,alpha=0.7)
    for w in resultBean.warning_points:
        plt.axvline(w, color='orange',linewidth=0.8,alpha=0.4)
        
    plt.savefig('train_loss_windowed.pdf')
    plt.show()    


# ## Plot BPR loss per batch
# 
# Row value for each batch

# In[ ]:

def plot_train_loss_batch(resultBean):
    plt.rcParams['figure.figsize'] = 20, 8
    plt.xlabel('batches')
    plt.ylabel('bpr loss')
    #plt.yscale('log') 
    plt.plot(resultBean.bpr_loss_list,linewidth=0.8)
    for d in resultBean.drift_points:
        plt.axvline(d, color='red',linewidth=0.8,alpha=0.7)
    for w in resultBean.warning_points:
        plt.axvline(w, color='orange',linewidth=0.8,alpha=0.4)

    plt.savefig('train_loss_per_batch.pdf')
    plt.show()


# ## Test set - Plot BPR loss on windowed
# 
# Computed on a mobile window

def plot_loss(resultBean):
    windows_length = 5

    plt.rcParams['figure.figsize'] = 20, 10
    plt.xlabel('Batches')
    plt.ylabel('BPR Loss')
    #plt.yscale('log') 
    plt.plot(ma(resultBean.testset_loss_list, windows_length),linewidth=0.8)
    for d in resultBean.drift_points:
        plt.axvline(d, color='red',linewidth=0.8,alpha=0.7)
    for w in resultBean.warning_points:
        plt.axvline(w, color='orange',linewidth=0.8,alpha=0.4)

    plt.savefig('test_loss_windowed.pdf')
    plt.show()    


# ## Test set - Plot BPR loss per batch
# 
# Row value for each batch

def plot_loss_batch(resultBean):
    plt.rcParams['figure.figsize'] = 20, 8
    plt.xlabel('Batches')
    plt.ylabel('BPR Loss')
    #plt.yscale('log') 
    plt.plot(resultBean.testset_loss_list,linewidth=0.8)
    for d in resultBean.drift_points:
        plt.axvline(d, color='red',linewidth=0.8,alpha=0.7)
    for w in resultBean.warning_points:
        plt.axvline(w, color='orange',linewidth=0.8,alpha=0.4)
        
    plt.savefig('test_loss_per_batch.pdf')
    plt.show()


# ## Test set - Hitrate

def plot_hitrate(resultBean):
    if ENABLE_HITRATE_STAT:

        fig, axs = plt.subplots(len(resultBean.metrics_stream), 1, figsize=(50, 50))

        axs = axs.ravel()

        for k, ax in zip(sorted(resultBean.metrics_stream.keys()), axs):

            x = resultBean.metrics_stream[k]

            ax.plot(x,linewidth=0.8)
            ax.set_title(f'Hitrate@{k}')

            for d in resultBean.drift_points:
                ax.axvline(d, color='red',linewidth=0.8,alpha=0.7)
            for w in resultBean.warning_points:
                ax.axvline(w, color='orange',linewidth=0.8,alpha=0.4)

        plt.xlabel('batches')
        plt.savefig('test_hitrate.pdf')
        plt.show()

        plt.rcParams['figure.figsize'] = 20, 10
        # single charts
        for k in sorted(resultBean.metrics_stream.keys()):

            fig = plt.figure()
            x = resultBean.metrics_stream[k]

            plt.plot(x,linewidth=0.8)

            for d in resultBean.drift_points:
                plt.axvline(d, color='red',linewidth=0.8,alpha=0.7)
            for w in resultBean.warning_points:
                plt.axvline(w, color='orange',linewidth=0.8,alpha=0.4)

            plt.xlabel('Batches')
            plt.savefig(f'test_hitrate_{k}.pdf')
            plt.close(fig)



    if ENABLE_HITRATE_STAT:
      total_warnings = np.zeros(len(resultBean.bpr_loss_list))
      total_drift = np.zeros(len(resultBean.bpr_loss_list))
      
      if resultBean.warning_points:
        total_warnings[resultBean.warning_points] = 1
      if resultBean.drift_points:
        total_drift[resultBean.drift_points] = 1
        
      for k in resultBean.metrics_stream:
          res = resultBean.metrics_stream[k]
          if len(res) != total_drift.shape[0]:
            res2 = np.zeros(len(resultBean.bpr_loss_list))
            i = min(len(res), res2.shape[0]) + 1
            print('missing result. shapes are ', len(res), res2.shape[0])
            res2[:i] = res[:i-1]
            res = res2
          hitrate_export_pd = pd.DataFrame({'hirate': res, 
                                            'warning': total_warnings,
                                            'drift': total_drift})
          hitrate_export_pd.to_csv(f'result_hitrate_{k}.csv', index=False)


# ## Plot train loss while training
# 
# 

def other_plots(resultBean, dataLoader):
    #plt.rcParams['figure.figsize'] = 20, 8
    fig = plt.figure(figsize=(100, 20))

    plt.xlabel('batches * epochs')
    plt.ylabel('bpr loss')
    #plt.yscale('log') 
    plt.plot(resultBean.train_batch_losses,linewidth=0.8)


    d_fixed = np.array(resultBean.drift_points) * epochs
    w_fixed = np.array(resultBean.warning_points) * epochs

    for d in d_fixed:
        plt.axvline(d, color='red',linewidth=0.8,alpha=0.7)
    for w in w_fixed:
        plt.axvline(w, color='orange',linewidth=0.8,alpha=0.4)
        
    plt.savefig('train_model_losses.pdf')
    plt.show()


    # ### Component 1 - Loss BPR

    # In[ ]:


    print(f'Loss values range: [{min(resultBean.train_losses_bpr)}, {max(resultBean.train_losses_bpr)}]')

    fig = plt.figure(figsize=(100, 20))

    plt.xlabel('batches * epochs')
    plt.ylabel('bpr loss')
    #plt.yscale('log') 
    plt.plot(resultBean.train_losses_bpr,linewidth=0.8)


    d_fixed = np.array(resultBean.drift_points) * epochs
    w_fixed = np.array(resultBean.warning_points) * epochs

    for d in d_fixed:
        plt.axvline(d, color='red',linewidth=0.8,alpha=0.7)
    for w in w_fixed:
        plt.axvline(w, color='orange',linewidth=0.8,alpha=0.4)
        
    plt.savefig('train_model_losses_bpr.pdf')
    plt.show()


    # ### Component 2 - Loss L^sim

    # In[ ]:


    print(f'Loss values range: [{min(resultBean.train_losses_loss_sim)}, {max(resultBean.train_losses_loss_sim)}]')
    fig = plt.figure(figsize=(100, 20))

    plt.xlabel('batches * epochs')
    plt.ylabel('L-sim loss')
    #plt.yscale('log') 
    plt.plot(resultBean.train_losses_loss_sim,linewidth=0.8)


    d_fixed = np.array(resultBean.drift_points) * epochs
    w_fixed = np.array(resultBean.warning_points) * epochs

    for d in d_fixed:
        plt.axvline(d, color='red',linewidth=0.8,alpha=0.7)
    for w in w_fixed:
        plt.axvline(w, color='orange',linewidth=0.8,alpha=0.4)
        
    plt.savefig('train_model_losses_Lsim.pdf')
    plt.show()


    # ### Component 3 - Loss regularization

    # In[ ]:


    print(f'Loss values range: [{min(resultBean.train_losses_reg)}, {max(resultBean.train_losses_reg)}]')

    fig = plt.figure(figsize=(100, 20))

    plt.xlabel('batches * epochs')
    plt.ylabel('reg. loss')
    #plt.yscale('log') 
    plt.plot(resultBean.train_losses_reg,linewidth=0.8)


    d_fixed = np.array(resultBean.drift_points) * epochs
    w_fixed = np.array(resultBean.warning_points) * epochs

    for d in d_fixed:
        plt.axvline(d, color='red',linewidth=0.8,alpha=0.7)
    for w in w_fixed:
        plt.axvline(w, color='orange',linewidth=0.8,alpha=0.4)
        
    plt.savefig('train_model_losses_reg.pdf')
    plt.show()


    # ### When items appear to the model

    # In[ ]:


    # both positive and negative items are counted
    items_viewed = set()

    items_in_data_stream = []
    perc_new_item_per_batch = []
    history_perc_new_item = []

    for batch in dataLoader.iter():

      #users = batch[0]
      pos_items = batch[1]
      neg_items = batch[2]
     
      cnt_distinct = set()
      cnt = 0
      for i in np.concatenate((pos_items, neg_items)):
        if i not in items_viewed:
          cnt += 1
          items_viewed.add(i)
        cnt_distinct.add(i)

      items_in_data_stream.append(cnt)
      perc_new_item_per_batch.append(len(cnt_distinct))
      history_perc_new_item.append(len(items_viewed))


    print(f'number of items {len(items_viewed)} (== {sum(items_in_data_stream)})')
    mean_new_items_per_batch = np.mean(items_in_data_stream)

    perc_new_item_per_batch = np.array(items_in_data_stream) / np.array(perc_new_item_per_batch)

    history_perc_new_item = np.array(history_perc_new_item) / len(items_viewed)

    del items_viewed
    fig, ax1 = plt.subplots(figsize=(100, 20))

    plt.xlabel('batches')
    plt.ylabel('#new items')
    #plt.yscale('log') 
    #plt.scatter(list(range(len(items_in_data_stream))), items_in_data_stream)
    ax1.plot(items_in_data_stream,linewidth=0.8)

    ax1.axhline(y=mean_new_items_per_batch, color='violet', linestyle='--')

    #for d in drift_points:
    #    plt.axvline(d, color='red',linewidth=0.8,alpha=0.7)
    #for w in warning_points:
    #    plt.axvline(w, color='orange',linewidth=0.8,alpha=0.4)

    ax2 = ax1.twinx()
    ax2.plot(perc_new_item_per_batch,linewidth=0.8, color='r', label='perc new items')
    ax2.plot(history_perc_new_item,linewidth=0.8, color='y', label='perc total items viewed')
    ax2.set_ylim((0, 1))
        
    print(f'mean_new_items_per_batch = {mean_new_items_per_batch}')
    plt.savefig('when_items_appear.pdf')
    plt.show()


# # Save models

# In[ ]:

def save_models(resultBean, dataLoader):
    if resultBean.model_list:
        import shutil
        print(f'Save {len(resultBean.model_list)} models on disk')
        
        model_out_dir = './trained_models'
        if os.path.exists(model_out_dir):
            shutil.rmtree(model_out_dir)
            
        os.mkdir(model_out_dir)
        
        for i, m in tqdm(enumerate(resultBean.model_list)):
            torch.save(m.state_dict(), os.path.join(model_out_dir, f'model_{i:03}.pth'))
    else:
        print('no model to save')


    # print(len(resultBean.model_list), '\n',
    # len(resultBean.bpr_loss_list), '\n',
    # len(resultBean.drift_points), '\n',
    # len(resultBean.warning_points), '\n',
    # len(resultBean.testset_loss_list), '\n',
    # len(resultBean.train_batch_losses), '\n',
    # len(resultBean.train_losses_bpr), '\n',
    # len(resultBean.train_losses_loss_sim), '\n',
    # len(resultBean.train_losses_reg), '\n',
    # len(resultBean.latent_sizes), '\n',
    # len(resultBean.loss_detailed), '\n',
    # len(resultBean.metrics_stream))


# In[ ]:


def save_results(resultBean, dataLoader):
    
    resultBeanToSave = {f.name: getattr(resultBean, f.name) for f in dataclasses.fields(resultBean) if f.name != 'model_list'}

    resultBeanToSave['num_users'] = dataLoader.M
    resultBeanToSave['num_items'] = dataLoader.N

    with open('last_result.json', 'w') as fp:
        json.dump(resultBeanToSave, fp, cls=NpEncoder)
        print('results raw data saved')




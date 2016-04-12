--[[
This class implements a generative model of concept mentions, which factorizes
as p(concepts)p(mentions|concepts), as well as functions to maximize a lower
bound on the likelihood of unlabeled mentions using an EM-like algorithm
]]--
local class = require 'class'
local hdf5 = require 'hdf5'
local nn = require 'nn'
local nngraph = require 'nngraph'
local rnn = require 'rnn'
local cutorch = require 'cutorch'
local cunn = require 'cunn'
local CSLM = require 'CSLM'
local ChainMRF = require 'ChainMRF'
local Emissions = require 'Emissions'

local SemiCE = class('SemiCE')


-- expand Narrow to just take offset as argument (without specifying length)
function nn.Narrow:updateOutput(input)
  local save = false
  if self.length <= 0 then
    save = self.length  
    self.length = input:size()[self.dimension] + self.length - self.index + 1
  end
  local output = input:narrow(self.dimension, self.index, self.length)
  self.output = self.output:typeAs(output)
  self.output:resizeAs(output):copy(output)
  if save then
    self.length = save
  end
  return self.output
end


function nn.Narrow:updateGradInput(input, gradOutput)
  local save = false
  if self.length <= 0 then
    save = self.length  
    self.length = input:size()[self.dimension] + self.length - self.index + 1
  end
  self.gradInput = self.gradInput:typeAs(input)
  self.gradInput:resizeAs(input):zero()
  self.gradInput:narrow(self.dimension,self.index,self.length):copy(gradOutput)
  if save then
    self.length = save
  end
  return self.gradInput
end


-- fix a bug in the rnn implementation of padded lookup table
function nn.LookupTableMaskZero:accGradParameters(input, gradOutput, scale)
	nn.LookupTable.accGradParameters(self, torch.add(input, 1), gradOutput, scale)
end


-- Define a model class
function SemiCE:__init(config)
  self.conf = config or {}
  self.word_d = self.conf.word_dim or 100
  self.cui_d = self.conf.cui_dim or 100
  self.mrf_d = self.conf.mrf_dim or 0
  self.K = self.conf.num_neighbours or 2
  self.cuda = self.conf.cuda == 1
end


function SemiCE:save_to_file(filename)
  reco_params, gp = self.recognition:getParameters()
  if self.conf.emission_nn == 1 then
    emissions_param, gp = self.emissions.emissions_nn:getParameters()
  else
    emissions_param = false
  end
  U = semi.MRF.U
  W = semi.MRF.W
  torch.save(filename, 
             {reco_params = reco_params,
              emissions_param = emissions_param,
              U = U, W = W, config = self.conf})
end


function SemiCE:load_from_file(filename)
  -- TODO
end


function SemiCE:read_vocabs(filenames)
	local function read_lines(filename)
	  lines = {}
	  for line in io.lines(filename) do 
		lines[#lines + 1] = line
	  end
	  return lines
  end
  self.men_voc = read_lines(filenames.men_voc_file)
  self.word_voc = read_lines(filenames.word_voc_file)
  self.cui_voc = read_lines(filenames.cui_voc_file)
end


function SemiCE:read_data(filename)
  self.data = {}
  data_items = {"train_sentences", "dev_sentences", "unsup_sentences",
                "train_mentions", "dev_mentions", "unsup_mentions",
                "men_to_cui_supports", "men_to_words"}
  readFile = hdf5.open(filename, 'r')
  for i, item_name in ipairs(data_items) do
    dataset = readFile:read(item_name)
    self.data[item_name] = dataset:all()
  end
  -- transform -1 to 0 for padded LookupTable in mention to voc
  self.data.men_to_words = torch.cmax(self.data.men_to_words, 0)
  self.max_words = self.data.men_to_words:size()[2]
  -- transform -1 to <NONE> in support
  self.max_cuis = self.data.men_to_cui_supports:size()[2]
  support_filter = (torch.cmax(torch.cmin(self.data.men_to_cui_supports, 1), 0) - 1) * (- 1)
  support_filter:select(3, 2):fill(0)
  support_filter:cmul(torch.range(#self.cui_voc + 1,
                                  #self.cui_voc + 25):view(1, self.max_cuis,
                                                           1):expand(#self.men_voc,
                                                                     self.max_cuis, 2))
  self.data.men_to_cui_supports = torch.add(torch.cmax(self.data.men_to_cui_supports, 0), support_filter)
  self.data.men_to_cui_supports[1][1][1] = 1
  -- add bogus labels to unsup_sentences
  local unsup_length = self.data.unsup_sentences:size()[1]
  local my_ones = torch.ones(unsup_length, 1)
  self.data.unsup_sentences = self.data.unsup_sentences:view(unsup_length, 1):contiguous()
  self.data.unsup_sentences = torch.cat(self.data.unsup_sentences, my_ones, 2)
  -- put UMLS and training together for global training of the mention-to-cui
  self.data.all_train_mentions = nn.JoinTable(1):forward{self.data.train_mentions,
                                                         self.data.unsup_mentions}
  for cui = 1, self.max_cuis, 1 do
    self.cui_voc[#self.cui_voc + 1] = '<NONE>' .. cui
  end
end


function make_embedding(input_words_current, input_words_left, input_words_right,
                        word_embedding, word_embedding_context_left, word_embedding_context_right,
                        left_transfo, right_transfo, lin_layer)
  
  -- words_rep_current: batch_size x word_d
  local words_rep_current = nn.Sum(2)(
                              word_embedding(input_words_current))
  -- words_rep_left: batch_size x word_d
  local words_rep_left = nn.Tanh()(
                           left_transfo(
                             nn.Sum(2)(
                               word_embedding_context_left(input_words_left))))
  -- words_rep_right: batch_size x word_d
  local words_rep_right = nn.Tanh()(
                            right_transfo(
                              nn.Sum(2)(
                                word_embedding_context_right(input_words_right))))
  -- words_rep_neighbours: batch_size x word_d
  local words_rep_neighbours = nn.CAddTable()({words_rep_left,
                                               words_rep_right})
  
  -- full_embedding: batch_size x cui_d x 1
  local full_embedding = nn.Replicate(1, 3)(
                           nn.Tanh()(
                             lin_layer(
                               nn.JoinTable(2)({words_rep_current,
                                                words_rep_neighbours}))))
  
  return full_embedding
end


-- The recognition model gives a mean-field approximation to the posterior
-- distribution p(concepts|mentions)
function SemiCE:make_recognition()
  self.word_embedding = nn.LookupTableMaskZero(#self.word_voc, self.word_d)
  if self.cuda then
    self.word_embedding = self.word_embedding:cuda()
  end
  -- unsup
  self.word_embedding_context_left = self.word_embedding:clone('weight','bias', 'gradWeight', 'gradBias')
  self.word_embedding_context_right = self.word_embedding:clone('weight','bias', 'gradWeight', 'gradBias')
  self.left_transfo = nn.Linear(self.word_d, self.word_d)
  self.right_transfo = nn.Linear(self.word_d, self.word_d)
  self.lin_layer = nn.Linear(2 * self.word_d, self.cui_d)
  if self.cuda then
    self.left_transfo = self.left_transfo:cuda()
    self.right_transfo = self.right_transfo:cuda()
    self.lin_layer = self.lin_layer:cuda()
  end
  
  local input_mask = nn.Identity()()
  
  local input_words_current = nn.Identity()()
  local input_words_left = nn.Identity()()
  local input_words_right = nn.Identity()()
  
  -- full_embedding: batch_size x max_cuis x cui_d
  local full_embedding = make_embedding(input_words_current,
									    input_words_left,
									    input_words_right,
									    self.word_embedding,
									    self.word_embedding_context_left,
									    self.word_embedding_context_right,
									    self.left_transfo,
									    self.right_transfo,
									    self.lin_layer)
  
  -- cui_rep: batch_size x max_cuis x cui_d
  self.cui_embedding = nn.LookupTable(#self.cui_voc, self.cui_d)
  if self.cuda then
    self.cui_embedding = self.cui_embedding:cuda()
  end
  local input_support = nn.Identity()()
  local cui_rep = self.cui_embedding(input_support)
  
  -- scores and probas: batch_size x max_cuis
  local scores = nn.Sum(3)(nn.MM()({cui_rep, full_embedding}))
  local log_probas = nn.LogSoftMax()(
                       nn.CAddTable()(
                          {scores, input_mask, proximities}))
  
  self.recognition = nn.gModule({input_words_current,
                                 input_words_left,
                                 input_words_right,
                                 input_support,
                                 input_mask}, -- input_proxs
                                {log_probas})
  
  if self.cuda then
    self.recognition = self.recognition:cuda()
  end
end


function SemiCE:make_lower_bound()
  input_thetas = nn.Identity()()        -- (batch - k) x (k x max_cuis x max_cuis)
  input_log_probas = nn.Identity()()    -- batch x max_cuis
  input_emissions = nn.Identity()()     -- (batch - k) x max_cuis
  
  -- combine log_probas
  local conv_probas = nn.TemporalConvolution(self.max_cuis,
                                             self.K * self.max_cuis * self.max_cuis,
                                             self.K + 1, 1)
  conv_probas.bias:fill(0)
  conv_probas.weight:fill(0)
  my_weight = conv_probas.weight:view(self.K, self.max_cuis, self.max_cuis, self.max_cuis * (self.K + 1))
  for a = 1, self.max_cuis do
     my_weight:select(4, a):select(2, a):fill(1)
  end
  for k = 1, self.K do
      for b = 1, self.max_cuis do
          my_weight:select(4, k * 25 + b):select(3, b):select(1, k):fill(1)
      end
  end
  function conv_probas:accGradParameters(input, gradOutput, scale) end
  -- q_i q_j \theta_i_j
  local pair_probas = nn.Exp()(conv_probas(input_log_probas))
  local pre_mrf_bound = nn.Sum()(nn.Sum()(nn.CMulTable()({pair_probas, input_thetas})))
  local div_k = nn.Mul()
  div_k.weight[1] = 1 / (self.K + 1)
  local mrf_bound = div_k(pre_mrf_bound)
  -- q_i \log(p_i)
  local narrow_log_probas = nn.Narrow(1, 1, -self.K)(input_log_probas)
  local narrow_probas = nn.Exp()(narrow_log_probas)
  local emission_bound = nn.Sum()(nn.Sum()(
                            nn.CMulTable()(
                              {narrow_probas, nn.Log()(input_emissions)}
                            )
                         ))
  -- -q_i \log(q_i)
  local minus = nn.Mul()
  minus.weight[1] = -1
  local recog_entropy = minus(nn.Sum()(nn.Sum()(
                          nn.CMulTable()(
                            {narrow_probas, narrow_log_probas}
                          )
                        )))
  self.lower_bound = nn.gModule({input_thetas,
                                 input_log_probas,
                                 input_emissions},
                                {mrf_bound,
                                 emission_bound,
                                 recog_entropy,
                                 nn.Sum()(nn.Sum()(pair_probas))})
  if self.cuda then
    self.lower_bound = self.lower_bound:cuda()
  end
end


function SemiCE:make_generative()
  if self.mrf_d > 0 then
    self.MRF = self.MRF or CSLM(self.K, #self.cui_voc, self.mrf_d)
    self.MRF:normal()
  else
    self.MRF = self.MRF or ChainMRF(self.K, #self.cui_voc)
  end
  self.emissions = self.emissions or Emissions(self)
  self.emissions:fill_counts(1)
  self.emissions:normalize_cuis(self.conf.emission_add_count,
                                self.conf.emission_add_count_umls)
  if self.conf.emission_nn == 1 then
    self.emissions:make_emissions(50, 30, false, 16)
  end
end


function SemiCE:make_batch(begin, batch_size, data_set,
                           use_neighbours, use_sentence_boundaries,
                           drop_cui_less)
  local close
  local batch = {}
  batch.current_mentions = torch.Tensor(batch_size):fill(1)
  
  -- recognition model inputs
  batch.current_words = torch.Tensor(batch_size, self.max_words):fill(0)
  batch.current_words:select(2, 1):fill(1)
  batch.left_words = torch.Tensor(batch_size, self.K * self.max_words):fill(0)
  batch.left_words:select(2, 1):fill(1)
  batch.right_words = torch.Tensor(batch_size, self.K * self.max_words):fill(0)
  batch.right_words:select(2, 1):fill(1)
  
  -- sparsity and proximity features
  batch.mask = torch.Tensor(batch_size, self.max_cuis):fill(-1e12)
  batch.proximities = torch.Tensor(batch_size, self.max_cuis, 5):fill(0) -- TODO: generalize
  batch.supports = torch.Tensor(batch_size, self.max_cuis):fill(#self.cui_voc)
  for b = 1, batch_size, 1 do
    batch.supports[b]:copy(self.data.men_to_cui_supports[1]:select(2, 1))
  end
  batch.labels = torch.LongTensor(batch_size):fill(1)
  
  local sentences = self.data[data_set]
  local data_size = sentences:size()[1]
  local tok = begin
  local b = 1
  while b <= batch_size do
    batch.current_mentions[b] = sentences[tok][1]
    if sentences[tok][1] == 1 then
      if use_sentence_boundaries then
        for k = 1, self.K, 1 do
          if b <= batch_size then
            batch.mask[b][1] = 0
            batch.proximities[b][1] = 1
          end
          b = b + 1
        end
      end
      tok = tok % data_size + 1
    elseif drop_cui_less and sentences[tok][2] == 2 then
      tok = tok % data_size + 1
    else
      batch.supports[b]:copy(self.data.men_to_cui_supports[sentences[tok][1]]:select(2, 1))
      close = self.data.men_to_cui_supports[sentences[tok][1]]:select(2, 2)
      batch.mask[b]:copy(
          close:clone():apply(
              function(x) return (x > 0) and 0 or -1e12 end))
      batch.proximities[b]:select(2, 1):copy(
          close:clone():apply(
              function(x) return (x == 1) and 1 or 0 end))
      batch.proximities[b]:select(2, 2):copy(
          close:clone():apply(
              function(x) return (x >= 0.9) and 1 or 0 end))
      batch.proximities[b]:select(2, 3):copy(
          close:clone():apply(
              function(x) return (x >= 0.8) and 1 or 0 end))
      batch.proximities[b]:select(2, 4):copy(
          close:clone():apply(
              function(x) return (x >= 0.7) and 1 or 0 end))
      batch.proximities[b]:select(2, 5):copy(
          close:clone():apply(
              function(x) return (x >= 0.5) and 1 or 0 end))
      batch.labels[b] = -1
      for cui = 1, self.max_cuis, 1 do
        if batch.supports[b][cui] == sentences[tok][2] or
           (batch.supports[b][cui] == 2 and batch.labels[b] == -1) then
          batch.labels[b] = cui
        end
        -- HACK TO REMOVE CUI-less
        if batch.supports[b][cui] == 2 then
          batch.proximities[b][cui]:fill(0)
        end
      end
      batch.current_words[b]:copy(self.data.men_to_words[sentences[tok][1]])
      if use_neighbours then
        local sentence_bound = false
        local k = 1
        while k <= self.K do
          if tok - k <= 0 or sentences[tok - k][1] == 1 then
            sentence_bound = true
          end
          if sentence_bound then
            batch.left_words[b][(k - 1) * self.max_words + 1] = 1
          else
            batch.left_words[b]:sub((k - 1) * self.max_words + 1,
                                    k * self.max_words):copy(self.data.men_to_words[sentences[tok - k][1]])
          end
          k = k + 1
        end
        sentence_bound = false
        k = 1
        while k <= self.K do
          if tok + k >= data_size or sentences[tok + k][1] == 1 then
            sentence_bound = true
          end
          if sentence_bound then
            batch.right_words[b][(k - 1) * self.max_words + 1] = 1
          else
            batch.right_words[b]:sub((k - 1) * self.max_words + 1,
                                     k * self.max_words):copy(self.data.men_to_words[sentences[tok + k][1]])
          end
          k = k + 1
        end
      end
      b = b + 1
      tok = tok % data_size + 1
    end
  end
  batch.last_tok = tok
  
  if self.cuda then
    --  pcall(function() torch.CudaTensor(1, 1) end)                            -- TODO: REMOVE HACK!!!!!!
    batch.current_words = batch.current_words:cuda()
    batch.left_words = batch.left_words:cuda()
    batch.right_words = batch.right_words:cuda()
    
    batch.mask = batch.mask:cuda()
    batch.proximities = batch.proximities:cuda()
    
    batch.supports = batch.supports:cuda()
    batch.labels = batch.labels:cuda()
  end
  
  return batch
end


-- This function evaluates the accuracy of the recognition model when used
-- to predict concepts given mentions
function SemiCE:evaluate(data_set, mb_size, use_neighbours)
  local criterion = nn.ClassNLLCriterion()
  if self.cuda then
    criterion = criterion:cuda()
  end
  
  local total_error = 0
  local total_accuracy = 0
  local seen = 0
  local prev_tok = 0
  local tok = 1
  while prev_tok < tok do
    local batch = self:make_batch(tok, mb_size, data_set, use_neighbours, false, true)
    local inputs = {batch.current_words,
                    batch.left_words,
                    batch.right_words,
                    batch.supports,
                    batch.mask}
                    -- batch.proximities}
    
    local log_probas = self.recognition:forward(inputs)
    -- criterion
    local err = criterion:forward(log_probas, batch.labels)
    total_error = total_error + err
    -- accuracy
    local preds = nn.ArgMax(2):forward(log_probas)
    local accuracy = torch.sum(torch.eq(preds, batch.labels:long()))
    total_accuracy = total_accuracy + accuracy
    seen = seen + 1
    prev_tok = tok
    tok = batch.last_tok
  end
  
  return total_accuracy / (seen * mb_size)
end


-- The following funtion trains the recognition model in a discriminative
-- way in labeled data
function SemiCE:train_supervised(data_set, mb_size, n_epochs, valid_set, use_neighbours)
  local criterion = nn.ClassNLLCriterion()
  if self.cuda then
    criterion = criterion:cuda()
  end
  local mb_size = mb_size or 32
  local n_epochs = n_epochs or 1
  
  local config = {}
  local state = {}
  
  local train_data = self.data[data_set]
  for e = 1, n_epochs, 1 do
    local shuffle = torch.randperm(train_data:size()[1] / mb_size):type('torch.LongTensor')
    local offset = torch.random(1, mb_size)
    for b = 1, train_data:size()[1] / mb_size, 1 do
      local pos = (shuffle[b] - 1) * mb_size + offset
      local batch = self:make_batch(pos, mb_size, data_set, true, false, true)
      local inputs = {batch.current_words,
                      batch.left_words,
                      batch.right_words,
                      batch.supports,
                      batch.mask}
                    -- batch.proximities}
      
      self.params, self.grad_params = self.recognition:getParameters()
      
      -- feval: optimization function for optim
      local feval = function(x)
      
        if x ~= self.params then
          self.params:copy(x)
        end
        self.grad_params:zero()
            
        local log_probas = self.recognition:forward(inputs)
        local err = criterion:forward(log_probas, batch.labels)
        err = err + self.conf.l1reg * torch.sum(torch.abs(x))
        
        local t = criterion:backward(log_probas, batch.labels)
        self.recognition:backward(inputs, t)
        self.grad_params:add(self.params * self.conf.l1reg)
        
        return err, self.grad_params -- :double()        -- add L2 reg
      end
      
      optim.adam(feval, self.params) -- , config, state)
    end
    
    -- self.prox_drop.p = 0
    
    if valid_set and e % 10 == 0 or self.conf.verbose then
      local val_accu = self:evaluate(valid_set, mb_size, use_neighbours)
      local train_accu = self:evaluate(data_set, mb_size, use_neighbours)
      print(e, data_set, train_accu, valid_set, val_accu, 'l1norm', torch.sum(torch.abs(self.params)))
    end
  end
end


function SemiCE:update_emissions(current_mentions, log_probas)
  local mb_size = current_mentions:size()[1]
  local probas = torch.exp(log_probas):double()
  for b = 1, mb_size, 1 do
    self.emissions:add_to_mention(current_mentions[b], probas[b])
  end
end


-- The first part of the E step tightens the log-likelihood lower bound by
-- optimizing the recognition model parameters
function SemiCE:e_step(data_set, mb_size, max_length, n_epochs)  
  local config = {}
  local state = {}
  
  local max_length = max_length or self.data[data_set]:size()[1]
  local n_epochs = n_epochs or 1
  
  if not self.params then
    self.params, self.grad_params = self.recognition:getParameters()
  end
  
  -- batch variables
  local gradient_dir = {torch.Tensor{-1}, torch.Tensor{-1}, torch.Tensor{-1}, torch.Tensor{0}}
  local batch_thetas = torch.Tensor(mb_size - self.K,
                                    self.K * self.max_cuis * self.max_cuis):fill(0)
  local batch_emissions = torch.Tensor(mb_size - self.K,
                                       self.max_cuis):fill(1 / #self.men_voc):fill(0)
  if self.cuda then
    batch_thetas = batch_thetas:cuda()
    batch_emissions = batch_emissions:cuda()
    for i = 1, 3 do
      gradient_dir[i] = gradient_dir[i]:cuda()
    end
  end
  local sup_a, sup_b, indices
  local b_bound = {}
  local batch_bound = 0
  
  local previous_bound = -1e10
  local batches = 0; local seen = 0
  local total_bound = {mrf_bound = 0, emission_bound = 0, recog_entropy = 0}
  local total_probs = 0
  for e = 1, n_epochs, 1 do
    local tok = 1
    while tok <= max_length - mb_size do
      batches = batches + 1
      -- forward pass: recognition
      local batch = self:make_batch(tok, mb_size, data_set, true, true)
      local inputs = {batch.current_words,
                      batch.left_words,
                      batch.right_words,
                      batch.supports,
                      batch.mask}
      local log_probas = self.recognition:forward(inputs)
      -- select emission probas from supports, words
      for b = 1, mb_size - self.K, 1 do
        batch_emissions[b]:copy(self.emissions:mention_to_support(batch.current_mentions[b]))
      end
      -- select thetas from supports
      for b = 1, mb_size - self.K, 1 do
        sup_a = (torch.expand(batch.supports[b]:view(self.max_cuis, 1),
                              self.max_cuis,
                              self.max_cuis):long():view(self.max_cuis * self.max_cuis) - 1) * #self.cui_voc
        for k = 1, self.K, 1 do
          sup_b = torch.expand(batch.supports[b + k]:view(1, self.max_cuis),
                               self.max_cuis,
                               self.max_cuis):long():view(self.max_cuis * self.max_cuis)
          
          indices = sup_a + sup_b  + (k - 1) * #self.cui_voc * #self.cui_voc
          batch_thetas[b]:sub((k - 1) * self.max_cuis * self.max_cuis + 1,
                              k * self.max_cuis * self.max_cuis):copy(
                                self.theta:view(
                                  self.K * #self.cui_voc * #self.cui_voc
                                ):index(1, indices)
                              )
        end
      end
      -- feval: optimization function for optim
      local feval = function(x)
        if x ~= self.params then
          self.params:copy(x)
        end
        self.grad_params:zero()
        -- forward through recognition function, then likelihood lower bound
        local log_probas = self.recognition:forward(inputs)
        b_bound = self.lower_bound:forward({batch_thetas,
                                            log_probas,
                                            batch_emissions})
        -- backward through likelihood lower bound, then recognition function
        local log_probas_gradient = self.lower_bound:backward({batch_thetas,
                                                               log_probas,
                                                               batch_emissions},
                                                              gradient_dir)[2]
        self.recognition:backward(inputs, log_probas_gradient)
        -- sum bound components
        return -nn.JoinTable(1):forward(b_bound):sum(), self.grad_params   -- need -1 to maximize
      end
      
      gp, batch_bound = optim.adam(feval, self.params, config, state)
      total_bound.mrf_bound = total_bound.mrf_bound + b_bound[1][1]
      total_bound.emission_bound = total_bound.emission_bound + b_bound[2][1]
      total_bound.recog_entropy = total_bound.recog_entropy + b_bound[3][1]
      total_probs = total_probs + b_bound[4][1]
      
      tok = batch.last_tok - self.K -- next batch
      
      if (batches % 100) == 0 then
        seen = batches * (mb_size - self.K)
        print('progress', tok / max_length,
              'mrf_bound', total_bound.mrf_bound / seen - self.MRF.part / (self.K + 1),
              'emission_bound', total_bound.emission_bound / seen,
              'recog_entropy', total_bound.recog_entropy / seen,
              'bound', (total_bound.mrf_bound + total_bound.emission_bound + total_bound.recog_entropy) / seen 
                        - self.MRF.part / (self.K + 1))
      end
    end
    seen = batches * (mb_size - self.K)
    local lower_bound = (total_bound.mrf_bound + total_bound.emission_bound + total_bound.recog_entropy) / seen 
                        - self.MRF.part / (self.K + 1)
    print('mrf_bound', total_bound.mrf_bound / seen - self.MRF.part / (self.K + 1),
          'emission_bound', total_bound.emission_bound / seen,
          'recog_entropy', total_bound.recog_entropy / seen,
          'bound', lower_bound)
    if lower_bound < previous_bound then return 'up' end
    previous_bound = lower_bound
  end
end


-- The second part of the E step computes the expected moments under the
-- recognition model
function SemiCE:make_moments(data_set, mb_size, max_length, start)
  self.record_bound = self.record_bound or {}
  
  local start = start or 1
  local rec_entropy = 0
  
  local max_length = max_length or self.data[data_set]:size()[1]
  self.emissions:fill_counts(0)
  
  local moments = torch.Tensor(self.K * #self.cui_voc * #self.cui_voc):fill(0)
  
  local sup_a, sup_b, probs_a, probs_b
  local probs_prod = torch.zeros(self.max_cuis, self.max_cuis)
    
  local probas = torch.Tensor(mb_size, 2, self.max_cuis)
  local pad_probas = torch.zeros(self.max_cuis)
  local new_probs = torch.Tensor(self.max_cuis * self.max_cuis)
  pad_probas[1] = 1
  
  local tok = start
  local batches = 0
  while tok <= start + max_length - mb_size - 1 do
    batches = batches + 1
    if batches % 100 == 0 then
      print('progress', (tok - start) / max_length)
    end
    local batch = self:make_batch(tok, mb_size, data_set, true, true)
    local inputs = {batch.current_words,
                    batch.left_words,
                    batch.right_words,
                    batch.supports,
                    batch.mask}
                    -- batch.proximities}
    local log_probas = self.recognition:forward(inputs)
    rec_entropy = rec_entropy + torch.cmul(torch.exp(log_probas), log_probas):sum()
    -- write q(z_i | w)
    probas:select(2, 1):copy(batch.supports)
    probas:select(2, 2):copy(torch.exp(log_probas))
    -- collect emissions moments
    self:update_emissions(batch.current_mentions, log_probas)
    -- collect MRF moments
    for b = 1, mb_size - self.K, 1 do
      sup_a = (torch.expand(probas[b][1]:view(self.max_cuis, 1),
                            self.max_cuis,
                            self.max_cuis):long():view(self.max_cuis * self.max_cuis) - 1) * #self.cui_voc
      probs_a = probas[b][2]
      for k = 1, self.K, 1 do
        sup_b = torch.expand(probas[b + k][1]:view(1, self.max_cuis),
                             self.max_cuis,
                             self.max_cuis):long():view(self.max_cuis * self.max_cuis)
        probs_b = probas[b + k][2]
        probs_prod:fill(0)
        probs_prod:addr(probs_a, probs_b)
        -- print(tok, b, k, probs_prod:sum())
        indices = sup_a + sup_b  + (k - 1) * #self.cui_voc * #self.cui_voc
        new_probs:add(moments:index(1, indices), probs_prod:view(self.max_cuis * self.max_cuis))
        moments:indexCopy(1, indices, new_probs)
      end
    end
    -- next batch
    tok = batch.last_tok - self.K
  end
  
  rec_entropy = -rec_entropy / mb_size / batches  
  self.record_bound[#self.record_bound + 1] = {rec_entropy = rec_entropy,
                                               data_set = 'train'}
  
  moments = moments:view(self.K, #self.cui_voc, #self.cui_voc):contiguous()
  local norm_div = moments[1]:sum()
  for k = 1, self.K do
    moments[k] = moments[k]:div(norm_div)
  end
  return moments
end


-- The M step maximizes the log-likelihood lower bound given by the recognition
-- model by optimizing the generative model parameters
function SemiCE:m_step(moments, config)  
  -- M step: MRF
  local last_obj = self.MRF:optimize(moments, config)
  self.theta = self.MRF.theta
  self.record_bound[#self.record_bound].mrf_bound = - last_obj
  
  -- M step: emissions
  local emission_bound
  if self.conf.emission_nn == 1 then
    emission_bound = self.emissions:learn_nn()
  else
    emission_bound = self.emissions:normalize_cuis(self.conf.emission_add_count,
                                                   self.conf.emission_add_count_umls)
  end
  
  self.record_bound[#self.record_bound].emission_bound = emission_bound
  
  -- compute total bound
  local bound = self.record_bound[#self.record_bound].mrf_bound +
                self.record_bound[#self.record_bound].emission_bound +
                self.record_bound[#self.record_bound].rec_entropy
  
  self.record_bound[#self.record_bound].total_bound = bound
  
  return bound
end


-- This function computes the current value of the log-likelihood lower bound
function SemiCE:compute_bound(data_set, mb_size, max_length, start)
  local max_length = max_length or self.data[data_set]:size()[1]
  local start = start or 1
  self.params, self.grad_params = self.recognition:getParameters()
  
  -- batch variables
  local batch_thetas = torch.Tensor(mb_size - self.K,
                                    self.K * self.max_cuis * self.max_cuis)
  local batch_emissions = torch.Tensor(mb_size - self.K,
                                       self.max_cuis):fill(1 / #self.men_voc)
  if self.cuda then
    batch_thetas = batch_thetas:cuda()
    batch_emissions = batch_emissions:cuda()
  end
  local sup_a, sup_b, indices
  local b_bound = {}
  local batch_bound = 0
    
  local batches = 0; local seen = 0
  local total_bound = {mrf_bound = 0, emission_bound = 0, recog_entropy = 0}
  
  local tok = start
  while tok <= start + max_length - mb_size - 1 do
    batches = batches + 1
    -- forward pass: recognition
    local batch = self:make_batch(tok, mb_size, data_set, true, true)
    local inputs = {batch.current_words,
                    batch.left_words,
                    batch.right_words,
                    batch.supports,
                    batch.mask}
    local log_probas = self.recognition:forward(inputs)
    -- select emission probas from supports, words
    for b = 1, mb_size - self.K, 1 do
      batch_emissions[b]:copy(self.emissions:mention_to_support(batch.current_mentions[b]))
    end
    -- select thetas from supports
    for b = 1, mb_size - self.K, 1 do
      sup_a = (torch.expand(batch.supports[b]:view(self.max_cuis, 1),
                            self.max_cuis,
                            self.max_cuis):long():view(self.max_cuis * self.max_cuis) - 1) * #self.cui_voc
      for k = 1, self.K, 1 do
        sup_b = torch.expand(batch.supports[b + k]:view(1, self.max_cuis),
                             self.max_cuis,
                             self.max_cuis):long():view(self.max_cuis * self.max_cuis)
        
        indices = sup_a + sup_b  + (k - 1) * #self.cui_voc * #self.cui_voc
        batch_thetas[b]:sub((k - 1) * self.max_cuis * self.max_cuis + 1,
                            k * self.max_cuis * self.max_cuis):copy(
                              self.theta:view(
                                self.K * #self.cui_voc * #self.cui_voc
                              ):index(1, indices)
                            )
      end
    end
    
    local log_probas = self.recognition:forward(inputs)
    b_bound = self.lower_bound:forward({batch_thetas,
                                          log_probas,
                                          batch_emissions})
    
    total_bound.mrf_bound = total_bound.mrf_bound + b_bound[1][1]
    total_bound.emission_bound = total_bound.emission_bound + b_bound[2][1]
    total_bound.recog_entropy = total_bound.recog_entropy + b_bound[3][1]
    
    tok = batch.last_tok - self.K -- next batch
    
    if (batches % 100) == 0 then
      seen = batches * (mb_size - self.K)
      print('progress', tok / max_length,
            'mrf_bound', total_bound.mrf_bound / seen - self.MRF.part / (self.K + 1),
            'emission_bound', total_bound.emission_bound / seen,
            'recog_entropy', total_bound.recog_entropy / seen,
            'bound', (total_bound.mrf_bound + total_bound.emission_bound + total_bound.recog_entropy) / seen 
                      - self.MRF.part / (self.K + 1))
    end
  end
  seen = batches * (mb_size - self.K)
  local mrf_bound = total_bound.mrf_bound / seen - self.MRF.part / (self.K + 1)
  local emission_bound = total_bound.emission_bound / seen
  local recog_entropy = total_bound.recog_entropy / seen
  print('mrf_bound', mrf_bound,
        'emission_bound', emission_bound,
        'recog_entropy', recog_entropy,
        'bound', mrf_bound + emission_bound + recog_entropy)
  self.record_bound[#self.record_bound + 1] = {mrf_bound = mrf_bound,
                                               rec_entropy = recog_entropy,
                                               emission_bound = emission_bound,
                                               total_bound = mrf_bound + emission_bound + recog_entropy,
                                               data_set = 'dev' .. start}
  return mrf_bound + emission_bound + recog_entropy
end


return SemiCE

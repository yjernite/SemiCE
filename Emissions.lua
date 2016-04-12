local class = require 'class'

local Emissions = class('Emissions')


function Emissions:__init(semi)
  self.non_zeros = {}
  self.closeness = {}
  
  self.semi = semi
  
  local cui, close
  for i = 1, #semi.cui_voc, 1 do
    self.non_zeros[i] = {}
    self.closeness[i] = {}
  end
  for men = 1, #semi.men_voc do
    for j = 1, semi.max_cuis do
      cui = semi.data.men_to_cui_supports[men][j][1]
      close = semi.data.men_to_cui_supports[men][j][2]
      self.non_zeros[cui][#self.non_zeros[cui] + 1] = men
      self.closeness[cui][#self.closeness[cui] + 1] = close
    end
  end
  
  -- sort cuis by support size
  self.sorted_cuis = {}
  for i, sup in pairs(self.non_zeros) do
    if i > 2 and i < #self.non_zeros - self.semi.max_cuis then --TODO: fix
      local tab = {}
      tab.cui = i
      tab.len_sup = #sup
      if not sup then print(i) end
      table.insert(self.sorted_cuis, tab)
    end
  end
  table.sort(self.sorted_cuis, function(a, b) return a.len_sup < b.len_sup end)
  
  -- cuis to mentions
  self.cui_to_mentions = {}
  local idx = 1
  for i = 1, #self.non_zeros do
    self.cui_to_mentions[i] = {min_idx = idx, max_idx = idx, indices={}}
    if #self.non_zeros[i] == 0 then print(i) end
    for j = 1, #self.non_zeros[i] do
      men = self.non_zeros[i][j]
      self.cui_to_mentions[i].indices[men] = idx
      idx = idx + 1
    end
    self.cui_to_mentions[i].max_idx = idx - 1
  end
  
  -- mentions to cuis
  self.men_to_cuis = semi.data.men_to_cui_supports:select(3, 1):long()
  for men = 1, #semi.men_voc do
    for j = 1, semi.max_cuis do
      cui = self.men_to_cuis[men][j]
      self.men_to_cuis[men][j] = self.cui_to_mentions[cui].indices[men]
    end
  end
  
  self.counts_storage = torch.DoubleTensor(idx - 1):fill(0)
  self.storage = torch.DoubleTensor(idx - 1):fill(1)
  
  for cui_id = 1, #self.cui_to_mentions do
    local first = self.cui_to_mentions[cui_id].min_idx
    local last = self.cui_to_mentions[cui_id].max_idx
      if first <= last then
        local support = self.storage:sub(first, last)
        support:div(support:sum())
    end
  end
  
end


-- fill counts vector
function Emissions:fill_counts(val)
  return self.counts_storage:fill(val)
end


-- add to counts vector
function Emissions:add_to_counts(val)
  return self.counts_storage:add(val)
end


-- returns vector of (counts(men, cui_id))_men
function Emissions:cui_to_support(cui_id)
  return self.counts_storage:sub(self.cui_to_mentions[cui_id].min_idx, self.cui_to_mentions[cui_id].max_idx)
end


-- returns vector of (p(men_id | cui))_cui (does not sum to one)
function Emissions:mention_to_support(men_id)
  return self.storage:index(1, self.men_to_cuis[men_id])
end


-- adds CUI probas to the counts vector of mention men_id
function Emissions:add_to_mention(men_id, probas)
  local current = self.counts_storage:index(1, self.men_to_cuis[men_id]):clone()
  return self.counts_storage:indexCopy(1, self.men_to_cuis[men_id], current:add(probas))
end


-- Learns emissions multinomial parameters, regularize with UMLS
function Emissions:normalize_cuis(add_prox, add_equal)
  local add_close = add_prox or 1
  local add_same = add_equal or 1
  for cui_id = 2, #self.cui_to_mentions - 25 do
    local first = self.cui_to_mentions[cui_id].min_idx
    local last = self.cui_to_mentions[cui_id].max_idx
      if first <= last then
        local support = self.counts_storage:sub(first, last)
        local moments = support:clone()
        local close = torch.Tensor(self.closeness[cui_id])
        local same = close:clone():apply(function(x) return (x == 1) and 1 or 0 end)
        moments:add(add_close)
        moments:add(same * add_same)
        moments:div(moments:sum())
        self.storage:sub(first, last):copy(moments)
    end
  end
  return torch.cmul(self.counts_storage,
                    torch.log(self.storage)):sum() / self.counts_storage:sum()
end


-- computes the optimal objective bound (debugging)
function Emissions:test_obj()
  return torch.cmul(self.counts_storage,
                    torch.log(self.storage)):sum() / self.counts_storage:sum()
end


--[[
Parameterized emission distribution. TODO: fix CUDA
]]--
-- makes and initialize emission distribution NN
function Emissions:make_emissions(word_d, cui_d, cuda, batch_size)
  self.word_d = word_d or self.semi.word_d
  self.cui_d = cui_d or self.semi.cui_d
  self.cuda = cuda or false
  self.batch_size = batch_size or 32
  
  self.word_embedding = nn.LookupTableMaskZero(#self.semi.word_voc, self.word_d)
  self.cui_embedding = nn.LookupTableMaskZero(#self.semi.cui_voc, self.cui_d)
  
  local input_cui = nn.Identity()()
  local input_words = nn.Identity()()
  local input_prox = nn.Identity()()
  local input_mask = nn.Identity()()
  
  local cui_rep = self.cui_embedding(input_cui)                             -- batch x 1 x cui_d
  
  local word_reps = self.word_embedding(input_words)                        -- (batch * n_mentions) x max_words x word_d
  local word_rep = nn.Sum(2)(word_reps)                                     -- (batch * n_mentions) x word_d
  local word_rep_proj = nn.Linear(self.word_d, self.cui_d)(word_rep)        -- (batch * n_mentions) x cui_d
  
  local word_rep_final = nn.View(self.batch_size, -1, self.cui_d)(
                           word_rep_proj)                                   -- batch x n_mentions x cui_d
  local word_cui = nn.Sum(3)(nn.MM(false, true)({word_rep_final, cui_rep})) -- batch x n_mentions
  
  local prox_scores = nn.Linear(5, 1)(nn.View(-1, 5)(input_prox))            -- (batch * n_mentions) x 1
  local proximities = nn.View(self.batch_size, -1)(prox_scores)             -- batch x n_mentions
  
  local log_probas = nn.LogSoftMax()(
                        nn.CAddTable()(
                          {word_cui, proximities, input_mask}))             -- batch x n_mentions
  
  self.emissions_nn = nn.gModule({input_cui,
                                  input_words,
                                  input_prox,
                                  input_mask},
                                 {log_probas})
  
  if self.cuda then
    self.emissions_nn = self.emissions_nn:cuda()
  end
end


-- prepares inputs to compute partial objective on a batch of CUIS
-- given current counts and emissions NN
function Emissions:make_batch(batch_start)
  local batch_len = self.batch_size
  local batch_end = math.min(batch_start + batch_len - 1, #self.sorted_cuis)
  local batch_length = batch_end - batch_start + 1
  local max_len = self.sorted_cuis[batch_end].len_sup
  
  collectgarbage()
  local in_cuis = torch.LongTensor(self.batch_size, 1):fill(0)
  local in_words = torch.LongTensor(self.batch_size, max_len, self.semi.max_words):fill(0)
  local in_proxs = torch.Tensor(self.batch_size, max_len, 5):fill(0)
  local in_masks = torch.Tensor(self.batch_size, max_len):fill(-1e12)
  
  local close = torch.Tensor(max_len):fill(0)
  
  local cui, len_sup, men
  for i = 1, batch_length do
    cui = self.sorted_cuis[batch_start + i - 1].cui
    len_sup = self.sorted_cuis[batch_start + i - 1].len_sup
    
    in_cuis[i][1] = cui
    for j, men in pairs(self.non_zeros[cui]) do
      in_words[i][j]:copy(self.semi.data.men_to_words[men])
    end
    in_masks[i]:sub(1, len_sup):fill(0)
    close:sub(1, len_sup):copy(torch.Tensor(self.closeness[cui]))
    in_proxs[i]:select(2, 1):copy(
        close:clone():apply(
            function(x) return (x == 1) and 1 or 0 end))
    in_proxs[i]:select(2, 2):copy(
        close:clone():apply(
            function(x) return (x >= 0.9) and 1 or 0 end))
    in_proxs[i]:select(2, 3):copy(
        close:clone():apply(
            function(x) return (x >= 0.8) and 1 or 0 end))
    in_proxs[i]:select(2, 4):copy(
        close:clone():apply(
            function(x) return (x >= 0.7) and 1 or 0 end))
    in_proxs[i]:select(2, 5):copy(
        close:clone():apply(
            function(x) return (x >= 0.5) and 1 or 0 end))
  end
  
  in_words = in_words:view(self.batch_size * max_len, self.semi.max_words):contiguous()
  
  if self.cuda then
    pcall(function() torch.CudaTensor(1, 1) end)                            -- TODO: REMOVE HACK!!!!!!
    in_proxs = in_proxs:cuda()
    in_masks = in_masks:cuda()
  end
  
  return {in_cuis, in_words, in_proxs, in_masks}
end


-- computes objective and gradients for emissions NN
function Emissions:run_corpus(x)
  local cui_id, cui_moments, inputs, log_probas, grads
  
  self.params:copy(x)
  pcall(function() torch.CudaTensor(1, 1) end)                            -- TODO: REMOVE HACK!!!!!!
  self.grad_params:zero()
  
  local batch_len = 32
  local batch_start = 1
  
  local total_bound = 0
  local seen = 0
  
  while batch_start < #self.sorted_cuis do
    if self.sorted_cuis[batch_start].len_sup > 0 then
      inputs = self:make_batch(batch_start)
      log_probas = self.emissions_nn:forward(inputs)
      
      seen = seen + 1
      if seen % 50 == 0 then
        print(batch_start, inputs[3]:size(2))
        collectgarbage()
      end
      
      grads = torch.Tensor(self.batch_size, inputs[3]:size(2)):fill(0)
      for i = 1, self.batch_size do
        cui_id = inputs[1][i][1]
        if cui_id > 0 then
          -- update emissions for cui_id
          self.storage:sub(self.cui_to_mentions[cui_id].min_idx,
                           self.cui_to_mentions[cui_id].max_idx):copy(
                              torch.exp(log_probas[i]):sub(1, #self.non_zeros[cui_id]))
          -- compute bound
          grads[i]:sub(1, #self.non_zeros[cui_id]):copy(self:cui_to_support(cui_id))
        end
      end
      
      if self.cuda then grads = grads:cuda() end
      
      total_bound = total_bound + torch.cmul(grads, log_probas):sum()
      
      self.emissions_nn:backward(inputs, -grads)
      batch_start = batch_start + self.batch_size
    else
      batch_start = batch_start + 1
    end
  end
  print('gone through data')
  total_bound = - total_bound / (self.counts_storage:sub(self.cui_to_mentions[3].min_idx,
                                                         self.cui_to_mentions[#self.cui_to_mentions - 25].max_idx):sum())
  
  print(total_bound)
  return total_bound, self.grad_params:double()        -- add L2 reg
end


-- Learns emissions NN parameters with lbfgs / adam / adagrad
function Emissions:learn_nn(conf)
  local conf = conf or {}
  conf.algo = conf.algo or 'lbfgs'
  conf.maxIter = conf.maxIter or 10
  conf.maxEval = conf.maxEval or 15
  conf.tolFun = conf.tolFun or 1e-3
  
  local config, state
  
  if not self.params then
    self.params, self.grad_params = self.emissions_nn:getParameters()
  end
  
  local feval = function(x)
    if conf.algo == 'lbfgs' then
      print('#iter ', state.nIter, '#eval ', state.funcEval)
    end
    return self:run_corpus(x)
  end
  
  if conf.algo == 'lbfgs' then
    config = {}
    config.maxIter = conf.maxIter or 10
    config.maxEval = conf.maxEval or 15
    config.tolFun = conf.tolFun or 2e-3
    config.tolX = conf.tolX or 1e-15
    config.lineSearch = nil -- conf.lineSearch or optim.lswolfe
    config.lineSearchOptions = {}
    config.lineSearchOptions.maxIter = 5
    config.verbose = true
    state = {}
    local gp, objs = optim.lbfgs(feval, self.params:double(), config, state)
    return objs[#objs]
  elseif conf.algo == 'adam' then
    config = {}
    state = {}
    local x = self.params:double()
    for epoch = 1, conf.maxIter do
      optim.adam(feval, x, config, state)
    end
  elseif conf.algo == 'adagrad' then
    config = {}
    state = {}
    local x = self.params:double()
    for epoch = 1, conf.maxIter do
      optim.adagrad(feval, x, config, state)
    end
  end
end

return Emissions

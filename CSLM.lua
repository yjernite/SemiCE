--[[
This class implements a chain Markov Chain of order K with factorized
log-potentials, along with functions to maximize a lifted lower bound on
data log-likelihood
]]--
local class = require 'class'
local optim = require 'optim'
local cutorch = require 'cutorch'
local nn =  require 'nn'

local CSLM = class('CSLM')


function CSLM:__init(K, T, D , U, W, verbose)
  self.verbose = verbose
  if U then
    -- hyper-parameters
    self.K = W:size(1)
    self.T = W:size(2)
    self.D = W:size(3)
    -- parameters
    self.U = U:view(1,self.T,self.D):contiguous():cuda()
    self.W = W:transpose(2, 3):contiguous():cuda()
  else
    -- hyper-parameters
    self.K = K    -- length of dependency
    self.T = T    -- vocabulary size
    self.D = D    -- representation dimension
    -- parameters
    self.U = torch.CudaTensor(1, self.T, self.D):uniform(-1, 1)
    self.W = torch.CudaTensor(self.K, self.D, self.T):uniform(-1, 1)
  end
  self.totheta = nn.MM():cuda()
  self.theta = torch.DoubleTensor(self.K, self.T, self.T)
  self.max_rows=torch.CudaTensor(self.K, self.T, 1)
  self.max_cols=torch.CudaTensor(self.K, 1, self.T)
  -- gradients
  self.gradU = torch.CudaTensor(1, self.T, self.D)
  self.gradW = torch.CudaTensor(self.K, self.D, self.T)
  -- marginals
  self.mu = torch.CudaTensor(self.K + 1, self.T)
  self.delta = torch.CudaTensor(self.K + 1, self.T):fill(0)
  self.mu2 = torch.DoubleTensor(self.K, self.T, self.T) -- not necessarily useful: strored on the GPU in mem after running md:mp(true)
  self.mem = torch.CudaTensor(self.K, self.T, self.T)
  -- log-partition
  self.part = 1e3 * torch.pow(self.T, self.K+1)
end


-- makes theta, keeps a version on the CPU and 3 on the GPU
-- (precomputes the exponential twice to avoid overflow)
-- requires a GPU with at least 4GB for K=2, voc size 10,000
function CSLM:make_theta()
  self.thetap = self.totheta:forward({self.U:expand(self.K, self.T, self.D),
                                      self.W}):mul(self.K + 1)
  self.theta = self.thetap:double()
  self.max_rows = self.theta:max(3):cuda()
  self.max_cols = self.theta:max(2):cuda()
  self.phi_rows = self.thetap:clone():add(-1,
                                          self.max_rows:expand(self.K,
                                                               self.T,
                                                               self.T)):exp()
  self.phi_cols = self.thetap:clone():add(-1,
                                          self.max_cols:expand(self.K,
                                                               self.T,
                                                               self.T)):exp()
end


-- ensures that the maximum norm of an embedding is maxn (default 2)
function CSLM:normal(maxn)
  local maxn = maxn or 2
  self.U:view(self.T, self.D):renorm(2, 1, maxn):view(1, self.T, self.D)
  self.W = self.W:double():permute(1, 3, 2):contiguous()
  self.W:view(self.K * self.T, self.D):renorm(2, 1, maxn):view(self.K,
                                                               self.T,
                                                               self.D)
  self.W = self.W:permute(1, 3, 2):contiguous():cuda()
end

-- reinitializes embeddings at random with a max norm of 2,
-- does NOT call make_theta
function CSLM:random_init()
  self.U:uniform(-1, 1)
  self.W:uniform(-1, 1)
  self:normal(2)
  self.delta:fill(0)
end


-- message passing in log space to avoid overflow for bigger GPUs
function CSLM:big_mp(final)
  local t = sys.clock()
  sys.tic()
  local final = final or false
  local phi = self.mem:copy(self.phi_rows)  --phi_rows=exp(theta - max_rows)
  local max_delta = self.delta:max(2)
  local exp_delta = self.delta:clone():add(-1, max_delta:expand(self.K + 1,
                                                                self.T)):exp()
  local log_psi = torch.CudaTensor(self.K + 1,self.T):fill(0)
  local psi = torch.CudaTensor(self.K + 1,self.T)
  local messages_up = torch.CudaTensor(self.K, self.T)
  local messages_down = torch.CudaTensor(self.K, self.T)
  --going up
  --pre_exp = exp(delta), (K+1) x T x T
  local pre_exp = exp_delta:sub(1, self.K):view(self.K,
                                                1,
                                                self.T):expand(self.K,
                                                               self.T,
                                                               self.T)
  -- exp(\theta_{k, i, j} + \delta_{k, j})_{k, i, j}
  phi:cmul(pre_exp)
  t = sys.toc()
  if self.verbose then
    print('phimin', phi:min(), 'phisummin', phi:sum(3):min())
  end
  -- log(\sum_j \phi_{k, i, j})
  messages_up = phi:sum(3):log()
  messages_up:add(max_delta:sub(1, self.K):expand(self.K, self.T))
  messages_up:add(self.max_rows:view(self.K, self.T))
  --
  log_psi[self.K+1] = messages_up:sum(1)
  log_psi[self.K+1]:add(self.delta[self.K + 1])
  log_psi:sub(1, self.K):add(-1, messages_up)
  log_psi:sub(1, self.K):add(log_psi[self.K + 1]:view(1,
                                                      self.T):expand(self.K,
                                                                     self.T))
  if self.verbose then
    print('exp_delta', exp_delta:sum(), 'phisum', phi:sum(),
          'logpsisum', log_psi:sum())
  end
  local max_psi = log_psi:max(2)
  log_psi:add(-1, max_psi:expand(self.K + 1, self.T))
  psi:exp(log_psi)
  --
  local log_part = torch.log(psi[self.K+1]:sum()) + max_psi[self.K + 1][1]
  log_psi:add(max_psi:expand(self.K + 1, self.T))
  self.mu[self.K+1]=log_psi[self.K+1]:double()
  self.mu[self.K+1]:add( -log_part )
  self.mu[self.K+1]:exp()
  -- pairwise
  if  final then   
    self.mem:copy(self.thetap)
    self.mem:add(log_psi:sub(1, self.K):view(self.K,
                                             self.T,
                                             1):expand(self.K,
                                                       self.T,
                                                       self.T))
    self.mem:add(self.delta:sub(1, self.K):view(self.K,
                                                1,
                                                self.T):expand(self.K,
                                                               self.T,
                                                               self.T))
    self.mem:add(-1 * log_part)
    self.mu2 = self.mem:exp():double()
    for k = 1, self.K do
        semi.MRF.mu2[k]:div(semi.MRF.mu2[k]:sum())
    end
  else
    --going down
    phi = self.mem:copy(self.phi_cols) --phi_cols=exp(theta-max_cols)
    pre_exp = psi:sub(1,self.K):view(self.K, self.T, 1):expand(self.K,
                                                               self.T,
                                                               self.T)
    phi:cmul(pre_exp)
    messages_down = phi:sum(2):view(self.K, self.T)
    messages_down:log()
    messages_down:add(max_psi:sub(1, self.K):expand(self.K, self.T))
    messages_down:add(self.max_cols:view(self.K, self.T))
    messages_down:add(self.delta:sub(1, self.K))
    self.mu:sub(1, self.K):exp( messages_down:add( -1 * log_part ) )
  end
  self.part = log_part
  return log_part
end


-- simple dual decomposition
function CSLM:dd(rate)
  local rate = rate or 10
  local grad_delta = torch.CudaTensor(self.K+1, self.T):fill(0)
  local oldpart = math.huge
  local function hasnan()
    local allreal = (self.mu:sum() == self.mu:sum()) and
                    (self.part == self.part)
    return not allreal
  end
  for i = 1, 50 do
    oldpart = self.part  
    md:big_mp()
    if self.part > oldpart or hasnan() then
      if self.verbose then
        print("back")
      end
      self.delta:add(-rate, grad_delta)
      rate = rate/2
      if not i == 50 then
        self.part = oldpart
      end
    else
      grad_delta = grad_delta:fill(0)
      for l=1, K do
        grad_delta[l] = grad_delta[l]:add(self.mu[self.K + 1], -1, self.mu[l])
        grad_delta[self.K + 1] = grad_delta[self.K + 1]:add(-1, grad_delta[l])
      end
    end
    self.delta:add(rate, grad_delta)
  end
end


-- optim-based dual decomposition (currently using LBFGS)
function CSLM:dd_opt(rate, conf)
  local grad_delta = torch.CudaTensor(self.K + 1, self.T):fill(0)
  local function hasnan()
    local allreal = (self.mu:sum() == self.mu:sum()) and
                    (self.part == self.part)
    return not allreal
  end
  
  local delta_opt = torch.DoubleTensor((self.K + 1) * self.T):copy(self.delta:view((self.K + 1) * self.T))
  local grad_delta_opt = torch.DoubleTensor((self.K + 1) * self.T)
 
  local function feval(x)
    self.delta = x:view(self.K + 1, self.T):cuda()
    local obj = self:big_mp()
    grad_delta = grad_delta:fill(0)
    for l=1, self.K do
      grad_delta[l] = grad_delta[l]:add(self.mu[self.K + 1], -1, self.mu[l])
      grad_delta[self.K + 1] = grad_delta[self.K + 1]:add(-1, grad_delta[l])
    end
    grad_delta_opt:copy(grad_delta:view((self.K + 1) * self.T)):mul(-1)
    return obj, grad_delta_opt
  end
  
  local conf = conf or {}
  local config = {}
  config.maxIter = conf.maxIter or 50
  config.maxEval = conf.maxEval or 100
  config.tolFun = conf.tolFun or 1e-5
  config.tolX = conf.tolX or 1e-5
  config.lineSearch = conf.lineSearch or optim.lswolfe
  config.verbose = false
  optim.lbfgs(feval, delta_opt, config)
  return grad_delta:norm()
end


-- back-propagates the moments in theta to U,W
function CSLM:make_grad(data, reg)
  local reg = reg or 0      -- L2 regularization
  local grad = self.mem:copy(torch.DoubleTensor():add(data, -1, self.mu2))
  local gradPar = self.totheta:backward({self.U:expand(self.K, self.T, self.D), self.W}, grad)
  self.gradU = gradPar[1]:sum(1):add(-reg, self.U)
  self.gradW = gradPar[2]:add(-reg, self.W)
end


-- gradient_step isn't used with the optim framework
function CSLM:gradient_step(data, rate, norm, reg)
  print('|U[1]|   ',self.U[1][1]:norm(),'|U[2]|   ',self.U[1][2]:norm())
  print('|W[1][1]|   ',self.W:norm(2,2)[1][1][1],'|W[2][1]|   ',self.W:norm(2,2)[2][1][1])
  rate = rate or 10
  norm = norm or false
  reg = reg or 0
  md.delta:fill(0)
  self:dd(10,true)
  local obj = self:objective(data)
  print('Objective', obj, 'Regularized',obj - reg * (self.U:norm() + self.W:norm()), 'deltasum', self.delta:sum())
  self:big_mp(true)
  self:make_grad(data)
  ---option 1: renormalize after each gradient step
  if norm then
    self.U:add(rate, self.gradU)
    self.W:add(rate, self.gradW)
    self:normal()
  ---option 2: L2 regularization
  else
    self.gradU:add(-reg, self.U)
    self.gradW:add(-reg, self.W)
    self.U:add(rate, self.gradU)
    self.W:add(rate, self.gradW)
  end
  print('|gradU|   ',self.gradU:norm(),'|gradW|   ',self.gradW:norm())
  self:make_theta()
end


-------- some functions for the optimisation packages
-- computes -objective (because the optim package minimizes)
function CSLM:objective(data, reg, pre)
  local pre = pre or -1
  local reg = reg or 0
  local score = torch.DoubleTensor(self.K, self.T, self.T):cmul(data, self.theta)
  local sc = score:sum()
  local nm = self.U:norm() + self.W:norm()
  print('dd-prec ' .. pre .. '\t Objective ' .. (sc - self.part) / (self.K + 1) .. '\t Regularized ' .. (sc - self.part) / (self.K + 1) - reg * nm)
  return -(sc - self.part) / (self.K + 1) + reg * nm
end


-- flattens the parameters and gradients (also multiplies the gradients by -1 to fit within the minimization framework)
function CSLM:getParameters()
  self.parameters = torch.cat(self.U:view(self.T * self.D):double(),
                              self.W:view(self.K * self.T * self.D):double())
  self.gradParameters = torch.cat(self.gradU:view(self.T * self.D):double(),
                                  self.gradW:view(self.K * self.T * self.D):double())
  self.gradParameters:mul(-1)
end


-- optimization using LBFGS
function CSLM:optimize(data, conf)
  self:getParameters()
  local pre
  local conf = conf or {}
  local reg = conf.reg or 0
  local max_norm = conf.max_norm or false
  local algorithm = conf.algo or 'lbfgs'
  --
  local function feval(x)
    self.U = x:sub(1, self.T * self.D):view(1, self.T, self.D):cuda()
    self.W = x:sub(self.T * self.D + 1,
                   (self.K + 1) * self.T * self.D):view(self.K,
                                                        self.D,
                                                        self.T):cuda()
    
    if max_norm then
      self:normal(max_norm)
    end
    self:make_theta()
    if self.delta:norm() > 1e4 then
      print ("zeroing delta")
      self.delta:fill(0)
    end
    pre = self:dd_opt()
    local obj = self:objective(data, reg, pre)
    self:big_mp(true)
    self:make_grad(data, reg)
    self:getParameters()
    return obj, self.gradParameters
  end
  --
  config = conf
  config.maxIter = conf.maxIter or 20
  config.maxEval = conf.maxEval or 50
  config.tolFun = conf.tolFun or 1e-4
  config.tolX = conf.tolX or 1e-15
  if algorithm == 'lbfgs' then
    print('running lbfgs with parameters ', config)
    config.lineSearch = conf.lineSearch or optim.lswolfe
    config.verbose = true
    collectgarbage()
    local gp, objs = optim.lbfgs(feval, self.parameters, config)
    return objs[#objs]
  elseif algorithm == 'adam' then
    print('running adam with parameters ', config)
    local gp, f_val
    for iter = 1, config.maxIter do
      gp, f_val = optim.adam(feval, self.parameters, config)
    end
    pre = self:dd_opt()
    return self:objective(data, reg, pre)
  end
end

return CSLM





--[[





function CSLM:big_mp(final)
  local t = sys.clock()
  sys.tic()
  local final = final or false
  local phi
  local max_delta = self.delta:max(2)     -- K + 1
  local exp_delta = self.delta:clone():add(-1, max_delta:expand(self.K + 1,
                                                                self.T)):exp()
  local log_psi = torch.CudaTensor(self.K + 1,self.T):fill(0)
  local psi = torch.CudaTensor(self.K + 1,self.T)
  local messages_up = torch.CudaTensor(self.K, self.T)    -- from 1, ..., K to K+1
  local messages_down = torch.CudaTensor(self.K, self.T)  -- from K+1 to 1, ..., K
  --going up
  
  local pre_exp = exp_delta:sub(1, self.K):view(self.K,
                                                1,
                                                self.T):expand(self.K,
                                                               self.T,
                                                               self.T)
  -- exp(\theta_{k, i, j} + \delta_{k, j})_{k, i, j}
  phi = self.mem:copy(self.phi_rows)  -- phi_rows = exp(theta - max_rows), K x T x T
  phi:cmul(pre_exp)                   -- pre_exp = exp(delta_{1, ..., K} - max_delta), K x (1 * T) x T
  -- phi_{k, i, j} = exp(\theta_{k, i, j} + \delta_{k, j} - max_rows_{k, i} - max_delta_{k})
  t = sys.toc()
  if self.verbose then
    print('phimin', phi:min(), 'phisummin', phi:sum(3):min())
  end
  -- log(\sum_j \phi_{k, i, j})
  messages_up = phi:sum(3):log()
  messages_up:add(max_delta:sub(1, self.K):expand(self.K, self.T))
  messages_up:add(self.max_rows:view(self.K, self.T))
  -- m_up_{k, i} = log(\sum_j exp(\theta_{k, i, j} + \delta_{k, j}))
  log_psi[self.K+1] = messages_up:sum(1)
  log_psi[self.K+1]:add(self.delta[self.K + 1])
  log_psi:sub(1, self.K):add(-1, messages_up)
  log_psi:sub(1, self.K):add(log_psi[self.K + 1]:view(1,
                                                      self.T):expand(self.K,
                                                                     self.T))
  -- log_psi_{K+1, i} = \delta_{K+1, i} + \sum_k log(\sum_j exp(\theta_{k, i, j} + \delta_{k, j}))
  -- log_psi_{K+1, i} = log(\prod_k(\sum_j exp(\theta_{k, i, j} + \delta_{K+1, i} + \delta_{k, j})))
  -- and
  -- log_psi_{k, i} = log(\prod_{l ~= k }(\sum_j exp(\theta_{l, i, j} + \delta_{K+1, i} + \delta_{l, j})))
  if self.verbose then
    print('exp_delta', exp_delta:sum(), 'phisum', phi:sum(),
          'logpsisum', log_psi:sum())
  end
  local max_psi = log_psi:max(2)
  log_psi:add(-1, max_psi:expand(self.K + 1, self.T))
  psi:exp(log_psi) -- minus max_psi !!!
  --
  local log_part = torch.log(psi[self.K + 1]:sum()) + max_psi[self.K + 1][1]
  log_psi:add(max_psi:expand(self.K + 1, self.T))
  self.mu[self.K+1]=log_psi[self.K+1]:double()
  self.mu[self.K+1]:add( -log_part )
  self.mu[self.K+1]:exp()
  -- pairwise
  if  final then   
    self.mem:copy(self.thetap)
    self.mem:add(log_psi:sub(1, self.K):view(self.K,
                                             self.T,
                                             1):expand(self.K,
                                                       self.T,
                                                       self.T))
    self.mem:add(self.delta:sub(1, self.K):view(self.K,
                                                1,
                                                self.T):expand(self.K,
                                                               self.T,
                                                               self.T))
    self.mem:add(-1 * log_part)
    self.mu2 = self.mem:exp():double()
    for k = 1, self.K do
        semi.MRF.mu2[k]:div(semi.MRF.mu2[k]:sum())
    end
  else
    --going down
    phi = self.mem:copy(self.phi_cols) --phi_cols = exp(theta - max_cols)
    pre_exp = psi:sub(1, self.K):view(self.K, self.T, 1):expand(self.K,
                                                                self.T,
                                                                self.T)
    phi:cmul(pre_exp)
    messages_down = phi:sum(2):view(self.K, self.T)
    messages_down:log()
    messages_down:add(max_psi:sub(1, self.K):expand(self.K, self.T))
    messages_down:add(self.max_cols:view(self.K, self.T))
    messages_down:add(self.delta:sub(1, self.K))
    self.mu:sub(1, self.K):exp( messages_down:add( -1 * log_part ) )
    print(self.mu:sum(2))
  end
  self.part = log_part
  return log_part
end


]]--

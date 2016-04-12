local class = require 'class'
local optim = require 'optim'
local cutorch = require 'cutorch'
local nn =  require 'nn'

local ChainMRF = class('ChainMRF')


function ChainMRF:__init(K, T, verbose)
  self.verbose = verbose
  -- hyper-parameters
  self.K = K    -- length of dependency
  self.T = T    -- vocabulary size
  -- parameters
  self.theta = torch.DoubleTensor(self.K, self.T, self.T)
  self.grad_theta = torch.DoubleTensor(self.K, self.T, self.T)
  self.max_rows=torch.CudaTensor(self.K, self.T, 1)
  self.max_cols=torch.CudaTensor(self.K, 1, self.T)
  -- gradients
  self.theta_p = self.theta:cuda()
  -- marginals
  self.mu = torch.CudaTensor(self.K + 1, self.T)
  self.delta = torch.CudaTensor(self.K + 1, self.T):fill(0)
  self.mu2 = torch.DoubleTensor(self.K, self.T, self.T) -- not necessarily useful: strored on the GPU in mem after running md:mp(true)
  self.mem = torch.CudaTensor(self.K, self.T, self.T)
  -- log-partition
  self.part = 1e3 * torch.pow(self.T, self.K + 1)
end


-- makes theta, keeps a version on the CPU and 3 on the GPU
-- (precomputes the exponential twice to avoid overflow)
-- requires a GPU with at least 4GB for K=2
function ChainMRF:make_theta()
  self.theta_p = self.theta:cuda()
  self.max_rows = self.theta:max(3):cuda()
  self.max_cols = self.theta:max(2):cuda()
  self.phi_rows = self.theta_p:clone():add(-1,
                                          self.max_rows:expand(self.K,
                                                               self.T,
                                                               self.T)):exp()
  self.phi_cols = self.theta_p:clone():add(-1,
                                          self.max_cols:expand(self.K,
                                                               self.T,
                                                               self.T)):exp()
end


-- ensures that the maximum norm of an embedding is maxn (default 2)
function ChainMRF:normal(maxn)
  local maxn = maxn or 2
  self.theta:renorm(2, 1, maxn)
end


-- reinitializes embeddings at random with a max norm of 2,
-- does NOT call make_theta
function ChainMRF:random_init()
  self.theta:uniform(-1, 1)
  self:normal(2)
  self.delta:fill(0)
end


-- message passing in log space to avoid overflow for bigger GPUs
function ChainMRF:big_mp(final)
  local t = sys.clock()
  sys.tic()
  local final = final or false
  local phi = self.mem:copy(self.phi_rows)  --phi_rows=exp(theta-max_rows)
  local max_delta = self.delta:max(2)
  local exp_delta = self.delta:clone():add(-1, max_delta:expand(self.K + 1,
                                                                self.T)):exp()
  local log_psi = torch.CudaTensor(self.K + 1,self.T):fill(0)
  local psi = torch.CudaTensor(self.K + 1,self.T)
  local messages_up = torch.CudaTensor(self.K, self.T)
  local messages_down = torch.CudaTensor(self.K, self.T)
  --going up
  local pre_exp = exp_delta:sub(1, self.K):view(self.K,
                                                1,
                                                self.T):expand(self.K,
                                                               self.T,
                                                               self.T)
  phi:cmul(pre_exp)
  t = sys.toc()
  if self.verbose then
    print('phimin', phi:min(), 'phisummin', phi:sum(3):min())
  end
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
    self.mem:copy(self.theta_p)
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


-- optim-based dual decomposition (currently using LBFGS)
function ChainMRF:dd_opt(rate, conf)
  local grad_delta = torch.CudaTensor(self.K + 1, self.T):fill(0)
  local function hasnan()
    local allreal = (self.mu:sum() == self.mu:sum()) and
                    (self.part == self.part)
    return not allreal
  end
  
  local delta_opt = torch.DoubleTensor((self.K + 1) * self.T):copy(self.delta:view((self.K + 1) * self.T))
  local grad_delta_opt = torch.DoubleTensor((self.K + 1) * self.T)
 
  local function feval(x)
    self.delta = x:view(self.K+1, self.T):cuda()
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


-- add gradients from score, partition and regularization
function ChainMRF:make_grad(data, reg)
  local reg = reg or 0      -- L2 regularization
  self.grad_theta:add(data, -1, self.mu2)
  if reg > 0 then
    self.grad_theta:add(-reg, self.theta)
  end
end


-------- some functions for the optimisation packages
-- computes -objective (because the optim package minimizes)
function ChainMRF:objective(data, reg, pre)
  local pre = pre or -1
  local reg = reg or 0
  local score = torch.DoubleTensor(self.K, self.T, self.T):cmul(data, self.theta)
  local sc = score:sum()
  local nm = self.theta_p:norm()
  print('dd-prec ' .. pre .. '\t Objective ' .. (sc - self.part) / (self.K + 1) .. '\t Regularized ' .. (sc - self.part) / (self.K + 1) - reg * nm)
  return -(sc - self.part) / (self.K + 1) + reg * nm
end


-- flattens the parameters and gradients (also multiplies the gradients by -1 to fit within the minimization framework)
function ChainMRF:getParameters()
  self.parameters = self.theta:view(self.K * self.T * self.T):contiguous()
  self.gradParameters = self.grad_theta:view(self.K * self.T * self.T):contiguous()
  self.gradParameters:mul(-1)
end


-- optimization using LBFGS
function ChainMRF:optimize(data, conf)
  self:getParameters()
  local pre
  local conf = conf or {}
  local reg=conf.reg or 0
  --
  local function feval(x)
    self.theta:copy(x:view(self.K, self.T, self.T))
    self:make_theta()
    pre = self:dd_opt()
    local obj = self:objective(data, reg, pre)
    self:big_mp(true)
    self:make_grad(data, reg)
    self:getParameters()
    return obj, self.gradParameters
  end
  --
  config = {}
  config.maxIter = conf.maxIter or 20
  config.maxEval = conf.maxEval or 50
  config.tolFun = conf.tolFun or 1e-9
  config.tolX = conf.tolX or 1e-15
  config.lineSearch = conf.lineSearch or optim.lswolfe
  config.verbose = true
  collectgarbage()
  local gp, objs = optim.lbfgs(feval, self.parameters, config)
  return objs[#objs]
end

return ChainMRF

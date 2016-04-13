--[[
Training script for the model
]]--
paths = require 'paths'
SemiCE = require 'SemiCE'

cmd = torch.CmdLine()
cmd:text()
cmd:text()
cmd:text('Semi-supervised mixed model training')
cmd:text()
cmd:text('Options')
cmd:option('-seed', 123, 'initial random seed')

-- model options
cmd:option('-K', 3, 'length of dependencies')
cmd:option('-word_dim', 50, 'dimension of word embedding in recognition function')
cmd:option('-cui_dim', 50, 'dimension of cui embedding in recognition function')
cmd:option('-mrf_dim', 100, 'rank of factorized theta in the MRF (defaults to 0: unfactorized)')

-- data options
cmd:option('-unsup_train_size', 2e5, 'size of unlabeled data to use as training set for semi-supervised training')
cmd:option('-unsup_dev_size', 1e5, 'size of unlabeled data to use as development set for semi-supervised training')

-- initialization options
cmd:option('-init_type', 'semeval', 'Initialization type')
cmd:option('-init_mrf_iter', 100, 'initialize the MRF by running init_recog_iter of LBFGS')
cmd:option('-init_recog_iter', 10, 'if init_type == semeval: initialize the recognition model by running init_recog_iter of supervised training')
cmd:option('-l1reg', 0.01, 'if init_type == semeval: Initial L1 regularization')
cmd:option('-init_m_step_epochs', 10, 'if init_type == umls_pseudo: number of epochs in the M step SGD for initialization')

-- EM options
cmd:option('-m_step_epochs', 3, 'number of epochs in the M step SGD')
cmd:option('-mrf_iter', 20, 'max number of LBFGS iterations for the MRF in the e step')
cmd:option('-em_iter', 15, 'number of EM iterations')

-- optimization options
cmd:option('-unsup_batch', 512, 'size of batches on unlabeled data for moments collection')
cmd:option('-unsup_batch_train', 256, 'size of batches on unlabeled for M step SGD')
cmd:option('-sup_batch', 256, 'size of batches on unlabeled data')
cmd:option('-dropout', 0, 'dropout in recognition model')
cmd:option('-mrf_reg', 0, 'dropout in recognition model')

-- emission distribution options
cmd:option('-emission_nn', 0, 'use an emission distribution parameterized as a neural network')
cmd:option('-emission_add_count', 1, 'add counts to moments for all mentions in support')
cmd:option('-emission_add_count_umls', 100, 'add counts to moments for all mentions that are umls descriptions')

-- House keeping
cmd:option('-stroption', 'name', 'string option')
cmd:option('-cuda', 1, 'enabling gpu use')
cmd:option('-gpu_id', 0, 'if a specific gpu is required (1-indexed)')

cmd:text()

-- parse input params
params = cmd:parse(arg)

params.rundir = cmd:string('run', params, {dir=true})
paths.mkdir(params.rundir)

-- create log file
cmd:addTime('semi-supervised', '%x %X ')
cmd:log(params.rundir .. '/out.log', params)

if params.gpu_id > 0 then
    cutorch.setDevice(params.gpu_id)
end

-- Start program
semi = SemiCE(params)

vocab_files = {men_voc_file = 'processed_data/mention_vocab.txt',
               cui_voc_file = 'processed_data/label_vocab.txt',
               word_voc_file = 'processed_data/word_vocab.txt'}
semi:read_vocabs(vocab_files)
semi:read_data('processed_data/processed_data.hdf5')

semi:make_recognition(params.dropout)
semi:make_generative()
semi:make_lower_bound()

-- Initialization
if params.init_type == 'semeval' then
    -- Initializes the recognition model by discriminative training
    semi:train_supervised('train_sentences', params.sup_batch, params.init_recog_iter, 'dev_sentences', true)
elseif params.init_type == 'umls_pseudo' then
    -- Performs an initial E step using only the pseudo-counts from UMLS
    semi.emissions:fill_counts(0)
    semi.emissions:normalize_cuis(params.emission_add_count,
                                  params.emission_add_count_umls)
    semi.theta = semi.MRF.theta
    semi.theta:fill(0)
    semi.MRF.part = 0
    semi:e_step('unsup_sentences', params.unsup_batch_train, params.unsup_train_size, params.init_m_step_epochs)
end

my_moments = semi:make_moments('unsup_sentences', params.unsup_batch, params.unsup_train_size)
-- The initial M step is necessary to make the MRF log-likelihood lifted
-- lower bound tight enough to be useful
semi:m_step(my_moments, {maxIter = params.init_mrf_iter,
                         maxEval = 1.5 * params.init_mrf_iter,
                         reg = params.mrf_reg})

-- Saving
torch.save(params.rundir .. '/moments_epoch_0.t7', semi.emissions)
semi:save_to_file(params.rundir .. '/model_save_epoch_0.t7')

-- Run params.em_iter EM iterations
for i = 1, params.em_iter do
    print('iteration \t', i)
    
    -- Evaluating on semeval data...
    local train_acc = semi:evaluate('train_sentences', params.sup_batch, true)
    local dev_acc = semi:evaluate('dev_sentences', params.sup_batch, true)
    print(train_acc, dev_acc)
    semi.record_bound[#semi.record_bound + 1] = {iter = i, train_acc = train_acc, dev_acc = dev_acc}
    
    -- E step
    print('E step \t', i)
    semi:e_step('unsup_sentences', params.unsup_batch_train, params.unsup_train_size, params.m_step_epochs)
    collectgarbage()
    my_moments = semi:make_moments('unsup_sentences', params.unsup_batch, params.unsup_train_size)
    
    -- M step
    print('M step \t', i)
    semi:m_step(my_moments, {maxIter = params.mrf_iter,
                             maxEval = 1.5 * params.mrf_iter,
                             reg = params.mrf_reg})
    
    -- Evaluate on dev and save moments and model parameters
    print('bound_dev\t', i)
    semi:compute_bound('unsup_sentences', params.unsup_batch, params.unsup_dev_size, params.unsup_train_size + 1e4)
    print('Saving \t', i)
    torch.save(params.rundir .. '/moments_epoch_' .. i .. '.t7', semi.emissions)
    torch.save(params.rundir .. '/unsup_bounds_' .. params.m_step_epochs .. '.t7', semi.record_bound)
    semi:save_to_file(params.rundir .. '/model_save_epoch_' .. i .. '.t7')
end

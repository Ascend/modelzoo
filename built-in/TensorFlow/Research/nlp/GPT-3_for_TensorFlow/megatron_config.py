import tensorflow as tf


config = {
      'mode': 'train',       # train, eval or train_and_eval
      'model': 'megatron1',
      'batch_size':2,       # for training
      'dtype': tf.float16,
      'loss_scale': 'Backoff',             #could be float or string. If float, static loss scaling is applied.
                                           #If string, the corresponding automatic loss scaling algorithm is used.
                                           #Must be one of 'Backoff' of 'LogMax' (case insensitive).
      'scale_min': 1.0,
      'scale_max': 2.**15,
      'step_window': 200,
      'optimizer': 'smdp_adamw',
      'optimizer_params': {
        'beta1': 0.9,
        'beta2': 0.99,
        'epsilon': 1e-06,
      },

      'learning_rate': 0.0002,
      'training_steps': 300000,
      'warmup_steps': 10000,
      'lr_warmup_mode':'linear',          #'linear','cosine'
      'lr_decay_mode':'cosine',           #'linear','cosine','constant'


      'iter_size': [1.0],
      'weight_decay': 0.01,
      #'data_path' :"/autotest/CI_daily/ModelZoo_GPT-3_TF/data/tfrecord_dataset",
      'data_path' :"file:///data/tfrecord_dataset",
      #'data_path' :"/development/h00452838/new/GPT_NPU/tfrecord_dataset",
   
      #----- logger config ---------
      'display_every': 1,
      'log_name': 'megatron.log',
      'log_dir': './results/megatron_result/',
      'save_summary_steps': 10000,
      'save_checkpoints_steps': 1000,

      #----------- Megatron related -----------------------
      'n_vocab': 50519,
      'n_ctx': 512,
      'embed_dropout': 0.1,
      'attn_dropout': 0.1,
      'res_dropout': 0.1,
      'scale_by_in': True,
      'scale_by_depth': True,
      'model_parallel': False,
      'model_parallel_dim': 1,          #当前混合并行模式下，如果不使用模型并行，必须设为1


      'megatron_params_1': {
        'n_head': 16,
        'n_embd': 1920,
        'n_layer': 4, },

      'megatron_params_2': {
        'n_head': 16,
        'n_embd': 1024,
        'n_layer': 4, },

      'megatron_params_3': {
        'n_head': 40,
        'n_embd': 3840,
        'n_layer': 25, },

      'megatron_params_4': {
        'n_head': 40,
        'n_embd': 5120,
        'n_layer': 32, },

      'gpt_345m': {
        'n_head': 16,
        'n_embd': 1024,
        'n_layer': 24, },

         
}


def megatron_config():
  config['global_batch_size'] = config['batch_size']
  config['num_training_samples'] = 10000
  # config['nstep'] = config['training_steps'] / config['global_batch_size']
  config['nstep'] = config['training_steps']
  config['do_checkpoint'] = True

  if config['model'] == 'megatron1':
    config['n_head'] = config['megatron_params_1']['n_head']
    config['n_embd'] = config['megatron_params_1']['n_embd']
    config['n_layer'] = config['megatron_params_1']['n_layer']
  elif config['model'] == 'megatron2':
    config['n_head'] = config['megatron_params_2']['n_head']
    config['n_embd'] = config['megatron_params_2']['n_embd']
    config['n_layer'] = config['megatron_params_2']['n_layer']
  elif config['model'] == 'megatron3':
    config['n_head'] = config['megatron_params_3']['n_head']
    config['n_embd'] = config['megatron_params_3']['n_embd']
    config['n_layer'] = config['megatron_params_3']['n_layer']
  elif config['model'] == 'megatron4':
    config['n_head'] = config['megatron_params_4']['n_head']
    config['n_embd'] = config['megatron_params_4']['n_embd']
    config['n_layer'] = config['megatron_params_4']['n_layer']
  elif config['model'] == 'gpt_345m':
    config['n_head'] = config['gpt_345m']['n_head']
    config['n_embd'] = config['gpt_345m']['n_embd']
    config['n_layer'] = config['gpt_345m']['n_layer']
  else:
    print('Model not defined')
  return config

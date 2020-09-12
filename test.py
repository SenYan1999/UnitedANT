from transformer.nn_transformer import TRANSFORMER
audio_options = {
    'ckpt_file'     : './pretrained_model/states-1500000.ckpt',
    'load_pretrain' : 'True',
    'no_grad'       : 'True',
    'dropout'       : 'default',
    'spec_aug'      : 'False',
    'spec_aug_prev' : 'True',
    'weighted_sum'  : 'False',
    'select_layer'  : -1,
}
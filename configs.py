#Config file to set configs and parameters

import argparse
import os


def get_configs():

    parser = argparse.ArgumentParser()

    parser.add_argument('--save_dir', '-sd', type=str, default='models',
                        help='path where checkpoints will be saved')

    parser.add_argument('--project_name', '-pn', type=str, default='recipe-retrieval',
                        help='name of the directory where models will be saved within save_dir')

    parser.add_argument('--model_name', '-mn', type=str, default='recipe-retrieval',
                        help='save_dir/project_name/model_name will be the path where logs and checkpoints are stored')

    parser.add_argument('--transfer_from', '-tf', type=str, default='',
                        help='specify model name to transfer from')

    parser.add_argument('--suff', '-suff',type=str, default='',
                        help='the id of the dictionary to load for training')

    parser.add_argument('--image_model','-im', type=str, default='resnet50', choices=['resnet18', 'resnet50', 'resnet101',
                                                                                 'resnet152', 'inception_v3'])

    parser.add_argument('--recipe1m_dir', '-rd', type=str, default='data',
                        help='directory where recipe1m dataset is extracted')

    parser.add_argument('--aux_data_dir','-ad', type=str, default='data',
                        help='path to other necessary data files (eg. vocabularies)')

    parser.add_argument('--crop_size', '-cropsz',type=int, default=224, help='size for randomly or center cropping images')

    parser.add_argument('--image_size', '-imgsz',type=int, default=256, help='size to rescale images')

    parser.add_argument('--log_step', '-logst', type=int , default=10, help='step size for printing log info')

    parser.add_argument('--learning_rate', '-lr', type=float, default=0.001,
                        help='base learning rate')

    parser.add_argument('--scale_learning_rate_cnn', '-slrcnn', type=float, default=0.01,
                        help='lr multiplier for cnn weights')

    parser.add_argument('--lr_decay_rate', '-lrd', type=float, default=0.99,
                        help='learning rate decay factor')

    parser.add_argument('--lr_decay_every', '-lrde',type=int, default=1,
                        help='frequency of learning rate decay (default is every epoch)')

    parser.add_argument('--weight_decay', '-wd', type=float, default=0.)

    parser.add_argument('--embed_size', '-embsz', type=int, default=512,
                        help='hidden size for all projections')

    parser.add_argument('--n_att', '-natt', type=int, default=8,
                        help='number of attention heads in the instruction decoder')

    parser.add_argument('--n_att_ingrs', '-natting', type=int, default=4,
                        help='number of attention heads in the ingredient decoder')

    parser.add_argument('--transf_layers', '-trfly', type=int, default=16,
                        help='number of transformer layers in the instruction decoder')

    parser.add_argument('--transf_layers_ingrs', '-trflying',type=int, default=4,
                        help='number of transformer layers in the ingredient decoder')

    parser.add_argument('--num_epochs', '-numep', type=int, default=400,
                        help='maximum number of epochs')

    parser.add_argument('--batch_size', '-bs', type=int, default=128)

    parser.add_argument('--num_workers', '-numw', type=int, default=8)

    parser.add_argument('--dropout_encoder', '-dde', type=float, default=0.3,
                        help='dropout ratio for the image and ingredient encoders')

    parser.add_argument('--dropout_decoder_r', '-ddr', type=float, default=0.3,
                        help='dropout ratio in the instruction decoder')

    parser.add_argument('--dropout_decoder_i', '-ddi', type=float, default=0.3,
                        help='dropout ratio in the ingredient decoder')

    parser.add_argument('--finetune_after', '-fa', type=int, default=-1,
                        help='epoch to start training cnn. -1 is never, 0 is from the beginning')

    parser.add_argument('--loss_weight', '-lw', nargs='+', type=float, default=[1.0, 0.0, 0.0, 0.0],
                        help='training loss weights. 1) instruction, 2) ingredient, 3) eos 4) cardinality')

    parser.add_argument('--max_eval', '-maxev', type=int, default=4096,
                        help='number of validation samples to evaluate during training')

    parser.add_argument('--label_smoothing_ingr', '-lsing', type=float, default=0.1,
                        help='label smoothing for bce loss for ingredients')

    parser.add_argument('--patience', '-pat', type=int, default=50,
                        help='maximum number of epochs to allow before early stopping')

    parser.add_argument('--maxseqlen', '-maxsql', type=int, default=15,
                        help='maximum length of each instruction')

    parser.add_argument('--maxnuminstrs', '-maxnuminst', type=int, default=10,
                        help='maximum number of instructions')

    parser.add_argument('--maxnumims', '-maxnimg', type=int, default=5,
                        help='maximum number of images per sample')

    parser.add_argument('--maxnumlabels', '-maxnlabl',type=int, default=20,
                        help='maximum number of ingredients per sample')

    parser.add_argument('--es_metric', '-evalmet', type=str, default='loss', choices=['loss', 'iou_sample'],
                        help='early stopping metric to track')

    parser.add_argument('--eval_split', '-evalsplt', type=str, default='val')

    parser.add_argument('--numgens', '-numg', type=int, default=3)

    parser.add_argument('--greedy', '-greedy', dest='greedy', action='store_true',
                        help='enables greedy sampling (inference only)')
    parser.set_defaults(greedy=False)

    parser.add_argument('--temperature', '-temp', type=float, default=1.0,
                        help='sampling temperature (when greedy is False)')

    parser.add_argument('--beam', '-beam', type=int, default=-1,
                        help='beam size. -1 means no beam search (either greedy or sampling)')

    parser.add_argument('--ingrs_only','-ingonly', dest='ingrs_only', action='store_true',
                        help='train or evaluate the model only for ingredient prediction')
    parser.set_defaults(ingrs_only=False)

    parser.add_argument('--recipe_only', '-rcponly', dest='recipe_only', action='store_true',
                        help='train or evaluate the model only for instruction generation')
    parser.set_defaults(recipe_only=False)

    parser.add_argument('--log_term', '-logtrm', dest='log_term', action='store_true',
                        help='if used, shows training log in stdout instead of saving it to a file.')
    parser.set_defaults(log_term=False)

    parser.add_argument('--notensorboard', '-notensrbrd',dest='tensorboard', action='store_false',
                        help='if used, tensorboard logs will not be saved')
    parser.set_defaults(tensorboard=True)

    parser.add_argument('--resume', '-rsm',dest='resume', action='store_true',
                        help='resume training from the checkpoint in model_name')
    parser.set_defaults(resume=False)

    parser.add_argument('--nodecay_lr', '-nodcylr', dest='decay_lr', action='store_false',
                        help='disables learning rate decay')
    parser.set_defaults(decay_lr=True)

    parser.add_argument('--load_jpeg', '-ldjpg', dest='use_lmdb', action='store_false',
                        help='if used, images are loaded from jpg files instead of lmdb')
    parser.set_defaults(use_lmdb=True)

    parser.add_argument('--get_perplexity', '-gtprxpl', dest='get_perplexity', action='store_true',
                        help='used to get perplexity in evaluation')
    parser.set_defaults(get_perplexity=False)

    parser.add_argument('--use_true_ingrs', '-usetruing', dest='use_true_ingrs', action='store_true',
                        help='if used, true ingredients will be used as input to obtain the recipe in evaluation')
    parser.set_defaults(use_true_ingrs=False)

    parser.add_argument('--recipe1m_path', '-rp', type=str,
                        default='data',
                        help='recipe1m path')

    parser.add_argument('--save_path', '-sp', type=str, default='data/',
                        help='path for saving vocabulary wrapper')

    parser.add_argument('--threshold_ingrs', type=int, default=10,
                        help='minimum ingr count threshold')

    parser.add_argument('--threshold_words', type=int, default=10,
                        help='minimum word count threshold')

    parser.add_argument('--maxnumingrs', type=int, default=20,
                        help='max number of ingredients')

    parser.add_argument('--minnuminstrs', type=int, default=2,
                        help='max number of instructions (sentences)')

    parser.add_argument('--minnumingrs', type=int, default=2,
                        help='max number of ingredients')

    parser.add_argument('--minnumwords', type=int, default=20,
                        help='minimum number of characters in recipe')

    parser.add_argument('--forcegen', dest='forcegen', action='store_true')
    parser.set_defaults(forcegen=False)

    args = parser.parse_args()

    return args

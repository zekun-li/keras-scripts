import argparse
        
def main(args):
    print (args)
    MODE = args.mode
    FREEZE_LAYERS = args.iffreeze
    
    USE_CHECKPOINT = args.ifcheckpoint
    Checkpoint = [args.checkpoint_file_rgb, args.checkpoint_file_flow]

    LEARNING_RATE =  args.lr
    prefix = str(args.ith) + '_mean_lr'+str(LEARNING_RATE) + '_'+args.eval_type
    INVERSE = args.ifinverse # if inverse the stream 
    
            
    

if __name__ == '__main__':
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval-type', 
        help='specify model type. 1 stream (rgb or flow) or 2 stream (joint = rgb and flow).', 
        type=str, choices=['rgb', 'flow', 'joint'], default='rgb')

    parser.add_argument('--no-imagenet-pretrained',
        help='If set, load model weights trained only on kinetics dataset. Otherwise, load model weights trained on imagenet and kinetics dataset.',
        action='store_true')

    parser.add_argument('--mode', help='training or testing', default = 'training')
    parser.add_argument('--iffreeze', help='whether freeze featex layers', default = False, action = 'store_true')
    parser.add_argument('--ifcheckpoint', help = 'whether use checkpoint', default = False, action = 'store_true')
    parser.add_argument('--ifinverse', help = 'whether inverse the stream', default = False , action = 'store_true')
    parser.add_argument('--checkpoint-file-rgb', help = 'path point to rgb checkpoint file', default = None)
    parser.add_argument('--checkpoint-file-flow', help = 'path point to flow checkpoint file', default = None)
    parser.add_argument('--lr', help = 'learning rate', default = 0.1, type = float)
    parser.add_argument('--ith', help = 'the ith training ', default = 0, type = int)

    args = parser.parse_args()
    main(args)

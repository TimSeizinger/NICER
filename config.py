# ----- application parameters:

verbosity = 2                                      # set from 0 (silent) to 3 (most verbose)
preview = True                                     # toggle to true to show preview imgs -> slower
interactive_preview_sliders = True                 # toggle to activate interactive sliders for preview when not training -> slower
interactive_training_sliders = True                # toggle to activate interactive sliders during training -> slower
use_auto_brightness_normalizer = False             # toggle to activate ABN

valid_assesors = ['NIMA_VGG16', 'NIMA_mobilenetv2', 'SSMTPIAA', 'SSMTPIAA_fine']
assessor = 'SSMTPIAA'                            # Which assessor network should be used by default

#TODO ADAPTIVE_MSE_SCORE_REG, ADAPTIVE_MSE_STYLE_CHANGES_REG
valid_SSMTPIAA_losses = ['MSE_SCORE_REG', 'ADAPTIVE_MSE_SCORE_REG', 'MOVING_MSE_SCORE_REG',
                         'BCE_SCORE', 'BCE_SCORE_REG',
                         'MSE_STYLE_CHANGES', 'MSE_STYLE_CHANGES_REG',
                         'MSE_STYLE_CHANGES_HINGE', 'MSE_STYLE_CHANGES_HINGE_REG']
SSMTPIAA_loss = 'MSE_SCORE_REG'                      # Which ia_pre losses should be used by default

automatic_epoch = True                             # Automatically stop enhancing image if loss remains unchanged
automatic_epoch_target = 0.002

save_animation = False                              # saves animation of the network editing
save_animation_with_extra_info = False              # saves animation of the network editing with extra loss and score values
save_composite_animation = True                    # saves animation of the network editing with animated graphs
save_loss_graph = True                             # saves loss graph
save_score_graph = True                            # saves score graph
animate_graphs = False                              # saves animated versions of graphs

choose_save_name_and_folder = False                # able to specify image output folder and name

debug_image_pipeline = False


gamma = 0.1
epochs = 100
optim = 'sgd'                                      # also supports adam
optim_lr = 0.025
optim_momentum = 0.9
hinge_val = 0.25                                    # value for hinge loss

# ----- image parameters:

rescale = True
final_size = 1920                                   # final size when saving. Attention: bigger imgs require more memory
supported_extensions = ['jpg', 'jpeg', 'png']
supported_extensions_raw = ['dng']                  # legacy, deprecated

# ----- Architecture parameters:
# (you should not have to change these)

can_filter_count = 8
can_checkpoint_path = 'models/can8_epoch10_final.pt'
nima_checkpoint_path = 'models/nima_vgg_bright2.pkl'
nima_mobilenet_checkpoint_path = 'models/fine-imagenet.pth'
IA_pre_checkpoint_path = 'models/pre-one-score-regression.pth'
IA_fine_checkpoint_path = 'models/fine-one-score-regression.pth'

desired_distribution = [0.0, 0.0, 0.0, 0.0, 0.0, 0.01, 0.09, 0.15, 0.55, 0.20]


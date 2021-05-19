# ----- application parameters:

verbosity = 2                                      # set from 0 (silent) to 3 (most verbose)
preview = True                                     # toggle to true to show preview imgs -> slower
interactive_preview_sliders = True                 # toggle to activate interactive sliders for preview when not training -> slower
interactive_training_sliders = True                # toggle to activate interactive sliders during training -> slower

legacy_loss_NIMA = False                           # switches to NIMA with the original NICER loss function
MSE_loss_NIMA = False                              # switches to NIMA with the MSE loss used by Judge.

save_animation = False                             # saves animation of the network editing
save_animation_with_extra_info = True              # saves animation of the network editing with extra loss and score values
save_composite_animation = True                    # saves animation of the network editing with animated graphs
save_loss_graph = True                             # saves loss graph
save_score_graph = True                            # saves score graph
animate_graphs = True                              # saves animated versions of graphs

choose_save_name_and_folder = False                # able to specify image output folder and name


gamma = 0.1
epochs = 50
optim = 'sgd'                                      # also supports adam
optim_lr = 0.05
optim_momentum = 0.9

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
nima_mobilenet_checkpoint_path = 'models/pre-one-score-regression.pth'
IA_pre_checkpoint_path = 'models/pre-one-score-regression.pth'
Judge_checkpoint_path = 'models/pre-one-score-regression.pth'

desired_distribution = [0.0, 0.0, 0.0, 0.0, 0.0, 0.01, 0.09, 0.15, 0.55, 0.20]


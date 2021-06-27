import queue
import time

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
from PIL import Image
from torchvision.transforms import transforms

import config
from autobright import normalize_brightness
from neural_models import error_callback, CAN, NIMA_VGG
from utils import nima_transform, print_msg, loss_with_l2_regularization, loss_with_filter_regularization, \
    weighted_mean, jans_normalization, jans_transform, jans_padding, tensor_debug, RingBuffer, hinge

from IA_folder.old.utils import mapping
from IA_folder.IA import IA
from IA2NIMA.NIMA import NIMA


class NICER(nn.Module):

    def __init__(self, checkpoint_can, checkpoint_nima, device='cpu', can_arch=8):
        self.config: config = config

        super(NICER, self).__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("Using", self.device)

        if can_arch != 8 and can_arch != 7:
            error_callback('can_arch')

        can = CAN(no_of_filters=8) if can_arch == 8 else CAN(no_of_filters=7)
        can.load_state_dict(torch.load(checkpoint_can, map_location=self.device)['state_dict'])
        can.eval()
        can.to(self.device)

        self.finetuned = "fine" in config.IA_fine_checkpoint_path  # TODO Remove this line

        nima_mobilenetv2 = NIMA("scores-one, change_regress")
        nima_mobilenetv2.load_state_dict(torch.load(config.nima_mobilenet_checkpoint_path)['model_state'])
        nima_mobilenetv2.eval()
        nima_mobilenetv2.to(self.device)

        nima_vgg16 = NIMA_VGG(models.vgg16(pretrained=True))
        nima_vgg16.load_state_dict(torch.load(checkpoint_nima, map_location=self.device))
        nima_vgg16.eval()
        nima_vgg16.to(self.device)

        ia_pre = IA("scores-one, change_regress", True, False, mapping, None, pretrained=False)
        ia_pre.load_state_dict(torch.load(config.IA_pre_checkpoint_path))
        ia_pre.eval()
        ia_pre.to(self.device)

        ia_fine = NIMA("scores-one, change_regress")
        ia_fine.load_state_dict(torch.load(config.IA_fine_checkpoint_path)['model_state'])
        ia_fine.eval()
        ia_fine.to(self.device)

        torch.autograd.set_detect_anomaly(True)

        ## queue for outputs to GUI
        self.queue = queue.Queue()

        ##queue for interactive slider inputs from GUI (deprecated)
        self.in_queue = queue.Queue()

        # self.filters is a leaf-variable, bc it's created directly and not as part of an operation
        self.filters = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=torch.float32, requires_grad=True,
                                    device=self.device)
        self.can = can

        # Image assessor networks
        self.nima_vgg16 = nima_vgg16
        self.nima_mobilenetv2 = nima_mobilenetv2
        self.ia_pre = ia_pre
        self.ia_fine = ia_fine

        self.loss_func_mse = nn.MSELoss('mean').to(self.device)
        self.loss_func_bce = nn.BCELoss(reduction='mean').to(self.device)
        self.weights = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]]).to(self.device)

        self.target0d = torch.tensor(1.0).to(self.device)
        self.target1d = torch.FloatTensor([1.0]).to(self.device)
        self.target2d = torch.FloatTensor([[1.0]]).to(self.device)
        self.length = torch.tensor(10.0).to(self.device)

        self.gamma = config.gamma

        if config.optim == 'sgd':
            self.optimizer = torch.optim.SGD(params=[self.filters], lr=config.optim_lr, momentum=config.optim_momentum)
        elif config.optim == 'adam':
            self.optimizer = torch.optim.Adam(params=[self.filters], lr=config.optim_lr)
        else:
            error_callback('optimizer')

    def forward(self, image: torch.Tensor, image_jan: torch.Tensor, fixedFilters=None, new=False, headless_mode=False,
                nima_vgg16=True, nima_mobilenetv2=True, ssmtpiaa=True, ssmtpiaa_fine=True):
        torch.cuda.synchronize()

        filter_tensor = torch.zeros((8, 224, 224), dtype=torch.float32).to(self.device)

        nima_vgg16_distr_of_ratings, nima_mobilenetv2_distr_of_ratings, ia_pre_ratings, ia_fine_distr_of_ratings = \
            None, None, None, None

        if nima_vgg16:
            # construct filtermap uniformly from given filters
            for l in range(8):
                if fixedFilters:  # filters were fixed in GUI, always use their passed values
                    if fixedFilters[l][0] == 1:
                        filter_tensor[l, :, :] = fixedFilters[l][1]
                    else:
                        filter_tensor[l, :, :] = self.filters.view(-1)[l]
                else:
                    filter_tensor[l, :, :] = self.filters.view(-1)[l]

            # concat filters and img
            mapped_img = torch.cat((image, filter_tensor.cpu()), dim=0).unsqueeze(dim=0).to(self.device)
            # start.record()
            enhanced_img = self.can(mapped_img)  # enhance img with CAN
            tensor_debug(enhanced_img, 'enhanced image')
        else:
            enhanced_img = None

        if nima_mobilenetv2 | ssmtpiaa | ssmtpiaa_fine:
            # construct filtermap for Jan's Assessors
            filter_tensor_jan = torch.zeros((8, image_jan.shape[1], image_jan.shape[2]), dtype=torch.float32).to(self.device)

            for l in range(8):
                if fixedFilters:  # filters were fixed in GUI, always use their passed values
                    if fixedFilters[l][0] == 1:
                        filter_tensor_jan[l, :, :] = fixedFilters[l][1]
                    else:
                        filter_tensor_jan[l, :, :] = self.filters.view(-1)[l]
                else:
                    filter_tensor_jan[l, :, :] = self.filters.view(-1)[l]

            #concat filters and img for Jan's assessors
            mapped_img_jan = torch.cat((image_jan, filter_tensor_jan.cpu()), dim=0).unsqueeze(dim=0).to(self.device)
            enhanced_img_jan = self.can(mapped_img_jan)  # enhance img with CAN
            tensor_debug(enhanced_img_jan, 'enhanced image jan')

            enhanced_img_jan = torch.clip(enhanced_img_jan, 0, 1)
            tensor_debug(enhanced_img_jan, 'enhanced image jan clipped')

            enhanced_img_jan = jans_normalization(enhanced_img_jan)
            tensor_debug(enhanced_img_jan, 'enhanced image jan normalized')

            if config.padding:
                enhanced_img_jan = jans_padding(enhanced_img_jan)
                tensor_debug(enhanced_img_jan, 'enhanced image jan padded')

        if nima_vgg16:
            # NIMA_VGG16, returns NIMA distribution as tensor
            nima_vgg16_distr_of_ratings = self.nima_vgg16(enhanced_img)  # get nima_vgg16 score distribution -> tensor

        if nima_mobilenetv2:
            # NIMA_mobilenetv2, returns NIMA distribution as tensor
            nima_mobilenetv2_distr_of_ratings = self.nima_mobilenetv2(
                enhanced_img_jan)  # get nima_mobilenetv2 score distribution -> tensor

        if ssmtpiaa:
            # IA_pre, returns Dict returns NIMA distribution as tensor
            ia_pre_ratings = self.ia_pre(enhanced_img_jan)  # get ia_pre score -> tensor

        if ssmtpiaa_fine:
            # IA_fine, returns NIMA distribution as tensor
            ia_fine_distr_of_ratings = self.ia_fine(enhanced_img_jan)  # get ia_fine score distribution -> tensor
        if not headless_mode:
            self.queue.put('dummy')  # dummy

        return enhanced_img, nima_vgg16_distr_of_ratings, nima_mobilenetv2_distr_of_ratings, ia_pre_ratings, ia_fine_distr_of_ratings

    def set_filters(self, filter_list):
        # usually called from GUI
        if max(filter_list) > 1:
            filter_list = [x / 100.0 for x in filter_list]

        with torch.no_grad():
            for i in range(5):
                self.filters[i] = filter_list[i]
            self.filters[5] = filter_list[6]  # llf is 5 in can but 6 in gui (bc exp is inserted)
            self.filters[6] = filter_list[7]  # nld is 6 in can but 7 in gui
            self.filters[7] = filter_list[5]  # exp is 7 in can but 5 in gui

    def set_gamma(self, gamma):
        self.gamma = gamma

    def single_image_pass_can(self, image, abn=False, filterList=None, mapToCpu=False):
        """
            pass an image through the CAN architecture 1 time. This is usually called from the GUI, to preview the images.
            It is also called when the image is to be saved, since we then need to apply the final filter intensities onto the image.

            if called_to_save_raw is False, this method will return an 8bit image to show what the current filter combination looks
            like (because PIL cannot process 16bit). If called_to_save_raw is true, it will return the enhanced 16bit image as
            np.uint16 array, to be saved with opencv.imwrite() as 16bit png.
        """

        # filterList is passable as argument because for previwing the imgs in the GUI while optimizing,
        # we cannot use self.filters, as this is used for gradient computatation

        device = self.device if mapToCpu is False else 'cpu'

        if abn:
            bright_norm_img = normalize_brightness(image, input_is_PIL=True)
            image = Image.fromarray(bright_norm_img)

        if image.size[1] > config.final_size or image.size[0] > config.final_size:
            image_tensor = transforms.Compose([
                transforms.Resize(config.final_size),
                transforms.ToTensor()])(image)
        else:
            image_tensor = transforms.ToTensor()(image)

        filter_tensor = torch.zeros((8, image_tensor.shape[1], image_tensor.shape[2]), dtype=torch.float32).to(
            device)  # tensorshape [c,w,h]
        for l in range(8):
            filter_tensor[l, :, :] = filterList[l] if filterList else self.filters.view(-1)[l]

        mapped_img = torch.cat((image_tensor.cpu(), filter_tensor.cpu()), dim=0).unsqueeze(dim=0).to(device)

        try:
            enhanced_img = self.can(mapped_img)  # enhance img with CAN
        except RuntimeError:
            self.can.to('cpu')
            try:
                enhanced_img = self.can(mapped_img)  # enhance img with CAN
            except RuntimeError:
                print("DefaultCPUAllocator - not enough memory to perform this operation")
                return None
            self.can.to('cuda')

        enhanced_img = enhanced_img.cpu()
        enhanced_img = enhanced_img.detach().permute(2, 3, 1, 0).squeeze().numpy()

        enhanced_clipped = np.clip(enhanced_img, 0.0, 1.0) * 255.0
        enhanced_clipped = enhanced_clipped.astype('uint8')

        # returns a np.array of type np.uint8

        return enhanced_clipped

    def re_init(self):
        # deprecated, formerly used for batch mode
        self.filters = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=torch.float32, requires_grad=True,
                                    device=self.device)
        if config.optim == 'sgd':
            self.optimizer = torch.optim.SGD(params=[self.filters], lr=config.optim_lr, momentum=config.optim_momentum)
        elif config.optim == 'adam':
            self.optimizer = torch.optim.Adam(params=[self.filters], lr=config.optim_lr)
        else:
            error_callback('optimizer')

    def enhance_image(self, image_path, re_init=True, fixFilters=None, epochs=config.epochs, thread_stopEvent=None,
                      headless_mode=False, nima_vgg16=True, nima_mobilenetv2=True, ssmtpiaa=True, ssmtpiaa_fine=True):
        """
            optimization routine that is called to enhance an image.
            Usually this is called from the NICER button in the GUI.
            Accepts image path as a string, but also as PIL image.

            Returns a re-sized 8bit image as np.array
        """
        if headless_mode:
            self.queue = None
            self.in_queue = None

        # Scores and Losses lists for visualization
        if nima_vgg16:
            nima_vgg16_scores = []
            nima_vgg16_losses = []
        if nima_mobilenetv2:
            nima_mobilenetv2_scores = []
            nima_mobilenetv2_losses = []
        if ssmtpiaa:
            ia_pre_scores = []
            ia_pre_losses = []
        if ssmtpiaa_fine:
            ia_fine_scores = []
            ia_fine_losses = []

        if re_init:
            self.re_init()
        else:
            # re-init is false, i.e. use user_preset filters that are selected in the GUI
            # re-init can be seen as test whether initial filter values (!= 0) should be used or not during optimization
            user_preset_filters = [self.filters[x].item() for x in range(8)]

        if isinstance(image_path, str):
            bright_normalized_img = normalize_brightness(image_path)
            pil_image = Image.fromarray(bright_normalized_img)
        else:
            pil_image = image_path
            bright_normalized_img = normalize_brightness(pil_image, input_is_PIL=True)
            pil_image = Image.fromarray(bright_normalized_img)

        image_tensor_transformed = nima_transform(pil_image)

        image_tensor_transformed_jan = jans_transform(pil_image)

        if fixFilters:  # fixFilters is bool list of filters to be fixed
            initial_filter_values = []
            for k in range(8):
                if fixFilters[k] == 1:
                    initial_filter_values.append([1, self.filters[k].item()])
                else:
                    initial_filter_values.append([0, self.filters[k].item()])

        loss_buffer = RingBuffer(6)
        score_target = None

        # optimize image:
        print_msg("Starting optimization", 2)
        start_time = time.time()

        for i in range(epochs):
            if headless_mode is False and thread_stopEvent.is_set():
                break

            if config.automatic_epoch and loss_buffer.get_std_dev() is not None:
                if max(loss_buffer.data) - min(loss_buffer.data) < 0.035:
                    break

            ## Check if sliders have been manually adjusted during last iteration, if yes apply adjustments (buggy af)
            # while not self.in_queue.empty():
            #    self.set_filters(self.in_queue.get())

            print_msg("Iteration {} of {}".format(i, epochs), 2)

            self.optimizer.zero_grad()

            if fixFilters:
                enhanced_img, nima_vgg16_distr_of_ratings, nima_mobilenetv2_distr_of_ratings, ia_pre_ratings, \
                ia_fine_distr_of_ratings = self.forward(image_tensor_transformed, image_tensor_transformed_jan,
                                                        fixedFilters=initial_filter_values, headless_mode=headless_mode,
                                                        nima_vgg16=nima_vgg16, nima_mobilenetv2=nima_mobilenetv2,
                                                        ssmtpiaa=ssmtpiaa, ssmtpiaa_fine=ssmtpiaa_fine)
            else:
                enhanced_img, nima_vgg16_distr_of_ratings, nima_mobilenetv2_distr_of_ratings, ia_pre_ratings, \
                ia_fine_distr_of_ratings = self.forward(image_tensor_transformed, image_tensor_transformed_jan,
                                                        headless_mode=headless_mode,
                                                        nima_vgg16=nima_vgg16, nima_mobilenetv2=nima_mobilenetv2,
                                                        ssmtpiaa=ssmtpiaa, ssmtpiaa_fine=ssmtpiaa_fine)

            # NIMA_VGG16 loss, either using MSE or l2 loss with target2d distribution (legacy_NICER_loss_for_NIMA_VGG16)
            if nima_vgg16:
                print('nima_vgg16_losses')
                if re_init:
                    # new for each image
                    nima_vgg16_loss = loss_with_l2_regularization(nima_vgg16_distr_of_ratings.cpu(), self.filters.cpu(),
                                                                  gamma=self.gamma)
                else:
                    nima_vgg16_loss = loss_with_l2_regularization(nima_vgg16_distr_of_ratings.cpu(), self.filters.cpu(),
                                                                  initial_filters=user_preset_filters, gamma=self.gamma)
            if nima_mobilenetv2:
                print('nima_mobilenetv2_loss')
                # NIMA_mobilenetv2 loss
                nima_mobilenetv2_loss = self.loss_func_mse(
                    weighted_mean(nima_mobilenetv2_distr_of_ratings, self.weights, self.length), self.target0d)

            if ssmtpiaa:
                # IA_pre loss
                if config.SSMTPIAA_loss == 'MSE_SCORE_REG':
                    if re_init:
                        ia_pre_loss = \
                            loss_with_filter_regularization(ia_pre_ratings['score'], self.target2d, self.loss_func_mse,
                                                            self.filters, gamma=self.gamma)
                    else:
                        ia_pre_loss = loss_with_filter_regularization(ia_pre_ratings['score'], self.target2d,
                                                                      self.loss_func_mse, self.filters,
                                                                      initial_filters=user_preset_filters, gamma=self.gamma)
                elif config.SSMTPIAA_loss == 'ADAPTIVE_MSE_SCORE_REG':
                    if score_target is None:
                        score_target = min(ia_pre_ratings['score'].item() + 0.3, 1.0)
                        print('score_target is: ' + str(score_target))
                        score_target = torch.FloatTensor([[score_target]]).to(self.device)
                    if re_init:
                        ia_pre_loss = \
                            loss_with_filter_regularization(ia_pre_ratings['score'], score_target, self.loss_func_mse.cpu(),
                                                            self.filters.cpu(), gamma=self.gamma)
                    else:
                        ia_pre_loss = loss_with_filter_regularization(ia_pre_ratings['score'], score_target,
                                                                      self.loss_func_mse.cpu(), self.filters.cpu(),
                                                                      initial_filters=user_preset_filters, gamma=self.gamma)
                elif config.SSMTPIAA_loss == 'MOVING_MSE_SCORE_REG':
                    score_target = min(ia_pre_ratings['score'].item() + 0.2, 1.0)
                    score_target = torch.FloatTensor([[score_target]]).to(self.device)
                    if re_init:
                        ia_pre_loss = \
                            loss_with_filter_regularization(ia_pre_ratings['score'], score_target, self.loss_func_mse.cpu(),
                                                            self.filters.cpu(), gamma=self.gamma)
                    else:
                        ia_pre_loss = loss_with_filter_regularization(ia_pre_ratings['score'], score_target,
                                                                      self.loss_func_mse.cpu(), self.filters.cpu(),
                                                                      initial_filters=user_preset_filters, gamma=self.gamma)
                elif config.SSMTPIAA_loss == 'MSE_STYLE_CHANGES':
                    ia_pre_loss = self.loss_func_mse(ia_pre_ratings['styles_change_strength'],
                                                     torch.zeros(ia_pre_ratings['styles_change_strength'].size()[1]).to(self.device))
                elif config.SSMTPIAA_loss == 'MSE_STYLE_CHANGES_REG':
                    if re_init:
                        ia_pre_loss = \
                            loss_with_filter_regularization((ia_pre_ratings['styles_change_strength'] * 1.5).cpu(),
                                                            torch.zeros(ia_pre_ratings['styles_change_strength'].size()[1]),
                                                            self.loss_func_mse.cpu(), self.filters.cpu(), gamma=self.gamma)
                    else:
                        ia_pre_loss = \
                            loss_with_filter_regularization((ia_pre_ratings['styles_change_strength'] * 1.5).cpu(),
                                                            torch.zeros(ia_pre_ratings['styles_change_strength'].size()[1]),
                                                            self.loss_func_mse.cpu(), self.filters.cpu(),
                                                            initial_filters=user_preset_filters, gamma=self.gamma)
                elif config.SSMTPIAA_loss == 'MSE_STYLE_CHANGES_HINGE':
                    ratings = hinge(ia_pre_ratings['styles_change_strength'], config.hinge_val)
                    ia_pre_loss = self.loss_func_mse(ratings,
                                                     torch.zeros(ratings.size()[1]).to(self.device))
                elif config.SSMTPIAA_loss == 'MSE_STYLE_CHANGES_HINGE_REG':
                    ratings = hinge(ia_pre_ratings['styles_change_strength'], config.hinge_val)
                    if re_init:
                        ia_pre_loss = \
                            loss_with_filter_regularization((ratings * 1.5).cpu(),
                                                            torch.zeros(ratings.size()[1]),
                                                            self.loss_func_mse.cpu(), self.filters.cpu(), gamma=self.gamma)
                    else:
                        ia_pre_loss = \
                            loss_with_filter_regularization((ratings * 1.5).cpu(),
                                                            torch.zeros(ratings.size()[1]),
                                                            self.loss_func_mse.cpu(), self.filters.cpu(),
                                                            initial_filters=user_preset_filters, gamma=self.gamma)
                elif config.SSMTPIAA_loss == 'BCE_SCORE':
                    ia_pre_loss = self.loss_func_bce(ia_pre_ratings['score'], self.target2d)
                elif config.SSMTPIAA_loss == 'BCE_SCORE_REG':
                    if re_init:
                        ia_pre_loss = \
                            loss_with_filter_regularization(ia_pre_ratings['score'], self.target2d, self.loss_func_bce.cpu(),
                                                            self.filters.cpu(), gamma=self.gamma)
                    else:
                        ia_pre_loss = loss_with_filter_regularization(ia_pre_ratings['score'], self.target2d,
                                                                      self.loss_func_bce.cpu(), self.filters.cpu(),
                                                                      initial_filters=user_preset_filters, gamma=self.gamma)
                elif config.SSMTPIAA_loss == 'COMPOSITE':
                    if re_init:
                        score_loss = \
                            loss_with_filter_regularization(ia_pre_ratings['score'], self.target2d, self.loss_func_mse,
                                                            self.filters, gamma=self.gamma)
                    else:
                        score_loss = loss_with_filter_regularization(ia_pre_ratings['score'], self.target2d,
                                                                      self.loss_func_mse, self.filters,
                                                                      initial_filters=user_preset_filters, gamma=self.gamma)

                    ratings = hinge(ia_pre_ratings['styles_change_strength'], config.hinge_val)
                    styles_loss = self.loss_func_mse(ratings,
                                                     torch.zeros(ratings.size()[1]).to(self.device))

                    composite_loss = score_loss * torch.pow(ia_pre_ratings['score'], config.composite_pow) + \
                                     styles_loss * (1 - torch.pow(ia_pre_ratings['score'], config.composite_pow))

                    if config.composite_balance == 0.0:
                        ia_pre_loss = composite_loss
                    elif config.composite_balance < 0:
                        ia_pre_loss = score_loss * abs(config.composite_balance) + \
                                      composite_loss * (1 - abs(config.composite_balance))
                    else:
                        ia_pre_loss = composite_loss * (1 - abs(config.composite_balance)) + \
                                      styles_loss * config.composite_balance
                else:
                    raise Exception('Illegal SSMTPIAA_loss')

            if ssmtpiaa_fine:
                # IA_fine loss
                ia_fine_loss = self.loss_func_mse(weighted_mean(ia_fine_distr_of_ratings, self.weights, self.length), self.target0d)

            # Append each score value to their respective list for later visualization
            # Append each loss value to their respective list for later visualization
            if nima_vgg16:
                nima_vgg16_scores.append(weighted_mean(nima_vgg16_distr_of_ratings, self.weights, self.length).item())
                nima_vgg16_losses.append(nima_vgg16_loss.item())
            if nima_mobilenetv2:
                nima_mobilenetv2_scores.append(
                    weighted_mean(nima_mobilenetv2_distr_of_ratings, self.weights, self.length).item())
                nima_mobilenetv2_losses.append(nima_mobilenetv2_loss.item())
            if ssmtpiaa:
                ia_pre_scores.append(ia_pre_ratings['score'].item())
                ia_pre_losses.append(ia_pre_loss.item())
            if ssmtpiaa_fine:
                ia_fine_scores.append(weighted_mean(ia_fine_distr_of_ratings, self.weights, self.length).item())
                ia_fine_losses.append(ia_fine_loss.item())

            if config.assessor == 'NIMA_VGG16':
                print('using NIMA_VGG16 with loss of: ' + str(nima_vgg16_loss))
                loss = nima_vgg16_loss
            elif config.assessor == 'NIMA_mobilenetv2':
                print('using NIMA_mobilenetv2 with loss of: ' + str(nima_mobilenetv2_loss))
                loss = nima_mobilenetv2_loss
            elif config.assessor == 'SSMTPIAA':
                print('using IA_pre with loss of: ' + str(ia_pre_loss.item()) + 'and score of: ' + str(ia_pre_ratings['score'].item()))
                loss = ia_pre_loss
            elif config.assessor == 'SSMTPIAA_fine':
                print('using IA_fine with loss of: ' + str(ia_fine_loss))
                loss = ia_fine_loss
            else:
                raise Exception("Invalid Assessor in config.assessor: " + config.assessor)

            loss_buffer.append(loss.item())

            loss.backward()
            print('Learning rate = ' + str(self.get_lr()))
            self.optimizer.step()

            if not headless_mode:
                filters_for_queue = [self.filters[x].item() for x in range(8)]
                self.queue.put(i + 1)
                self.queue.put(filters_for_queue)

        if headless_mode is True or not thread_stopEvent.is_set():
            print_msg("Optimization for %d epochs took %.3fs" % (epochs, time.time() - start_time), 2)

            # the entire rescale thing is not needed, bc optimization happens on a smaller image (for speed improvement)
            # real rescale is done during saving.
            original_tensor_transformed = transforms.ToTensor()(pil_image)

            final_filters = torch.zeros((8, original_tensor_transformed.shape[1], original_tensor_transformed.shape[2]),
                                        dtype=torch.float32).to(self.device)
            for k in range(8):
                if fixFilters:
                    if fixFilters[k] == 1:
                        final_filters[k, :, :] = initial_filter_values[k][1]
                    else:
                        final_filters[k, :, :] = self.filters.view(-1)[k]
                else:
                    final_filters[k, :, :] = self.filters.view(-1)[k]

            if not headless_mode:
                strings = ['Sat', 'Con', 'Bri', 'Sha', 'Hig', 'LLF', 'NLD', 'EXP']
                print_msg("Final Filter Intensities: {}".format(
                    [strings[k] + ": " + str(final_filters[k, 0, 0].item() * 100) for k in range(8)]), 3)
                self.queue.put([final_filters[k, 0, 0].item() for k in range(8)])

            mapped_img = torch.cat((original_tensor_transformed, final_filters.cpu()), dim=0).unsqueeze(dim=0).to(
                self.device)
            enhanced_img = self.can(mapped_img)
            enhanced_img = enhanced_img.cpu()
            enhanced_img = enhanced_img.detach().permute(2, 3, 1, 0).squeeze().numpy()
            enhanced_clipped = np.clip(enhanced_img, 0.0, 1.0) * 255.0
            enhanced_clipped = enhanced_clipped.astype('uint8')

            graph_data = {}
            if nima_vgg16:
                graph_data['nima_vgg16_scores'] = nima_vgg16_scores
                graph_data['nima_vgg16_losses'] = nima_vgg16_losses
            if nima_mobilenetv2:
                graph_data['nima_mobilenetv2_scores'] = nima_mobilenetv2_scores
                graph_data['nima_mobilenetv2_losses'] = nima_mobilenetv2_losses
            if ssmtpiaa:
                graph_data['ia_pre_scores'] = ia_pre_scores
                graph_data['ia_pre_losses'] = ia_pre_losses
            if ssmtpiaa_fine:
                graph_data['ia_fine_losses'] = ia_fine_losses
                graph_data['ia_fine_scores'] = ia_fine_scores

            if not headless_mode:
                self.queue.put(graph_data)

                self.queue.put(enhanced_clipped)
                self.in_queue = queue.Queue()

            # returns an 8bit image in any case ---
            if not headless_mode:
                return enhanced_clipped, None, None
            else:
                return enhanced_clipped, graph_data

    def update_optimizer(self, optim_lr):
        if config.optim == 'sgd':
            self.optimizer = torch.optim.SGD(params=[self.filters], lr=optim_lr, momentum=config.optim_momentum)
        elif config.optim == 'adam':
            self.optimizer = torch.optim.Adam(params=[self.filters], lr=optim_lr)
        else:
            error_callback('optimizer')

    def get_lr(self):
        for param_group in self.optimizer.param_groups:
            return param_group['lr']
import os
import cv2

import numpy as np
import matplotlib.pyplot as plt

import config


def make_animation(img, path, filename, export_video=True, resolution=300):
    frames = []
    for i in range(len(img)):
        fig = plt.figure(dpi=resolution)
        create_title(filename, img, i)
        plt.imshow(img[i])
        frames.append(get_openCV_image(fig))
    if export_video:
        convert_frames_to_video(frames, path, filename + '_animation.mp4')
    return frames


def make_animation_with_extra_info(img, graph_data: dict, path, filename, export_video=True, resolution=300):
    frames = []
    for i in range(len(img)):
        fig = plt.figure(dpi=resolution)
        create_title(filename, img, i)
        if config.assessor == 'NIMA_VGG16':
            plt.figtext(0.125, 0.035, "Score: " + " {:.4f}".format((graph_data['nima_vgg16_scores'][i] * 10)))
            plt.figtext(0.6, 0.035, "Loss: " + " {:.4f}".format(graph_data['nima_vgg16_losses'][i]))
        elif config.assessor == 'NIMA_mobilenetv2':
            plt.figtext(0.125, 0.035, "Score: " + " {:.4f}".format((graph_data['nima_mobilenetv2_scores'][i] * 10)))
            plt.figtext(0.6, 0.035, "Loss: " + " {:.4f}".format(graph_data['nima_mobilenetv2_losses'][i]))
        elif config.assessor == 'IA_pre':
            plt.figtext(0.125, 0.035, "Score: " + " {:.4f}".format((graph_data['ia_pre_scores'][i] * 10)))
            plt.figtext(0.6, 0.035, "Loss: " + " {:.4f}".format(graph_data['ia_pre_losses'][i]))
        elif config.assessor == 'IA_fine':
            plt.figtext(0.125, 0.035, "Score: " + " {:.4f}".format((graph_data['nima_mobilenetv2_scores'][i] * 10)))
            plt.figtext(0.6, 0.035, "Loss: " + " {:.4f}".format(graph_data['nima_mobilenetv2_losses'][i]))
        else:
            raise Exception("Unknown Asessor Network: " + config.assessor)
        frames.append(get_openCV_image(fig))
    if export_video:
        convert_frames_to_video(frames, path, filename + '_animation_with_extra_info.mp4')
    return frames


def make_full_info_animation(img, graph_data: dict, path, filename, resolution=300):
    editing_frames = make_animation_with_extra_info(img, graph_data, path, filename, export_video=False, resolution=resolution)
    loss_frames = get_animated_loss_graph_frames(graph_data, filename, resolution=int(resolution*0.5))
    score_frames = get_animated_score_graph_frames(graph_data, filename, resolution=int(resolution*0.5))
    graph_frames = stack_cv2video_frames_vertical(loss_frames, score_frames)
    print("Graph frames resolution and length:")
    print(graph_frames[0].shape)
    print(len(graph_frames))
    print("editing_frames frames resolution and length:")
    print(editing_frames[0].shape)
    print(len(editing_frames))
    final_frames = stack_cv2video_frames_horizontal(editing_frames, graph_frames)
    convert_frames_to_video(final_frames, path, filename + '_full_animation.mp4')


def make_graphs(graph_data: dict, path, filename):
    if config.save_loss_graph:
        plt.title(filename + " Losses")
        plt.plot(graph_data['nima_vgg16_losses'], label='NIMA_VGG16', color='blue')
        plt.plot(graph_data['nima_mobilenetv2_losses'], label='NIMA_mobilenetv2', color='cyan')
        plt.plot(graph_data['ia_pre_losses'], label='IA_pretrained', color='orange')
        plt.plot(graph_data['ia_fine_losses'], label='IA_finetuned', color='red')
        plt.xlabel('iterations')
        plt.ylabel('loss')
        plt.legend()
        plt.savefig(path + filename + '_losses', dpi=600)
        plt.close()

    if config.save_score_graph:
        plt.title(filename + " Scores")
        plt.plot(graph_data['nima_vgg16_scores'], label='NIMA_VGG16', color='blue')
        plt.plot(graph_data['nima_mobilenetv2_scores'], label='NIMA_mobilenetv2', color='cyan')
        plt.plot(graph_data['ia_pre_scores'], label='IA_pretrained', color='orange')
        plt.plot(graph_data['ia_fine_scores'], label='IA_finetuned', color='red')
        plt.xlabel('iterations')
        plt.ylabel('score')
        plt.legend()
        plt.savefig(path + filename + '_scores', dpi=600)
        plt.close()


def make_graph_animations(graph_data: dict, path, filename):
    if config.save_loss_graph:
        make_loss_animation(graph_data, path, filename)
    if config.save_score_graph:
        make_score_animation(graph_data, path, filename)


def make_loss_animation(graph_data: dict, path, filename):
    frames = get_animated_loss_graph_frames(graph_data, filename, 150)
    convert_frames_to_video(frames, path, filename + '_losses_animated.mp4')


def make_score_animation(graph_data: dict, path, filename):
    frames = get_animated_score_graph_frames(graph_data, filename, 150)
    convert_frames_to_video(frames, path, filename + '_scores_animated.mp4')


def create_title(filename, img, i):
    network_title = config.assessor
    if config.assessor == 'NIMA_VGG16':
        if config.legacy_NICER_loss_for_NIMA_VGG16:
            network_title += '_NICER_loss'
        else:
            network_title += '_MSE_loss'
    plt.title(network_title + " on " + filename.split('_')[0] + " iteration " + str(i + 1) + " out of " + str(len(img)))
    plt.imshow(img[i])


def get_openCV_image(fig):
    fig.canvas.draw()
    frame = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close()
    return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)


def get_animated_loss_graph_frames(graph_data: dict, filename, resolution = 150):
    frames = []
    length = len(graph_data['ia_fine_losses'])
    plt.plot(graph_data['nima_vgg16_losses'], label='NIMA_VGG16', color='blue')
    plt.plot(graph_data['nima_mobilenetv2_losses'], label='NIMA_mobilenetv2', color='cyan')
    plt.plot(graph_data['ia_pre_losses'], label='IA_pretrained', color='orange')
    plt.plot(graph_data['ia_fine_losses'], label='IA_finetuned', color='red')
    ylim = plt.gca().get_ylim()
    xlim = plt.gca().get_xlim()
    plt.close()

    for i in range(length):
        fig = plt.figure(dpi=resolution)
        plt.title(filename + " Losses")

        plt.plot(graph_data['nima_vgg16_losses'][:i+1], label='NIMA_VGG16', color='blue')
        plt.scatter(i, graph_data['nima_vgg16_losses'][i], color='blue')
        plt.plot(graph_data['nima_mobilenetv2_losses'][:i+1], label='NIMA_mobilenetv2', color='cyan')
        plt.scatter(i, graph_data['nima_mobilenetv2_losses'][i], color='cyan')
        plt.plot(graph_data['ia_pre_losses'][:i+1], label='IA_pretrained', color='orange')
        plt.scatter(i, graph_data['ia_pre_losses'][i], color='orange')
        plt.plot(graph_data['ia_fine_losses'][:i+1], label='IA_finetuned', color='red')
        plt.scatter(i, graph_data['ia_fine_losses'][i], color='red')

        plt.xlabel('iterations')
        plt.ylabel('loss')
        plt.legend(loc='upper right')
        plt.xlim(xlim)
        plt.ylim(ylim)
        frames.append(get_openCV_image(fig))

    return frames


def get_animated_score_graph_frames(graph_data: dict, filename, resolution = 150):
    frames = []
    length = len(graph_data['ia_fine_scores'])
    plt.plot(graph_data['nima_vgg16_scores'], label='NIMA_VGG16', color='blue')
    plt.plot(graph_data['nima_mobilenetv2_scores'], label='NIMA_mobilenetv2', color='cyan')
    plt.plot(graph_data['ia_pre_scores'], label='IA_pretrained', color='orange')
    plt.plot(graph_data['ia_fine_scores'], label='IA_finetuned', color='red')
    ylim = plt.gca().get_ylim()
    xlim = plt.gca().get_xlim()
    plt.close()

    for i in range(length):
        fig = plt.figure(dpi=resolution)
        plt.title(filename + " Scores")

        plt.plot(graph_data['nima_vgg16_scores'][:i+1], label='NIMA_VGG16', color='blue')
        plt.scatter(i, graph_data['nima_vgg16_scores'][i], color='blue')
        plt.plot(graph_data['nima_mobilenetv2_scores'][:i+1], label='NIMA_mobilenetv2', color='cyan')
        plt.scatter(i, graph_data['nima_mobilenetv2_scores'][i], color='cyan')
        plt.plot(graph_data['ia_pre_scores'][:i+1], label='IA_pretrained', color='orange')
        plt.scatter(i, graph_data['ia_pre_scores'][i], color='orange')
        plt.plot(graph_data['ia_fine_scores'][:i+1], label='IA_finetuned', color='red')
        plt.scatter(i, graph_data['ia_fine_scores'][i], color='red')

        plt.xlabel('iterations')
        plt.ylabel('score')
        plt.legend(loc='lower right')
        plt.xlim(xlim)
        plt.ylim(ylim)
        frames.append(get_openCV_image(fig))

    return frames


def stack_cv2video_frames_vertical(frames1, frames2):
    out_frames = []
    length = min(len(frames1), len(frames2))
    for i in range(length):
        out_frames.append(np.concatenate((frames1[i], frames2[i]), axis=0))
    return out_frames


def stack_cv2video_frames_horizontal(frames1, frames2):
    out_frames = []
    length = min(len(frames1), len(frames2))
    for i in range(length):
        out_frames.append(np.concatenate((frames1[i], frames2[i]), axis=1))
    return out_frames


def convert_frames_to_video(frames, path, output_name):
    wdir = os.getcwd()
    os.chdir(path)  # Change working directory to save directory for video.release() as it always saves to the correct working directory
    height, width, layers = frames[0].shape
    size = (width, height)
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    video = cv2.VideoWriter(output_name, fourcc, 10, size)
    for i in range(len(frames)):
        video.write(frames[i])
    video.release()
    os.chdir(wdir)
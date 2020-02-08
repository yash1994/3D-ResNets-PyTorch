from datasets.kinetics import Kinetics
from datasets.activitynet import ActivityNet
from datasets.ucf101 import UCF101
from datasets.hmdb51 import HMDB51
from datasets.custom import CustomNet


def get_training_set(opt, spatial_transform, temporal_transform,
                     target_transform):
    assert opt.dataset in ['kinetics', 'activitynet', 'ucf101', 'hmdb51', 'customnet']

    if opt.dataset == 'kinetics':
        training_data = Kinetics(
            opt.training_video_path,
            opt.training_annotation_path,
            'training',
            spatial_transform=spatial_transform,
            temporal_transform=temporal_transform,
            target_transform=target_transform)
    elif opt.dataset == 'activitynet':
        training_data = ActivityNet(
            opt.training_video_path,
            opt.training_annotation_path,
            'training',
            False,
            spatial_transform=spatial_transform,
            temporal_transform=temporal_transform,
            target_transform=target_transform)
    elif opt.dataset == 'ucf101':
        training_data = UCF101(
            opt.training_video_path,
            opt.training_annotation_path,
            'training',
            spatial_transform=spatial_transform,
            temporal_transform=temporal_transform,
            target_transform=target_transform)
    elif opt.dataset == 'hmdb51':
        training_data = HMDB51(
            opt.training_video_path,
            opt.training_annotation_path,
            'training',
            spatial_transform=spatial_transform,
            temporal_transform=temporal_transform,
            target_transform=target_transform)
    elif opt.dataset == 'customnet':
        training_data = CustomNet(
            opt.training_video_path,
            opt.training_annotation_path,
            'training',
            False,
            spatial_transform=spatial_transform,
            temporal_transform=temporal_transform,
            target_transform=target_transform)

    return training_data


def get_validation_set(opt, spatial_transform, temporal_transform,
                       target_transform, training_class_map):
    assert opt.dataset in ['kinetics', 'activitynet', 'ucf101', 'hmdb51', 'customnet']

    if opt.dataset == 'kinetics':
        validation_data = Kinetics(
            opt.validation_video_path,
            opt.validation_annotation_path,
            'validation',
            opt.n_val_samples,
            spatial_transform,
            temporal_transform,
            target_transform,
            sample_duration=opt.sample_duration)
    elif opt.dataset == 'activitynet':
        validation_data = ActivityNet(
            opt.validation_video_path,
            opt.validation_annotation_path,
            'validation',
            False,
            opt.n_val_samples,
            spatial_transform,
            temporal_transform,
            target_transform,
            sample_duration=opt.sample_duration)
    elif opt.dataset == 'ucf101':
        validation_data = UCF101(
            opt.validation_video_path,
            opt.validation_annotation_path,
            'validation',
            opt.n_val_samples,
            spatial_transform,
            temporal_transform,
            target_transform,
            sample_duration=opt.sample_duration)
    elif opt.dataset == 'hmdb51':
        validation_data = HMDB51(
            opt.validation_video_path,
            opt.validation_annotation_path,
            'validation',
            opt.n_val_samples,
            spatial_transform,
            temporal_transform,
            target_transform,
            sample_duration=opt.sample_duration)
    elif opt.dataset == 'customnet':
        validation_data = CustomNet(
            opt.validation_video_path,
            opt.validation_annotation_path,
            'validation',
            False,
            spatial_transform=spatial_transform,
            temporal_transform=temporal_transform,
            target_transform=target_transform,
            class_map=training_class_map)        
    return validation_data


def get_test_set(opt, spatial_transform, temporal_transform, target_transform):
    assert opt.dataset in ['kinetics', 'activitynet', 'ucf101', 'hmdb51']
    assert opt.test_subset in ['val', 'test']

    if opt.test_subset == 'val':
        subset = 'validation'
    elif opt.test_subset == 'test':
        subset = 'testing'
    if opt.dataset == 'kinetics':
        test_data = Kinetics(
            opt.testing_video_path,
            opt.annotation_path,
            subset,
            0,
            spatial_transform,
            temporal_transform,
            target_transform,
            sample_duration=opt.sample_duration)
    elif opt.dataset == 'activitynet':
        test_data = ActivityNet(
            opt.testing_video_path,
            opt.annotation_path,
            subset,
            True,
            0,
            spatial_transform,
            temporal_transform,
            target_transform,
            sample_duration=opt.sample_duration)
    elif opt.dataset == 'ucf101':
        test_data = UCF101(
            opt.testing_video_path,
            opt.annotation_path,
            subset,
            0,
            spatial_transform,
            temporal_transform,
            target_transform,
            sample_duration=opt.sample_duration)
    elif opt.dataset == 'hmdb51':
        test_data = HMDB51(
            opt.testing_video_path,
            opt.annotation_path,
            subset,
            0,
            spatial_transform,
            temporal_transform,
            target_transform,
            sample_duration=opt.sample_duration)

    return test_data

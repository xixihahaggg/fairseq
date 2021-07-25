from glob import glob
from pprint import pprint
from kaldiio import ReadHelper
import numpy as np
from tqdm import tqdm
import os
import bz2
import sys
import argparse
import pickle as pkl
from pathlib import Path
import cv2
import h5py
from torchvision import models, transforms
from glob import glob
import torch.nn as nn
import torch


def rewrite_scp(ARK_PATH):
    for i in glob(ARK_PATH + "/*.scp"):
        with open(i) as f:
            lines = f.readlines()
        if "ARK_PATH" in lines[0]:
            with open(i, "w") as f:
                for line in lines:
                    f.write(line.replace("ARK_PATH", ARK_PATH))
        else:
            print("Already reformat the scp files")
            continue


def write_utt2spk(ARK_PATH):
    spks = []
    with open(os.path.join(ARK_PATH, "cmvn_all_181506.scp")) as f:
        for line in f.readlines():
            spks.append(line.split()[0])
    utt2spk = {}
    for i in glob(ARK_PATH + "/*.scp"):
        with open(i) as f:
            lines = f.readlines()
            for line in lines:
                utt = line.split()[0]
                spk = utt[:11]
                if spk not in spks:
                    print(utt, spk)
                    print("{} not in speakar lists".format(spk))
                utt2spk[utt] = spk
    with open(os.path.join(ARK_PATH, "utt2spk"), "w") as f:
        for k, v in utt2spk.items():
            f.write(" ".join([k, v]) + "\n")


def audio_feature_reformat(ARK_PATH, out_dir, cmvn=False):
    if cmvn:
        out_dir = os.path.join(out_dir, "fbank_cmvn")
        os.makedirs(out_dir, exist_ok=True)
        for i in glob("{}/raw_fbank*.cmvn_ark".format(ARK_PATH)):
            with ReadHelper('ark:' + i) as reader:
                for key, numpy_array in tqdm(reader):
                    np.save(os.path.join(out_dir, '{}.npy'.format(key)), numpy_array)

    else:
        out_dir = os.path.join(out_dir, "raw_fbank")
        os.makedirs(out_dir, exist_ok=True)
        for i in glob("{}/raw_fbank*.cmvn_ark".format(ARK_PATH)):
            with ReadHelper('ark:' + i) as reader:
                for key, numpy_array in tqdm(reader):
                    np.save(os.path.join(out_dir, '{}.npy'.format(key)), numpy_array)


def audio_feature_prep(ARK_PATH="/home/xixihahaggg/Dataset/how2/fbank_pitch_181506",
                       out_dir="/home/xixihahaggg/Dataset/how2/",
                       cmvn=False):
    rewrite_scp(ARK_PATH)
    write_utt2spk(ARK_PATH)
    """
    Please use kaldi to apply cmvn
    """
    audio_feature_reformat(ARK_PATH, out_dir, cmvn)


def read_segments(data_dir):
    with open(os.path.join(data_dir, "segments")) as f:
        tmp = {}
        for line in f.readlines():
            t = line.strip().split()
            segment_id = t[0]
            case_id = segment_id[:11]
            if case_id not in tmp.keys():
                tmp[case_id] = {}
            tmp[case_id][segment_id] = {"audio": None, "video": None, "text": None}
    return tmp


def video_feature_extraction(model, transform, frames_path):
    img = []
    for path in glob("{}/*.jpg".format(frames_path)):
        img.append(transform(cv2.imread(path)))
    img = torch.stack(img, dim=0).cuda()
    feature = model(img).cpu().numpy()
    return feature


def write_file(data, output_path, suffix):
    print("write file")
    # pprint(data)
    dt1 = h5py.vlen_dtype(np.dtype('float64'))
    dt2 = h5py.special_dtype(vlen=str)
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    cnt = 0
    with h5py.File(os.path.join(output_path, "data_set_{}.h5py".format(str(suffix).zfill(3))), mode="w") as f:
        for segment_id, v in data.items():
            cnt += 1
            group = f.create_group(segment_id)
            group.create_dataset("Text", data=np.array([v["Text"]], dtype="S"))
            group.create_dataset('Audio', data=np.array(v["Audio"]))
            group.create_dataset('Video', data=np.array(v["Video"]))
        f.create_dataset("size", data=len(data))
    print("Successfully write file {}".format(output_path))


if __name__ == '__main__':
    data_dir = "/home/xixihahaggg/Dataset/how2/how2-300h-v1/data"
    audio_feature_dir = "/home/xixihahaggg/Dataset/how2/raw_fbank"
    audio_features = {i.split("/")[-1].replace(".npy", ""): i for i in glob("{}/*.npy".format(audio_feature_dir))}
    video_frame_dir = "/home/xixihahaggg/Dataset/how2/raw_video_frames_v2"
    raw_frames = {i.split("/")[-1]: i for i in glob("{}/*/*".format(video_frame_dir)) if os.path.isdir(i)}
    output_dir = "/home/xixihahaggg/PycharmProjects/fairseq/multimodal_pretraining/data"
    dic = {}
    for subdir in glob(data_dir + "/*"):
        suffix = subdir.split("/")[-1]
        dic[suffix] = {}
        with open(os.path.join(subdir, "text.id.en")) as f:
            for line in f.readlines():
                line = line.strip().split()
                segment_id = line[0]
                text = " ".join(line[1:])
                dic[suffix][segment_id] = {"Text": text.encode('ascii', 'ignore').decode('ascii'),
                                           "Audio": audio_features[segment_id],
                                           "Video": raw_frames[segment_id],
                                           }

    # for train_index, test_index in kf.split(keys):
    #     train_set = [keys[i] for i in train_index]
    #     test_set = [keys[i] for i in test_index]
    #     print(train_set, test_set)
    #     out_dir = "../medical_data/data/context_dependent/folder_{}/".format(cnt)
    #     if not os.path.exists(out_dir):
    #         os.makedirs(out_dir)
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.CenterCrop(512),
        transforms.Resize(224),
        transforms.ToTensor()
    ])
    model = models.resnet152(pretrained=True).cuda()
    model.fc = nn.Identity()
    model.eval()
    with torch.no_grad():
        chunk_size = 10000
        for dir, d in dic.items():
            data = {}
            # if dir == "train": continue
            suffix = 0
            cnt = 0
            for segment_id, v in tqdm(d.items()):
                if cnt == chunk_size:
                    write_file(data, output_path=os.path.join(output_dir, dir), suffix=suffix)
                    cnt = 0
                    suffix += 1
                    data = {}
                audio_feat = np.load(v["Audio"])
                video_feat = video_feature_extraction(model, transform, v["Video"])
                torch.cuda.empty_cache()
                data[segment_id] = {"Text": v["Text"], "Audio": audio_feat, "Video": video_feat}
                cnt += 1
            if len(data) != 0:
                write_file(data, output_path=os.path.join(output_dir, dir), suffix=suffix)

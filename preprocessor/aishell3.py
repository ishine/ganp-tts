#coding=utf-8
import sys
import os
import requests
import json
import time
from multiprocessing import Process
import traceback
import numpy as np
import librosa
from scipy.io import wavfile
from tqdm import tqdm

def run(process_method, process_num, data_list, in_dir, out_dir, sampling_rate, max_wav_value, dataset):
    print('process num: ', process_num)
    print('total data num: ', len(data_list))
    process_list = []

    eve_process_num = len(data_list) / process_num
    for i in range(process_num - 1):
        print(i * eve_process_num, (i + 1) * eve_process_num)
        process_list.append(
            Process(target=process_method, args=(data_list[int(i * eve_process_num): int((i + 1) * eve_process_num)], in_dir, out_dir, sampling_rate, max_wav_value, dataset)))
    i = process_num - 1
    process_list.append(Process(target=process_method, args=(data_list[int(i * eve_process_num):], in_dir, out_dir, sampling_rate, max_wav_value, dataset)))

    for process in process_list:
        process.start()
    process.join()

    # if Process.is_alive()
    process_run_num = 0
    for process in process_list:
        if process.is_alive():
            process_run_num += 1
    while process_run_num > 0:
        process_run_num = 0
        for process in process_list:
            if process.is_alive():
                process_run_num += 1
        # print "process_run_num\t" + str(process_run_num)
        time.sleep(0.1)


def proc_data(lines, in_dir, out_dir, sampling_rate, max_wav_value, dataset):
    for line in tqdm(lines):
        wav_name, text = line.strip("\n").split("\t")
        speaker = wav_name[:7]
        text = text.split(" ")[1::2]
        wav_path = os.path.join(in_dir, dataset, "wav", speaker, wav_name)
        if os.path.exists(wav_path):
            os.makedirs(os.path.join(out_dir, speaker), exist_ok=True)
            wav, _ = librosa.load(wav_path, sampling_rate)
            wav = wav / max(abs(wav)) * max_wav_value
            wavfile.write(
                os.path.join(out_dir, speaker, wav_name),
                sampling_rate,
                wav.astype(np.int16),
            )
            with open(
                os.path.join(out_dir, speaker, "{}.lab".format(wav_name[:11])),
                "w",
            ) as f1:
                f1.write(" ".join(text))


def prepare_align(config):
    in_dir = config["path"]["corpus_path"]
    out_dir = config["path"]["raw_path"]
    sampling_rate = config["preprocessing"]["audio"]["sampling_rate"]
    max_wav_value = config["preprocessing"]["audio"]["max_wav_value"]
    for dataset in ["train", "test"]:
        print("Processing {}ing set...".format(dataset))
        with open(os.path.join(in_dir, dataset, "content.txt"), encoding="utf-8") as f:
            lines = f.readlines()
            run(proc_data, 40, lines, in_dir, out_dir, sampling_rate, max_wav_value, dataset)
        


   
if __name__ == "__main__":
    import time
    start_time=time.time()
    #day_text()
    whois_run()
    #merge_data()
    print(time.time()-start_time)


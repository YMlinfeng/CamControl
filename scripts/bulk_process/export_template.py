import os
import sys
import pickle
import json
import csv
import time
import random
from functools import partial
from abc import abstractmethod
from dataclasses import dataclass

import lmdb
import zlib
import torch
import pypeln as pl
from loguru import logger
import pandas as pd
from tqdm import tqdm

@dataclass
class MultiprocessorConfig:
    csv_path: str = "/video/zhengmingwu/m2v-diffusers/middle/m2v-video-s1-v0.1-4090-part3.csv"
    tmp_dir: str = "/home/taoxin/Kwai/m2v/batch_video_3dvae"
    save_dir: str = "/home/taoxin/Kwai/m2v/batch_video_3dvae"
    task_chunk_size: int = 10000
    num_workers: int = 1
    max_sleeptime: int = 30
    debug: bool = False

class Multiprocessor:
    def __init__(self, config) -> None:
        logger.remove()
        logger.add(sys.stderr, format='{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}', level='INFO')

        # set proxy and other environ
        self.set_environ()

        self.config = config
        self.csv_path = config.csv_path
        self.tmp_dir = config.tmp_dir
        self.save_dir = config.save_dir

        
        os.makedirs(self.tmp_dir, exist_ok=True)

        # set path names
        self.csvpath_func = lambda task_id: os.path.join(self.tmp_dir, 'task-{:08d}.csv'.format(task_id))
        self.savecsvpath_func = lambda task_id: os.path.join(self.tmp_dir, 'savecsv-{:08d}.csv'.format(task_id))
        self.progpath_func = lambda task_id: os.path.join(self.tmp_dir, 'processing-{:08d}.txt'.format(task_id))


        # task-specific consts
        self.task_chunk_size = config.task_chunk_size
        self.num_workers = config.num_workers
        
    @staticmethod
    def set_environ():
        os.environ["http_proxy"] = "http://oversea-squid1.jp.txyun:11080"
        os.environ["https_proxy"] = "http://oversea-squid1.jp.txyun:11080"
        os.environ["no_proxy"] = "localhost,127.0.0.1,localaddress,localdomain.com,internal,corp.kuaishou.com,test.gifshow.com,staging.kuaishou.com"
        os.environ["TORCH_HOME"] = "/group/ckpt/torchhub"
        os.environ["HF_DATASETS_CACHE"] = "/video/cache/huggingface"
        os.environ["HF_DATASETS_OFFLINE"] = "1"


    def split_tasks(self):
        csv_path = self.csv_path
        tmp_dir = self.tmp_dir
        TASK_CHUNK_SIZE = self.task_chunk_size

        if not os.path.exists(csv_path):
            logger.error(f'File [{csv_path}] not exist.')
        if not os.path.exists(tmp_dir):
            os.makedirs(tmp_dir)

        total_tasks_path = os.path.join(tmp_dir, 'total_tasks.txt')

        # avoid clash
        if not self.config.debug:
            time.sleep(random.randint(1, self.config.max_sleeptime)) 

        if os.path.exists(total_tasks_path):
            while True:
                with open(total_tasks_path, 'rt') as f:
                    total_tasks = int(f.read().strip())
                    if total_tasks >= 0:
                        return total_tasks
                logger.info('Waiting task splitting...')    
                time.sleep(2)
        else:
            logger.info(f'Splitting tasks: {total_tasks_path}')
            with open(total_tasks_path, 'wt') as f:
                f.write('-1\n')

        task_id = -1
        data_id = -1
        with open(csv_path, newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            header = next(reader)
            for row in reader:
                data_id += 1
                # start a new task file
                if data_id == 0:
                    task_id += 1
                    if self.config.debug and task_id > 9: break # for debug
                    csvfid = open(self.csvpath_func(task_id), 'w', newline='')
                    writer = csv.writer(csvfid, delimiter=',')
                    writer.writerow(header) 

                writer.writerow(row)
                
                # close file until TASK_CHUNK_SIZE
                if data_id == TASK_CHUNK_SIZE - 1:
                    data_id = -1
                    if 'csvfid' in locals().keys(): csvfid.close()

        total_tasks = task_id
        with open(total_tasks_path, 'wt') as f:
            f.write(f'{total_tasks}\n')
        return total_tasks


    def get_task_id(self, TOTAL_TASKS):
        task_id = 0

        while task_id < TOTAL_TASKS:
            # avoid clash: avoid reading the same file
            time.sleep(random.random())
            if not os.path.exists(self.progpath_func(task_id)):
                with open(self.progpath_func(task_id), 'wt') as f:
                    f.write(f'\n')
                break
            else:
                task_id += 1
                
        if task_id >= TOTAL_TASKS:
            return -1
        return task_id


    @logger.catch(onerror=lambda e: logger.error('Failed to load input: {}', e.__traceback__.tb_frame.f_locals['args'][0]))
    def stage_load_input(self, task_id):
        logger.info(f"Start task: [{task_id}]")
        # vid_list = pd.read_csv(vid_csv, nrows=n_data, )['video_ceph_path']

        cvspath = self.csvpath_func(task_id)

        with open(cvspath, newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            header = next(reader)
            for row in reader:
                # return multiple paths, and next stage need to use flat_map
                yield row[0] # bug
        

    @logger.catch(onerror=lambda e: logger.error('Failed to process: {}', e.__traceback__.tb_frame.f_locals['args'][0]))
    def stage_process(self, video_path):
        basename = os.path.basename(video_path)
        logger.info(f"- Process: {video_path}")

        time.sleep(0.1)

        output_data = 0
        self.savepath_func = lambda video_path: (
            os.makedirs(os.path.join(self.save_dir, os.path.dirname(video_path)[1:]), exist_ok=True),
            os.path.join(self.save_dir, os.path.dirname(video_path)[1:], f'{os.path.basename(video_path)}.pt')
        )[-1]
        output_path = self.savepath_func(video_path)

        return [output_data, output_path]

    @logger.catch(onerror=lambda e: logger.error('Failed to output: {}', e.__traceback__.tb_frame.f_locals['args'][0]))
    def stage_save_output(self, out, writer):
        [output_data, output_path] = out
        # logger.info(f"Save output: {output_path}")
        
        writer.writerow(['m2v', output_path])
        
        return output_path


    def start_task_runner(self):
        # set up loggers
        logger.add(f'{self.tmp_dir}/log_error.log', format='{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}', level='ERROR',
                    rotation='1 day', compression='zip')
        logger.add(f'{self.tmp_dir}/log_debug.log', format='{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}', level='DEBUG',
                    rotation='1 day', compression='zip')

        # split tasks if not ready
        TOTAL_TASKS = self.split_tasks()
        logger.info(f'Total tasks [{TOTAL_TASKS}] with each task [{self.task_chunk_size}] data.')

        n_workers = 2

        task_id = self.get_task_id(TOTAL_TASKS)
        while task_id > -1:
            with open(self.savecsvpath_func(task_id), 'w', newline='') as csvfid:
                writer = csv.writer(csvfid, delimiter=',')
                writer.writerow(['name', '3dvae_latents_path'])
                results = (
                    [task_id]
                    | pl.thread.flat_map(self.stage_load_input, workers=n_workers, maxsize=50)
                    | pl.process.map(self.stage_process, workers=1, maxsize=30)
                    | pl.thread.map(partial(self.stage_save_output, writer=writer), workers=n_workers, maxsize=50)
                    | list
                )
            logger.info(f'Finish task [{task_id}]')
            task_id = self.get_task_id(TOTAL_TASKS)


    def start_mp_runner(self, n_workers=5):
        # this if for debug
        from multiprocessing import Process
        os.system(f'rm -rf {self.tmp_dir}/task* {self.tmp_dir}/processing* {self.tmp_dir}/total_tasks.txt {self.tmp_dir}/*.log')
        logger.info(f'Start multiprocess with [{n_workers}] processes.')

        # start_task_runner()
        pool = []
        for i in range(n_workers):
            p = Process(target=self.start_task_runner)
            p.start()
            pool.append(p)
        for p in pool:
            p.join()
        return 0


    def export_csv2lmdb(self, csv_path, lmdb_path):
        if not os.path.exists(csv_path):
            logger.error(f"Cannot find {csv_path}")

        data_list = pd.read_csv(csv_path, nrows=1000)

        data_list.to_csv(self.tmp_dir + '/temp.txt', index=False)

        env = lmdb.open(lmdb_path, map_size=1099511627776)
        with env.begin(write=True) as txn:
            for idx in range(len(data_list)):
                # data_torch = torch.load(f'vae_compress/vae_{i}.pt', map_location='cpu')
                data = ','.join(data_list.iloc[idx].to_list())
                # compressed = data.tobytes()
                compressed = zlib.compress(data.encode('utf-8'))
                # compressed = zlib.compress(json.dumps(data).encode('utf-8'), level=9)
                # compressed = json.dumps(data).encode('utf-8')
                if idx < 10:
                    print(len(data), len(compressed), data)
                txn.put(str(idx).encode(), data.encode('utf-8'))
        env.close()


if __name__ == "__main__":
    # export_csv2lmdb(csv_path, os.path.join(tmp_dir, 'm2v-video-s1-v0.1-4090-part3.csv'))

    # uncomment this line for online processing
    # start_task_runner()
    config = MultiprocessorConfig(
        csv_path="/video/zhengmingwu/m2v-diffusers/inputs/video2latents_240.csv",
        tmp_dir="/video/zhengmingwu/tmp",
        save_dir="/video/zhengmingwu/tmp",
        debug=True,
        task_chunk_size=10,
    )
    processor = Multiprocessor(config)
    processor.start_task_runner()
    # simulate distributed runner, debug only
    # start_mp_runner(n_workers=5)
    


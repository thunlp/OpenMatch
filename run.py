import os

devices = "3"
nproc_per_node = len(devices.split(","))
port = 12345
model = "t5-base"
task = "classification"
gradient_accumulation = 4
fold = 0
dev = "../ReInfoSelect_Testing_Data/robust04/fold_{}/rb04_dev.jsonl".format(fold)
qrels = "../ReInfoSelect_Testing_Data/robust04/rb04_qrels"
metric = "ndcg_cut_20"
per_gpu_train_batch_size = 8
per_gpu_eval_batch_size = 64
learning_rate = 2e-5
distributed = False
maxp = False


class Config:
    def __init__(self, train_file, task, model: str, rate, logging_step, eval_every, epoch, gradient_accumulation) -> None:
        self.train_file = train_file
        self.out_trec = os.path.join("results", "{}_{}_{}_maxp_{}.trec".format(model, task, rate, maxp))
        self.tb_dir = os.path.join("logs", "{}_{}_{}_maxp_{}".format(model, task, rate, maxp))
        self.logging_step = logging_step
        self.eval_every = eval_every
        self.save = os.path.join("checkpoints", "{}_{}_{}_maxp_{}".format(model, task, rate, maxp))
        self.epoch = epoch
        self.task = task
        self.model = model[0:model.index("-")]
        self.pretrained_model = os.path.join("../pretrained_models", model)
        self.gradient_accumulation = gradient_accumulation
        self.command = None
        if distributed:
            self.command = f"CUDA_VISIBLE_DEVICES={devices} python -u -m torch.distributed.launch --nproc_per_node={nproc_per_node} --master_port={port} train.py  -task {self.task}  -model {self.model}  -train {self.train_file}  -max_input 1280000  -save {self.save}  -dev {dev}  -qrels {qrels}  -vocab {self.pretrained_model} -pretrain {self.pretrained_model} -res {self.out_trec} -metric {metric} -max_query_len 20  -max_doc_len 489  -epoch {self.epoch} -batch_size {per_gpu_train_batch_size} -lr {learning_rate} -eval_every {self.eval_every} -optimizer adamw  -dev_eval_batch_size {per_gpu_eval_batch_size} -n_warmup_steps 0  -logging_step {self.logging_step} --log_dir {self.tb_dir} -gradient_accumulation_steps {self.gradient_accumulation}"
        else:
            self.command = f"CUDA_VISIBLE_DEVICES={devices} python train.py  -task {self.task}  -model {self.model}  -train {self.train_file}  -max_input 1280000  -save {self.save}  -dev {dev}  -qrels {qrels}  -vocab {self.pretrained_model} -pretrain {self.pretrained_model} -res {self.out_trec} -metric {metric} -max_query_len 20  -max_doc_len 489  -epoch {self.epoch} -batch_size {per_gpu_train_batch_size} -lr {learning_rate} -eval_every {self.eval_every} -optimizer adamw  -dev_eval_batch_size {per_gpu_eval_batch_size} -n_warmup_steps 0  -logging_step {self.logging_step} --log_dir {self.tb_dir} -gradient_accumulation_steps {self.gradient_accumulation}"
        
        if maxp:
            self.command += " -maxp"

    def run(self):
        with os.popen(self.command) as f:
            f.read()

    def print_command(self):
        print(self.command)

config1 = Config(train_file="../ReInfoSelect_Testing_Data/robust04/fold_{}/rb04_train_classification_sample_0.002.jsonl".format(fold),
                 task=task,
                 model=model,
                 rate=0.002,
                 logging_step=10,
                 eval_every=10,
                 epoch=5,
                 gradient_accumulation=gradient_accumulation)

config2 = Config(train_file="../ReInfoSelect_Testing_Data/robust04/fold_{}/rb04_train_classification_sample_0.02.jsonl".format(fold),
                 task=task,
                 model=model,
                 rate=0.02,
                 logging_step=20,
                 eval_every=100,
                 epoch=5,
                 gradient_accumulation=gradient_accumulation)

config3 = Config(train_file="../ReInfoSelect_Testing_Data/robust04/fold_{}/rb04_train_classification_sample_0.2.jsonl".format(fold),
                 task=task,
                 model=model,
                 rate=0.2,
                 logging_step=100,
                 eval_every=1000,
                 epoch=2,
                 gradient_accumulation=gradient_accumulation)

config4 = Config(train_file="../ReInfoSelect_Testing_Data/robust04/fold_{}/rb04_train_classification.jsonl".format(fold),
                 task=task,
                 model=model,
                 rate=1,
                 logging_step=100,
                 eval_every=1000,
                 epoch=2,
                 gradient_accumulation=gradient_accumulation)

config1.run()
config2.run()
config3.run()
config4.run()
# config1.print_command()
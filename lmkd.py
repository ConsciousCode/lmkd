#!/usr/bin/env python3
'''
Language Model Knowledge Distillation - LemMe Know Dat

Generic end to end causal autoregressive language model knowledge distillation.
'''

import os
import sys
from abc import ABC, abstractmethod
from typing import Optional, Protocol, Mapping, Final
import importlib
import json
from functools import cached_property

############################
## Configuration defaults ##
############################

# Need to load this as soon as possible
DEBUG_FALSE = {"0", "n", "no", "f", "false"}
DEBUG: Final = (lambda d:
	d is None or d.lower() not in DEBUG_FALSE
)(os.environ.get("DEBUG", "true"))

MAX_SEQ_LEN: Final = 1024 # Need some default just in case

STUDENT_PATH: Final = "student.pickle"
CACHE_PATH: Final = "dataloader.cache"
CHECKPOINT_PATH: Final = "checkpoints/"
DISTILL_PATH: Final = "distill.py"

SEED: Final = hash(os.urandom(4))
PRECISION: Final = "medium"
LEARNING_RATE: Final = 1e-4
TEMPERATURE: Final = 1.0
SAVE_STEPS: Final = 1000
LOG_STEPS: Final = 8
BATCHES: Final = 4 # 8 exhausts my 12 GB VRAM with a mere 12 layers
BATCH_SIZE: Final = None # None = max_seq_len from config
EPOCHS: Final = 3
WORKERS: Final = os.cpu_count()

DEVICE_NAME: Final = None # None = CUDA if available, else CPU

def expandpath(path):
	return os.path.abspath(os.path.expanduser(str(path)))

class TransformerLike(Protocol):
	'''
	Protocol for HuggingFace model-like objects and what we need from them.
	'''
	
	def __call__(self, *, input_ids, attention_mask): ...

class Distill(ABC):
	'''Abstract base class for distillation code.'''
	
	debug: bool
	device: str
	max_seq_len: Optional[int]
	
	def __init__(self, **options):
		self.debug = options['debug']
		self.device = options['device']
		self.options = options
	
	def dprint(self, *args, **kwargs):
		if self.debug:
			print(*args, **kwargs)
	
	@cached_property
	def max_seq_len(self): return MAX_SEQ_LEN
	
	@abstractmethod
	def teacher(self) -> str|TransformerLike:
		'''Build or load the teacher model. If it returns a string, it's a HuggingFace model.'''
		pass
	
	@abstractmethod
	def student(self, state_dict: Optional[Mapping]=None) -> str|TransformerLike:
		'''Build the student model. If it returns a string, it's a HuggingFace model.'''
		pass
	
	@abstractmethod
	def dataset(self, split: str) -> str|Mapping[str, list[str]]:
		'''Load the dataset. If it returns a string, it's a HuggingFace dataset.'''
		pass

def build_app(debug):
	'''
	Keep application logic in a separate function because the imports take an
	exceptionally long time to load
	'''
	
	def dprint(*args, **kwargs):
		if debug:
			print(*args, **kwargs)

	# Prints to help my impatient ass know it's not dead
	dprint("import torch")
	import torch
	import torch.optim as optim
	import torch.nn as nn
	import torch.nn.functional as F
	from torch.utils.data import DataLoader
	from torch.nn.utils.rnn import pad_sequence

	dprint("import lightning")
	import lightning as pl
	from lightning.pytorch.callbacks import LearningRateMonitor, Callback

	dprint("import transformers")
	from transformers.models.auto import AutoConfig, AutoModelForCausalLM, AutoTokenizer

	dprint("import datasets")
	from datasets import load_dataset

	class FrequentCheckpoint(Callback):
		'''Checkpoint mid-epoch, since they can last a long time.'''
		
		def __init__(self, debug: bool, save_steps: int, output_dir: str):
			super().__init__()
			self.debug = debug
			self.save_steps = save_steps
			self.output_dir = output_dir

		def on_batch_end(self, trainer, pl_module):
			global_step = trainer.global_step
			if global_step % self.save_steps == 0:
				ckpt_path = os.path.join(self.output_dir, f"cp{global_step}.ckpt")
				trainer.save_checkpoint(ckpt_path)
				
				if self.debug:
					print(f"Checkpoint saved at step {global_step}: {ckpt_path}")

	class DistillApp(pl.LightningModule):
		'''Application logic.'''
		
		def __init__(self, *,
				debug=debug,
				
				distill=None,
				student=None,
				cache=None,
				checkpoints=None,
				
				seed=None,
				precision=None,
				temperature=None,
				learning_rate=None,
				save_steps=None,
				log_steps=None,
				epochs=None,
				batches=None,
				batch_size=None,
				workers=None,
				device=None,
				options=None
			):
			super().__init__()
			
			if os.environ.get("TOKENIZERS_PARALLELISM") is None:
				os.environ["TOKENIZERS_PARALLELISM"] = "true"
			
			# These are the only attrs that we know before importing the distill module
			
			self.debug = debug
			self.distill_path = expandpath(distill)
			
			# Everything else just stores the config from the command line
			
			# Paths and names
			self.student_path = student
			self.cache_path = cache
			self.checkpoint_path = checkpoints
			
			# Training hyperparameters
			self.seed = seed
			self.precision = precision
			self.temperature = temperature
			self.learning_rate = learning_rate
			self.save_steps = save_steps
			self.log_steps = log_steps
			self.epochs = epochs
			self.batches = batches
			self.batch_size = batch_size
			self.workers = workers
			self.device_name = device
			self.options = options
			
			self.dataset_cache = {}
			self.options = None
			self.distill = None
			self.lmkd = None
			
			# Model components
			self.student = None
			self.teacher = None
			self.tokenizer = None
			self.distill_loss = nn.KLDivLoss(reduction="batchmean")
			self.ce_loss = nn.CrossEntropyLoss()

		def dprint(self, *args):
			if self.debug:
				print(*args)
		
		def param(self, key):
			ku = key.upper()
			return (
				getattr(self, key) or os.environ.get(ku) or
				getattr(self.distill, ku, None) or globals()[ku]
			)
		
		def unpack_config(self):
			'''Synthesize config from CLI, distill module, and defaults.'''
			
			debug = self.param("debug")
			if isinstance(debug, str):
				debug = debug.lower() not in DEBUG_FALSE
			self.debug = bool(debug)
			
			self.student_path = expandpath(self.param("student_path"))
			self.cache_path = expandpath(self.param("cache_path"))
			self.checkpoint_path = expandpath(self.param("checkpoint_path"))
			
			# Training hyperparameters
			self.seed = hash(self.param("seed"))
			precision = str(self.param("precision")).lower()
			if precision not in {"highest", "high", "medium"}:
				self.dprint(f"Invalid precision {precision!r}, defaulting to medium")
				precision = PRECISION
			self.precision = precision
			
			self.temperature = float(self.param("temperature"))
			self.learning_rate = float(self.param("learning_rate"))
			self.save_steps = int(self.param("save_steps"))
			self.log_steps = int(self.param("log_steps"))
			self.epochs = int(self.param("epochs"))
			self.batches = int(self.param("batches"))
			self.batch_size = int(
				self.param("batch_size") or
				getattr(self.lmkd, "max_seq_len", None) or
				MAX_SEQ_LEN
			)
			self.workers = int(self.param("workers"))
			self.device_name = str(
				self.param() or
				"cuda" if torch.cuda.is_available() else "cpu"
			)
			
			self.options = dict(
				debug=self.debug,
				device=self.device_name,
				**self.options
			)

		def training_step(self, batch, batch_idx):
			input_ids, attention_mask = batch
			input_ids = input_ids.reshape(self.batch_size, -1)
			attention_mask = attention_mask.reshape(self.batch_size, -1)
			
			#input_ids = torch.stack(input_ids, dim=0)
			#attention_mask = torch.stack(attention_mask, dim=0)
			
			# Teacher model output
			with torch.no_grad():
				teacher_logits = self.teacher(
					input_ids=input_ids,
					attention_mask=attention_mask
				).logits

			# Student model output
			student_logits = self.student(
				input_ids=input_ids,
				attention_mask=attention_mask
			).logits
			
			# Calculate distillation loss
			distill_loss = self.distill_loss(
				F.log_softmax(student_logits / self.temperature, dim=-1),
				F.softmax(teacher_logits / self.temperature, dim=-1)
			)
			student_logits = student_logits.view(-1, student_logits.shape[-1])
			teacher_logits = teacher_logits.view(-1, teacher_logits.shape[-1])
			input_ids = input_ids.view(-1)
			
			student_loss = self.ce_loss(student_logits, input_ids)
			teacher_loss = self.ce_loss(teacher_logits, input_ids)
			
			loss = distill_loss + student_loss
			
			self.log("train_loss/combined", loss)
			self.log("train_loss/distill", distill_loss)
			self.log("train_loss/ce_teacher", teacher_loss)
			self.log("train_loss/ce_student", student_loss)
			
			return loss

		def configure_optimizers(self):
			return optim.Adam(self.student.parameters(), lr=self.learning_rate)
		
		def _collate(self, batch):
			'''
			Collates the inputs as (seq*batch,) tensors because of weirdness with
			how torch handles parallelism, which can exhaust file descriptors:
			https://github.com/pytorch/pytorch/issues/65198
			'''
			
			input_ids = []
			attention_mask = []
			
			for item in batch:
				tok = self.tokenizer(item["text"])
				input_ids.append(tok['input_ids'].squeeze())
				attention_mask.append(tok['attention_mask'].squeeze())
			
			input_ids = pad_sequence(input_ids, batch_first=True)
			attention_mask = pad_sequence(attention_mask, batch_first=True, padding_value=0)
			
			return input_ids, attention_mask
		
		def _dataloader(self, split):
			dataset = self.lmkd.dataset(split)
			if isinstance(dataset, str):
				data = self.dataset_cache.get(dataset)
				if data is None:
					data = load_dataset(dataset)
					self.dataset_cache[dataset] = data
				dataset = data
			
			return DataLoader(
				dataset[split],
				batch_size=self.batch_size,
				num_workers=self.workers,
				collate_fn=self._collate
			)
		
		def train_dataloader(self): return self._dataloader("train")
		def val_dataloader(self): return self._dataloader("validation")
		def test_dataloader(self): return self._dataloader("test")

		def main(self):
			self.dprint("Importing distiller...")
			spec = importlib.util.spec_from_file_location(
				"__lmkd_distill", self.distill_path
			)
			distill = self.distill = importlib.util.module_from_spec(spec)
			sys.path.append(os.path.dirname(self.distill_path))
			sys.modules["__lmkd_distill"] = distill
			spec.loader.exec_module(distill)
			
			# Now that the module is imported, we can synthesize the config
			self.unpack_config()
			
			torch.set_float32_matmul_precision(self.precision)
			pl.seed_everything(self.seed)
			
			self.dprint("Building distiller...")
			lmkd = self.lmkd = distill.LMKD(**self.options)
			
			# Build/load the student, possibly from a file
			if os.path.exists(self.student_path):
				self.dprint("Loading student..")
				student_state = torch.load(self.student_path)
				student = lmkd.student(student_state)
				
				# Load structure from HuggingFace
				if isinstance(student, str):
					config = AutoConfig.from_pretrained(student)
					student = AutoModelForCausalLM.from_config(config)
					student.load_state_dict(student_state)
			else:
				self.dprint("Building student...")
				student = lmkd.student()
				
				# Load pretrained model as initial weights from HuggingFace
				if isinstance(student, str):
					student = AutoModelForCausalLM.from_pretrained(student)
				
				self.dprint("Saving student...")
				torch.save(student.state_dict(), self.student_path)
			
			self.student = student.to(self.device_name)
			
			self.dprint("Loading teacher...")
			teacher = lmkd.teacher()
			if isinstance(teacher, str):
				teacher = AutoModelForCausalLM.from_pretrained(teacher)
			self.teacher = teacher.to(self.device_name)
			
			self.dprint("Loading tokenizer...")
			tokenizer = lmkd.tokenizer()
			if isinstance(tokenizer, str):
				tokenizer = AutoTokenizer.from_pretrained(tokenizer)
			self.tokenizer = tokenizer
			
			self.dprint("Begin training")
			
			trainer = pl.Trainer(
				accelerator="auto",
				max_epochs=self.epochs,
				log_every_n_steps=self.log_steps,
				callbacks=[
					LearningRateMonitor(logging_interval='step'),
					FrequentCheckpoint(
						debug=self.debug,
						save_steps=self.save_steps,
						output_dir=self.checkpoint_path
					)
				]
			)
			trainer.fit(self)
	
	return DistillApp

def build_argparse():
	from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter, MetavarTypeHelpFormatter
	from contextlib import contextmanager
	
	class CombinedFormatter(ArgumentDefaultsHelpFormatter, MetavarTypeHelpFormatter):
		def _lmkd_metavar(self, action):
			'''Fixes a bug: choices without type doesn't use the metavar, but action.type is None.'''
			if getattr(self, "_lmkd_dont_use_type", False):
				return action.dest
			
			mv = action.type
			return action.dest if mv is None else mv.__name__
	
		_get_default_metavar_for_optional = _lmkd_metavar
		_get_default_metavar_for_positional = _lmkd_metavar
		
		@contextmanager
		def _lmkd_suppress_type(self):
			'''Suppresses the type from being shown in the help string.'''
			self._lmkd_dont_use_type = True
			yield
			self._lmkd_dont_use_type = False
		
		def _get_help_string(self, action):
			# Suppress default=None as well.
			if action.default is not None:
				return super()._get_help_string(action)
			return action.help
		
		def _format_action_invocation(self, action):
			# Only show the metavar once, not for each option string.
			with self._lmkd_suppress_type():
				if action.option_strings:
					default = self._lmkd_metavar(action)
					args_string = self._format_args(action, default)
					return f"{', '.join(action.option_strings)} {args_string}"
				else:
					return super()._format_action_invocation(action)
		
		def _format_actions_usage(self, actions, groups):
			# Nasty minimally invasive hack to fix the metavar for usage.
			with self._lmkd_suppress_type():
				return super()._format_actions_usage(actions, groups)
	
	ap = ArgumentParser(
		description="End to end knowledge distillation of a causal autoregressive model",
		formatter_class=CombinedFormatter
	)
	ap.add_argument("distill", type=str, default=DISTILL_PATH, help="Python file defining how to distill the model", nargs="?")
	ap.add_argument("--cache", type=str, default=CACHE_PATH, help="dataset cache file")
	ap.add_argument("--checkpoints", type=str, default=CHECKPOINT_PATH, help="checkpoint directory")
	
	ap.add_argument("-D", "--debug", action="store_true", default=DEBUG, help="debug mode")
	ap.add_argument("-o", "--options", type=str, help='comma-separated list of key=json entries passed to distill code (default: "")')
	ap.add_argument("-T", "--temperature", type=float, default=TEMPERATURE, help="temperature for KL Div loss")
	ap.add_argument("-S", "--seed", type=hash, help="random seed (default: urandom)")
	ap.add_argument("-p", "--precision", choices=['highest', 'high', 'medium'], default=PRECISION, help="Torch matmul precision")
	ap.add_argument("-s", "--save-steps", type=int, default=SAVE_STEPS, help="number of steps between checkpoints")
	ap.add_argument("-l", "--log-steps", type=int, default=LOG_STEPS, help="number of steps between log messages")
	ap.add_argument("-e", "--epochs", type=int, default=EPOCHS, help="number of epochs")
	ap.add_argument("-b", "--batches", type=int, default=BATCHES, help="number of batches to perform at once")
	ap.add_argument("-B", "--batch-size", type=int, help="batch size (default: max_seq_len from distill code)")
	ap.add_argument("-w", "--workers", type=int, default=WORKERS, help="number of workers for dataloader")
	ap.add_argument("-d", "--device", type=str, help="device to use for the student (default: 'cuda' if available)")
	
	return ap

def main(argv=None):
	'''Composable entry'''
	
	if argv is None:
		import sys
		argv = sys.argv[1:]
	
	args = build_argparse().parse_args(argv)
	options = {}
	if args.options is not None:
		for opt in args.options.split(","):
			kv = opt.split("=", 1)
			if len(kv) == 1:
				kv = kv[0].strip()
				if kv == "":
					continue
				k, v = kv, True
			else:
				k, v = kv
				k, v = k.strip(), json.loads(v)
			options[k] = v
	
	args.options = options
	
	# Automatically extract distill.py from the stack.
	if __name__ != "__main__":
		import inspect
		args.distill = inspect.stack()[-2].filename
	
	DistillApp = build_app(args.debug)
	DistillApp(**vars(args)).main()

if __name__ == "__main__":
	main()
# LemMe Know Dat
(Language Model Knowledge Distillation)

This is a CLI/library for generic end to end knowledge distillation of causal autoregressive language models. Just make a `distill.py` file, write a `LMKD` class, and run.

The purpose of this is to simplify knowledge distillation as much as possible, so more effort can be put into building the student model and preparing the teacher model. It handles the full suite of configurations, model loading/saving, logging, losses, training, testing, validation, dataset caching, and checkpointing. Thus the only things which need to be done are defining the models and dataset.

## Knowledge Distillation
End to end knowledge distillation is a method of training a student model to mimic a teacher model. It is done by training the student model on the teacher model's output logits, taking the teacher model's predictions, applying a temperature, and using the KL divergence loss between the teacher's predictions and the student's predictions. Typically it's used to compress a larger model into a smaller model while maintaining performance, but it can also be used to train a model of a different architecture. This uses significantly less compute than training a model from scratch, and can be done with a smaller dataset.

## Usage
`example.py` is a sample from another project of mine which I made this library for. The student has its weights cloned from a teacher, then the FF layers replaced with a custom layer which it needs to learn to use. Knowledge distillation in this case is perfect because it allows the student to extract the teacher's latent knowledge into the custom layers, which it would otherwise have to relearn from scratch.

For ease of use, you can add the following at the bottom of your file:
```python
if __name__ == "__main__":
	lmkd.main()
```
This handles the CLI arguments and calls your `LMKD` class without needing to use a separate program.

## Configurations
You can override these in your `distill.py` file, export in the environment, or pass them as CLI arguments, in decreasing order of precedence.

* `DEBUG` - Whether to use debug mode
* `STUDENT_PATH` - Path to the student model file
* `CACHE_PATH` - Path to the dataset cache file
* `CHECKPOINT_PATH` - Path to the checkpoint directory
* `SEED` - Random seed (uses the hash)
* `PRECISION` - Torch matmul precision {highest, high, medium}
* `LEARNING_RATE` - Learning rate
* `TEMPERATURE` - Temperature for KL Div loss
* `SAVE_STEPS` - Number of steps between checkpoints
* `LOG_STEPS` - Number of steps between log messages
* `BATCHES` - Number of batches to perform at once
* `BATCH_SIZE` - Batch size
* `EPOCHS` - Number of epochs
* `WORKERS` - Number of workers for dataloader
* `DEVICE_NAME` - Device to use

## LMKD Class
In your `distill.py` file, you need to define a `LMKD` class. `Distill` is provided as a base class and reference:
```python
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
	
	def __init__(self, debug, device, **options):
		self.debug = debug
		self.device = device
	
	def dprint(self, *args, **kwargs):
		if self.debug:
			print(*args, **kwargs)
	
	@cached_property
	def max_seq_len(self): return 1024
	
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
```

`teacher`, `student`, and `dataset` can return either a string or an object. If it's a string, it's a HuggingFace model or dataset which will be automatically loaded. If it's an object, it's a model that implements the `TransformerLike` protocol or a dataset. The dataset will be automatically cached to the `CACHE_PATH` and the model will be automatically saved to `STUDENT_PATH` if it doesn't exist, as well as `CHECKPOINT_PATH` at the end of each epoch.

## CLI Help
```
usage: lmkd.py [-h] [--cache cache] [--checkpoints checkpoints] [-D] [-o options] [-T temperature]
               [-S seed] [-p {highest,high,medium}] [-s save_steps] [-l log_steps] [-e epochs]
               [-b batches] [-B batch_size] [-w workers] [-d device]
               [distill]

End to end knowledge distillation of a causal autoregressive model

positional arguments:
  distill               Python file defining how to distill the model (default: distill.py)

options:
  -h, --help            show this help message and exit
  --cache cache         dataset cache file (default: dataloader.cache)
  --checkpoints checkpoints
                        checkpoint directory (default: checkpoints/)
  -D, --debug           debug mode (default: True)
  -o, --options options
                        comma-separated list of key=json entries passed to distill code (default: "")
  -T, --temperature temperature
                        temperature for KL Div loss (default: 1.0)
  -S, --seed seed       random seed (default: urandom)
  -p, --precision {highest,high,medium}
                        Torch matmul precision (default: medium)
  -s, --save-steps save_steps
                        number of steps between checkpoints (default: 1000)
  -l, --log-steps log_steps
                        number of steps between log messages (default: 8)
  -e, --epochs epochs   number of epochs (default: 3)
  -b, --batches batches
                        number of batches to perform at once (default: 4)
  -B, --batch-size batch_size
                        batch size (default: max_seq_len from distill code)
  -w, --workers workers
                        number of workers for dataloader (default: 8)
  -d, --device device   device to use for the student (default: 'cuda' if available)
```
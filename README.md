The original README here(./README2.md)


### Changes in this repository
- Introducing pruning to the model

The below will do pruning on the whole model

```python
import tensorflow_model_optimization as tfmot 
from tensorflow_model_optimization.python.core.sparsity.keras import pruning_schedule as pruning_sched

prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude

# Compute end step to finish pruning after 2 epochs.
batch_size = 8
epochs = 2
validation_split = 0.1 # 10% of training set will be used for validation set. 

num_images = 40000
end_step = 800


#'''
#Defining pruning parameters

pruning_params = {
      'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(initial_sparsity=0.50,
                                                               final_sparsity=0.80,
                                                               begin_step=0,
                                                               end_step=1000)
}

#'''

model_for_pruning = tfmot.sparsity.keras.prune_low_magnitude(model, **pruning_params)

model_for_pruning.compile(optimizer=adam, loss=ssd_loss.compute_loss)
#model_for_pruning.compile(optimizer=adam, loss='sparse_categorical_crossentropy')

model_for_pruning.summary()
#'''
```

Below will apply pruning to only Conv2D layers

```python
from tensorflow_model_optimization.python.core.sparsity.keras import pruning_schedule as pruning_sched

#'''
def apply_pruning_to_dense(layer):
  if isinstance(layer, tf.keras.layers.Conv2D):
    return tfmot.sparsity.keras.prune_low_magnitude(layer,pruning_schedule=pruning_sched.ConstantSparsity(0.5, 0))
  return layer
#'''
'''
def apply_pruning_to_dense(layer):
  if isinstance(layer, tf.keras.layers.Conv2D):
    return tfmot.sparsity.keras.prune_low_magnitude(layer,pruning_schedule=pruning_sched.PolynomialDecay(initial_sparsity=0.2,
        final_sparsity=0.8, begin_step=1000, end_step=2000))
  return layer
'''

# Use `tf.keras.models.clone_model` to apply `apply_pruning_to_dense` 
# to the layers of the model.
model_for_pruning = tf.keras.models.clone_model(
    model,
    clone_function=apply_pruning_to_dense,
)
```



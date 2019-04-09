# utils
## check_point.py
* instantiate a CheckPoint
Note: will automatically add History obj **defined before** into checkpoint
```python
ckpter = CheckPoint(model=model, optimizer=optimizer,
    path='./ckpt', prefix='Run01', interval=2, save_num=1)
```
* Explicitly bind history to save train/val history
```python
# best practice is to new CheckPoint before new History
# Which will automatically add History obj into checkpoint

ckpter.bind_histories([train_hist, val_hist])
```
* call .chech_on() on each epoch
```python
for epoch in range(10):
    ...
    ckpter.check_on(epoch=epoch, monitor='loss', loss_acc=val_loss_acc)
```
## history.py
a class to hold and plot acc&loss history
* instantiate a History
```python
his = History(name='train')
```
* call `.add()` method
```python
logs = {'loss': 1e-1, 'acc': 0.2}
his.add(epoch=1, log)
```
* plot train/val history
```python
his.plot()
```

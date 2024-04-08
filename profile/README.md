Nitrain is the easiest way to train medical imaging AI models locally or in the cloud using any of your favorite frameworks.

Fitting an image-to-image segmentation model in the cloud is as easy as this:

```python
dataset = FolderDataset('~/Desktop/kaggle-liver-ct',
                        x={'pattern': 'volumes/volume-{id}.nii'},
                        y={'pattern': 'segmentations/segmentation-{id}.nii'})

loader = DatasetLoader(dataset,
                       images_per_batch=2, 
                       sampler=SliceSampler(batch_size=12))


model_fn = models.fetch_architecture('unet', dim=2)
model = model_fn((128,128,1))

trainer = PlatformTrainer(model=model, task='image_to_image')
trainer.fit(loader, epochs=5)
```
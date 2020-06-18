from PIL import Image


im = Image.open('data/train-volume.tif')
for i in range(im.n_frames):
    im.seek(i)
    im.save('data/train/image/train%d.png' % i)
    
im = Image.open('data/test-volume.tif')
for i in range(im.n_frames):
    im.seek(i)
    im.save('data/test/test%d.png' % i)
    
im = Image.open('data/train-labels.tif')
for i in range(im.n_frames):
    im.seek(i)
    im.save('data/train/label/train-labels%d.png' % i)
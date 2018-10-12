import numpy as np

def randomHorizontalFlip(image, mask, u=0.5):
    if np.random.random() < u:
        image = np.flip(image, axis=(1))
        mask = np.flip(mask, axis=(1))           
    return image, mask

def randomVerticalFlip(image, mask, u=0.5):
    if np.random.random() < u:
        image = np.flip(image, axis=(2))
        mask = np.flip(mask, axis=(2))
    return image, mask

def randomRotation(image, mask, u=0.5):
    n_rand = np.random.random()
    if n_rand < u:
        if n_rand <= u/3:
            k=1
        elif n_rand > u - u/3:
            k=3
        else:
            k=2
        image = np.rot90(image, k=k, axes=(1,2))
        mask = np.rot90(mask, k=k, axes=(1,2))
    return image, mask

def BatchGenerator(trainX, trainY, batch_size, augment = True):
    while True:
        for start in range(0, len(trainX), batch_size):
            x_batch = []
            y_batch = []
            end = min(start + batch_size, len(trainX))
            for id in range(start,end):
                img = trainX[id]
                mask = trainY[id]
                # augment data by random horizontal and vertical flip
                if augment:
                    img, mask = randomHorizontalFlip(img, mask)
                    img, mask = randomVerticalFlip(img, mask)
                    img, mask = randomRotation(img, mask)
                x_batch.append(img)
                y_batch.append(mask)
            x_batch = np.array(x_batch)
            y_batch = np.array(y_batch)
            yield x_batch, y_batch

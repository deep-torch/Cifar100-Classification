import torch


# Custom function to map class index to category
def get_category_mapping(trainset):
    categories = [
        ['beaver', 'dolphin', 'otter', 'seal', 'whale'],
        ['aquarium_fish', 'flatfish', 'ray', 'shark', 'trout'],
        ['orchid', 'poppy', 'rose', 'sunflower', 'tulip'],
        ['bottle', 'bowl', 'can', 'cup', 'plate'],
        ['apple', 'mushroom', 'orange', 'pear', 'sweet_pepper'],
        ['clock', 'keyboard', 'lamp', 'telephone', 'television'],
        ['bed', 'chair', 'couch', 'table', 'wardrobe'],
        ['bee', 'beetle', 'butterfly', 'caterpillar', 'cockroach'],
        ['bear', 'leopard', 'lion', 'tiger', 'wolf'],
        ['bridge', 'castle', 'house', 'road', 'skyscraper'],
        ['cloud', 'forest', 'mountain', 'plain', 'sea'],
        ['camel', 'cattle', 'chimpanzee', 'elephant', 'kangaroo'],
        ['fox', 'porcupine', 'possum', 'raccoon', 'skunk'],
        ['crab', 'lobster', 'snail', 'spider', 'worm'],
        ['baby', 'boy', 'girl', 'man', 'woman'],
        ['crocodile', 'dinosaur', 'lizard', 'snake', 'turtle'],
        ['hamster', 'mouse', 'rabbit', 'shrew', 'squirrel'],
        ['maple_tree', 'oak_tree', 'palm_tree', 'pine_tree', 'willow_tree'],
        ['bicycle', 'bus', 'motorcycle', 'pickup_truck', 'train'],
        ['lawn_mower', 'rocket', 'streetcar', 'tank', 'tractor']
    ]
    class_to_category = {}
    for category_idx, category in enumerate(categories):
        for cls in category:
            cls_idx = trainset.class_to_idx[cls]
            class_to_category[cls_idx] = category_idx
            
    return class_to_category


def save_checkpoint(model, optimizer, epoch, file_path):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    torch.save(checkpoint, file_path)
    print(f"Checkpoint saved at {file_path}")


def load_checkpoint(model, optimizer, file_path):
    checkpoint = torch.load(file_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    print(f"Checkpoint loaded from {file_path}, starting from epoch {epoch+1}")

    return epoch

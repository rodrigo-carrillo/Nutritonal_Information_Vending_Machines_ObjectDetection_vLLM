import os
import ultralytics


# Create the YOLO_Models directory if it doesn't exist
yolo_home = '/Users/rodrigocarrillo/Documents/Computer Vision/Vending Machines/02_Models'
os.makedirs(yolo_home, exist_ok=True)

# Set YOLO_HOME BEFORE importing YOLO - this controls where models are cached
os.environ['YOLO_HOME'] = yolo_home

# Change to the writable directory
os.chdir(yolo_home)

# Get original model.
model = ultralytics.YOLO("yolo11n.pt")

# Train the model with specified parameters.
results = model.train(
    data = '/Users/rodrigocarrillo/Documents/Computer Vision/Vending Machines/HoloSelecta-FinalDataset_YOLO/data.yaml',
    epochs = 100,
    imgsz = 640,
    batch = 8,
    seed = 42,        # Random seed
    patience = 10,    # Early stopping
    save = True,
    verbose = True,
    project = yolo_home,
    save_period = 5,  # Save checkpoint every 5 epochs
    workers = 8,
    pretrained = True,
    name = 'yolo11n_HoloSelecta',

    optimizer = 'AdamW',    # Better than SGD for most tasks - try 'Adam', 'SGD', or 'AdamW'
    lr0 = 0.001,            # initial learning rate (reduced from 0.01 for better convergence)
    lrf = 0.01,             # final learning rate ratio (lr = lr0 * lrf at end of training)
    momentum = 0.937,       # momentum for SGD (ignored if using Adam)
    weight_decay = 0.0005,  # L2 regularization to prevent overfitting
    cos_lr = True,          # use cosine learning rate scheduler for smoother convergence
    warmup_epochs = 10,     # gradual warmup of learning rate in first 10 epochs
)
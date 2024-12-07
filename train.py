from Parser import FruitClassificationParser
from simple_cnn import SimpleCNN,main,update_best_config
from nets import AlexNet,ResNet50,ResNet101,ResNet152

NUM_CLASSES=5
parser=FruitClassificationParser()
args=parser.parse_args()

model_class=globals()[args.model_name]
model=model_class(NUM_CLASSES)


test_acc, val_acc = main(args,model)

results = {
    "Epochs": args.epochs,
    "Learning rate": args.lr,
    "Batch Size": args.batch_size,
    "Test Accuracy": test_acc,
    "Validation Accuracy": val_acc,
}

# Update the best configuration
update_best_config(results,filename=f"{args.model_name}.json")
 
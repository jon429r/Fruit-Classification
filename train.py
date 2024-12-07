from Parser import FruitClassificationParser
from simple_cnn import SimpleCNN,main
from nets import AlexNet,ResNet50,ResNet101,ResNet152

NUM_CLASSES=5
parser=FruitClassificationParser()
args=parser.parse_args()

model_class=globals()[args.model_name]
model=model_class(5)
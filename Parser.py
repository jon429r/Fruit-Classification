import argparse

class FruitClassificationParser:
    def __init__(self):
        self.parser = argparse.ArgumentParser(
            description="Fruit Image Classification with Simple CNN"
        )
        self._add_arguments()

    def _add_arguments(self):
        
        # Model name
        self.parser.add_argument(
            "--model_name",
            type=str,
            default="SimpleCNN",
            choices=["SimpleCNN", "ResNet50","ResNet101","ResNet142", "AlexNet"],
            help="Name of the model architecture to use -> Default: SimpleCNN",
            required=False
        )
        
        # Training hyperparameters
        self.parser.add_argument(
            "--epochs", 
            type=int, 
            default=10, 
            help="Number of training epochs -> Default: 10"
        )
        self.parser.add_argument(
            "--lr", 
            type=float, 
            default=0.0001, 
            help="Learning rate -> Default: 0.0001"
        )
        self.parser.add_argument(
            "--batch_size", 
            type=int, 
            default=32, 
            help="Batch size for training -> Default: 32"
        )

        # Flags
        self.parser.add_argument(
            "--train", 
            action="store_true", 
            help="Enable training mode -> Default: False"
        )
        self.parser.add_argument(
            "--save", 
            action="store_true", 
            help="Save the model -> Default: False"
        )
        self.parser.add_argument(
            "--graph",
            nargs='?',
            const=True,
            default=False,
            type=str,
            help="Plot the graphs. If a path is provided, saves the graphs to that location -> Default: False"
        )
        self.parser.add_argument(
            "--conf_matrix",
            type=str,
            default=None,
            help="Path to save the confusion matrix plot. If not provided, the matrix won't be saved -> Default: None"
        )

    def parse_args(self):
        """Parse and return command line arguments."""
        return self.parser.parse_args()
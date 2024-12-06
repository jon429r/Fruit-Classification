import argparse

class FruitClassificationParser:
    def __init__(self):
        self.parser = argparse.ArgumentParser(
            description="Fruit Image Classification with Simple CNN"
        )
        self._add_arguments()

    def _add_arguments(self):
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
            action="store_true", 
            help="Plot the graph -> Default: False"
        )

    def parse_args(self):
        """Parse and return command line arguments."""
        return self.parser.parse_args()
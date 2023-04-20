import nn
import numpy as np


class PerceptronModel(object):
    def __init__(self, dim):
        """
        Initialize a new Perceptron instance.

        A perceptron classifies data points as either belonging to a particular
        class (+1) or not (-1). `dim` is the dimensionality of the data.
        For example, dim=2 would mean that the perceptron must classify
        2D points.
        """
        self.w = nn.Parameter(1, dim)

    def get_weights(self):
        """
        Return a Parameter instance with the current weights of the perceptron.
        """
        return self.w

    def run(self, x_point):
        """
        Calculates the score assigned by the perceptron to a data point x.

        Inputs:
            x_point: a node with shape (1 x dimensions)
        Returns: a node containing a single number (the score)
        """
        # Returning an nn.DotProduct object.
        return nn.DotProduct(self.get_weights(), x_point)

    def get_prediction(self, x_point):
        """
        Calculates the predicted class for a single data point `x_point`.

        Returns: -1 or 1
        """
        # Calculates the predicted class
        pred = nn.as_scalar(self.run(x_point))
        # Returning 1 if the predicted class is non-negative, or -1 otherwise.
        if pred >= 0:
            return 1
        else:
            return -1

    def train_model(self, dataset):
        """
        Train the perceptron until convergence.
        """
        # Setting the batch size for the dataset.
        batch_size = 1
        # Setting a prerequisite of the while loop to True.
        boolean = True
        while boolean:
            # Setting the prerequisite of the while loop to False to continue.
            boolean = False
            for x, y in dataset.iterate_once(batch_size):
                # Calculates the predicted class.
                example = self.get_prediction(x)
                # If the example is misclassified.
                if example != nn.as_scalar(y):
                    # The multiplier argument is a Python scalar.
                    multiplier = nn.as_scalar(y)
                    # The direction argument is a Node with the
                    # same shape as the parameter.
                    direction = x
                    # Update the weights using update method of
                    # nn.Parameter class.
                    self.w.update(multiplier, direction)
                    # Setting the prerequisite of the while loop to True.
                    boolean = True


class RegressionModel(object):
    """
    A neural network model for approximating a function that maps from real
    numbers to real numbers. The network should be sufficiently large to be able
    to approximate sin(x) on the interval [-2pi, 2pi] to reasonable precision.
    """

    def __init__(self):
        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"
        self.rate_of_learning = 0.1  # the learning rate
        self.hidden_layers = []  # use for the hidden layer
        self.sizes_for_each_layers = [250, 250, 250]
        # three layers, each of size 250

    def run(self, x):
        """
        Runs the model for a batch of examples.
        Inputs:
            x: a node with shape (batch_size x 1)
        Returns:
            A node with shape (batch_size x 1) containing predicted y-values
        """
        "*** YOUR CODE HERE ***"
        predict_y_value = x
        # looping each items and get it prediction
        for i in range(len(self.hidden_layers)):
            first = nn.Linear(predict_y_value, self.hidden_layers[i][0])
            second = nn.AddBias(first, self.hidden_layers[i][1])
            third = nn.Linear(nn.ReLU(second), self.hidden_layers[i][2])
            predict_y_value = nn.AddBias(third, self.hidden_layers[i][3])
        return predict_y_value

    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.
        Inputs:
            x: a node with shape (batch_size x 1)
            y: a node with shape (batch_size x 1), containing the true y-values
                to be used for training
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"
        return nn.SquareLoss(self.run(x), y)

    def train_model(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        boolean = True
        batch_size = int(0.1 * dataset.x.shape[0])
        while len(dataset.x) % batch_size != 0:
            # evenly divisible by batch size
            batch_size += 1

        # get hidden layers
        for i in range(3):
            layers = [
                nn.Parameter(dataset.x.shape[1], self.sizes_for_each_layers[i]),
                nn.Parameter(1, self.sizes_for_each_layers[i]),
                nn.Parameter(self.sizes_for_each_layers[i], dataset.x.shape[1]),
                nn.Parameter(1, 1)]
            self.hidden_layers.append(layers)

        while boolean:
            losses_list = []
            for x_value, y_value in dataset.iterate_once(batch_size):
                param_list = []
                for items in self.hidden_layers:
                    # list of parameters from hidden layers
                    for sub_item in items:
                        param_list.append(sub_item)
                multiplier = nn.gradients(param_list,
                                          self.get_loss(x_value, y_value))
                for i in range(len(param_list)):
                    params = param_list[i]
                    params.update(-self.rate_of_learning, multiplier[i])
                losses_list.append(
                    nn.as_scalar(self.get_loss(x_value, y_value)))

            if np.mean(losses_list) < 0.001:  # the threshold and the test
                boolean = False


class DigitClassificationModel(object):
    """
    A model for handwritten digit classification using the MNIST dataset.

    Each handwritten digit is a 28x28 pixel grayscale image, which is flattened
    into a 784-dimensional vector for the purposes of this model. Each entry in
    the vector is a floating point number between 0 and 1.

    The goal is to sort each digit into one of 10 classes (number 0 through 9).

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """

    def __init__(self):
        # Initialize your model (hyper)parameters here
        "*** YOUR CODE HERE ***"
        self.learning_rate = .125  # [0.0001, 1] (usually around 0.1?)
        # rate = 0.1 -> 12 mins
        # rate = 0.2 -> MUCH faster run, but accuracy ~97.2%
        # while 0.1 can go up to 98% with the right adjustments
        # rate = 0.15 -> 5 mins, 97.6%! (2 layers)
        #                7 mins, 97.5% (3 layers)  BUT sometimes diverges
        # rate = 0.125 -> 7-10m (on a slower laptop, 5m on a fast one)
        #                       -> 97.52% (OK)

        self.batch_seg_mult = .005
        # testing size:
        # 1/200th of dataset per batch - 6m (ok)
        # 1/100th - 12+ mins (X)
        # 1/10th - 8+ (X)

        self.va_threshold = 0.976
        # stop on validation accuracy threshold  (97.5% - 98%)
        # - 98% - 20 mins!! (X)
        # - 97.6% - 6 mins (ok)

        self.hidden_layers = [
            [
                nn.Parameter(784, 150),
                nn.Parameter(1, 150),
                nn.Parameter(150, 784),
                nn.Parameter(1, 784)
            ],
            [
                nn.Parameter(784, 125),
                nn.Parameter(1, 125),
                nn.Parameter(125, 784),
                nn.Parameter(1, 784)
            ],

            # 2 layers - faster, overall accuracy 97.7%
            # 3 layers - deeper, w/0.125 rate - 97.5%
            [
                nn.Parameter(784, 100),
                nn.Parameter(1, 100),
                nn.Parameter(100, 10),
                nn.Parameter(1, 10)
            ],
        ]   # 3 layers
        # layer size = 250 -> 11+ mins (X)
        # 200 -> consistently hit 97.8% at 6 mins,
        # but h_l size too large -> takes too long to finish? (X)
        # 175-> 98%+! but 23 mins (X)
        # 150 -> 9 mins, 97.22%  (ok)

        # 50 apart - too long, size 50 (X)
        # 25 apart - 6 mins! (ok)

    def run(self, x):
        """
        Runs the model for a batch of examples.
        Your model should predict a node with shape (batch_size x 10),
        containing scores. Higher scores correspond to greater probability of
        the image belonging to a particular class.
        Inputs:
            x: a node with shape (batch_size x 784)
        Output:
            A node with shape (batch_size x 10) containing predicted scores
                (also called logits)
        """
        "*** YOUR CODE HERE ***"
        y = x
        for i in range(len(self.hidden_layers)-1):
            # no ReLU for last layer - we'll do it separately
            layer = self.hidden_layers[i]
            first = nn.Linear(y, layer[0])
            second = nn.AddBias(first, layer[1])
            non_lin = nn.ReLU(second)
            # adding non-linearity with ReLU
            third = nn.Linear(non_lin, layer[2])
            y = nn.AddBias(third, layer[3])

        # last layer without ReLU
        last = len(self.hidden_layers) - 1
        layer = self.hidden_layers[last]
        first = nn.Linear(y, layer[0])
        second = nn.AddBias(first, layer[1])
        third = nn.Linear(second, layer[2])
        y = nn.AddBias(third, layer[3])

        return y

    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.
        The correct labels `y` are represented as a node with shape
        (batch_size x 10). Each row is a one-hot vector encoding the correct
        digit class (0-9).
        Inputs:
            x: a node with shape (batch_size x 784)
            y: a node with shape (batch_size x 10)
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"
        return nn.SoftmaxLoss(self.run(x), y)

    def train_model(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        boolean = True
        batch_size = int(self.batch_seg_mult * dataset.x.shape[0])
        # batch size = 1/2000th of dataset

        while len(dataset.x) % batch_size != 0:
            # evenly divisible by batch size
            batch_size += 1

        while boolean:
            for x, y in dataset.iterate_once(batch_size):
                parameters = list()  # list of parameters from hidden layers
                for layer in self.hidden_layers:
                    for param in layer:
                        parameters.append(param)

                gradients = nn.gradients(parameters, self.get_loss(x, y))
                # wrt m, b (params)

                for i in range(len(parameters)):
                    parameters[i].update(-self.learning_rate, gradients[i])

            # test stopping condition
            if dataset.get_validation_accuracy() > self.va_threshold:
                # achieved threshold accuracy?
                boolean = False


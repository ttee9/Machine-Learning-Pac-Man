from game import *
from learningAgents import ReinforcementAgent
from featureExtractors import *

import random,util,math

class QLearningAgent(ReinforcementAgent):
    """
      Q-Learning Agent

      Functions you should fill in:
        - computeValueFromQValues
        - computeActionFromQValues
        - getQValue
        - getAction
        - update

      Instance variables you have access to
        - self.epsilon (exploration prob)
        - self.alpha (learning rate)
        - self.discount (discount rate)

      Functions you should use
        - self.getLegalActions(state)
          which returns legal actions for a state
    """

    def __init__(self, **args):
        "You can initialize Q-values here..."
        ReinforcementAgent.__init__(self, **args)
        "*** YOUR CODE HERE ***"
        # Initialize values with util.Counter()
        self.values = util.Counter()


    def getQValue(self, state, action):
        """
          Returns Q(state,action)
          Should return 0.0 if we have never seen a state or the Q node value otherwise
        """
        "*** YOUR CODE HERE ***"
        # Return value with (state, action) as key
        return self.values[(state, action)]

    def computeValueFromQValues(self, state):
        """
          Returns max_action Q(state,action) where the max is over legal actions.
          Note that if there are no legal actions, which is the case at the terminal state,
          you should return a value of 0.0.
        """
        "*** YOUR CODE HERE ***"
        allActions = self.getLegalActions(state)
        # No actions available
        if len(allActions) == 0:
            return float(0)
        valueOutput = float("-inf")
        # Go through every action
        for action in allActions:
            # Output highest value
            if valueOutput <= self.getQValue(state, action):
                valueOutput = self.getQValue(state, action)
        return valueOutput

    def computeActionFromQValues(self, state):
        """
          Compute the best action to take in a state.
          Note that if there are no legal actions, which is the case at the terminal state,
          you should return None.
        """
        "*** YOUR CODE HERE ***"
        allActions = self.getLegalActions(state)
        # No actions available
        if len(allActions) == 0:
            return None
        maxValue = float("-inf")
        actionOutput = None
        # Go through every action
        for action in allActions:
            # Set actionOutput to be the action with the highest Q value
            if maxValue <= self.getQValue(state, action):
                maxValue = self.getQValue(state, action)
                actionOutput = action
        return actionOutput

    def getAction(self, state):
        """
          Compute the action to take in the current state.
          With probability self.epsilon, we should take a random action and take the best policy action otherwise.
          Note that if there are no legal actions, which is the case at the terminal state,
          you should choose None as the action.

          HINT: You might want to use util.flipCoin(prob)
          HINT: To pick randomly from a list, use random.choice(list)
        """
        # Pick Action
        "*** YOUR CODE HERE ***"
        legalActions = self.getLegalActions(state)
        action = None
        if len(legalActions) == 0:
            return None
        # For positive epsilon, explore random action
        if util.flipCoin(self.epsilon):
            action = random.choice(legalActions)
        # Otherwise perform action based on highest Q value
        else:
            action = self.computeActionFromQValues(state)
        return action

    def update(self, state, action, nextState, reward):
        """
          The parent class calls this to observe a state = action => nextState and reward transition.
          You should do your Q-Value update here

          NOTE: You should never call this function, it will be called on your behalf
        """
        "*** YOUR CODE HERE ***"
        # Alpha = Learning rate
        # def computeValueFromQValues(self, state):
        # Values is a dictionary with tuple (state, action) as a key
        # Set value to updated value with current state and new state
        self.values[(state, action)] = ((1 - self.alpha) * self.getQValue(state, action)) + (self.alpha * (reward + self.discount * self.computeValueFromQValues(nextState)))

    def getPolicy(self, state):
        return self.computeActionFromQValues(state)

    def getValue(self, state):
        return self.computeValueFromQValues(state)

class PacmanQAgent(QLearningAgent):
    "Exactly the same as QLearningAgent, but with different default parameters"

    def __init__(self, epsilon=0.05,gamma=0.8,alpha=0.2, numTraining=0, **args):
        """
        These default parameters can be changed from the pacman.py command line.
        For example, to change the exploration rate, try:
            python pacman.py -p PacmanQLearningAgent -a epsilon=0.1
        alpha    - learning rate
        epsilon  - exploration rate
        gamma    - discount factor
        numTraining - number of training episodes, i.e. no learning after these many episodes
        """
        args['epsilon'] = epsilon
        args['gamma'] = gamma
        args['alpha'] = alpha
        args['numTraining'] = numTraining
        self.index = 0  # This is always Pacman
        QLearningAgent.__init__(self, **args)

    def getAction(self, state):
        """
        Simply calls the getAction method of QLearningAgent and then informs parent of action for Pacman.
        Do not change or remove this method.
        """
        action = QLearningAgent.getAction(self, state)
        self.doAction(state, action)
        return action

class ApproximateQAgent(PacmanQAgent):
    """
       ApproximateQLearningAgent
       You should only have to overwrite getQValue and update.
       All other QLearningAgent functions should work as is.
    """
    def __init__(self, extractor='IdentityExtractor', **args):
        self.featExtractor = util.lookup(extractor, globals())()
        PacmanQAgent.__init__(self, **args)

        # You might want to initialize weights here.
        "*** YOUR CODE HERE ***"
        self.weights = util.Counter()

    def getWeights(self):
        return self.weights

    def getQValue(self, state, action):
        """
          Should return Q(state,action) = w * featureVector
          where * is the dotProduct operator
        """
        "*** YOUR CODE HERE ***"
        qValue = 0
        allFeatures = self.featExtractor.getFeatures(state, action)
        # Approximate Q value is based on the weights of each feature
        for f in allFeatures:
            qValue += allFeatures[f] * self.weights[f]
        return qValue

    def update(self, state, action, nextState, reward):
        """
           Should update your weights based on transition
        """
        "*** YOUR CODE HERE ***"
        allFeatures = self.featExtractor.getFeatures(state, action)
        # If newState has greater value than current state, then difference is positive. Positive difference causes self.weights[f] to increase
        difference = (reward + (self.discount * self.getValue(nextState))) - self.getQValue(state, action)
        for f in allFeatures:
            self.weights[f] = self.weights[f] + (difference * self.alpha * allFeatures[f])

    def final(self, state):
        "Called at the end of each game."
        # call the super-class final method
        PacmanQAgent.final(self, state)

        # did we finish training?
        if self.episodesSoFar == self.numTraining:
            # you might want to print your weights here for debugging
            "*** YOUR CODE HERE ***"
            pass

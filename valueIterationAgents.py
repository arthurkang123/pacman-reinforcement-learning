import mdp, util

from learningAgents import ValueEstimationAgent
import collections

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        # Write value iteration code here
        for itr in range(self.iterations):
          uCounter = util.Counter()
          for state in self.mdp.getStates():
            val = -float("inf")
            for action in self.mdp.getPossibleActions(state):
              val = max(self.computeQValueFromValues(state, action), val)
              uCounter[state] = val
          self.values = uCounter
        return None

    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """

        result = 0
        for elem1, elem2 in self.mdp.getTransitionStatesAndProbs(state, action):
          result += elem2 * (self.mdp.getReward(state, action, elem1) + \
            self.discount * self.getValue(elem1))
        return result

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """

        if self.mdp.isTerminal(state):
          return None
        val = -float('inf')
        for elem in self.mdp.getPossibleActions(state):
          temp = self.computeQValueFromValues(state, elem)
          if (temp >= val):
            result = elem
            val = temp
        return result


    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)

class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        for i in range(self.iterations):
          if (i >= len(self.mdp.getStates())):
            i = i % len(self.mdp.getStates())
          state = self.mdp.getStates()[i]
          if self.mdp.isTerminal(state):
            continue
          else:
            temp = self.computeActionFromValues(state)
            result = self.computeQValueFromValues(state, temp)              
            self.values[state] = result
        return None

class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        pred = {}
        store = util.PriorityQueue()
        
        for state in self.mdp.getStates():
          if (len(self.mdp.getPossibleActions(state)) == 0):
            continue
          else:
            temp = -float('inf')
            for action in self.mdp.getPossibleActions(state):
              temp = max(self.computeQValueFromValues(state, action), temp)
              for elem in self.mdp.getTransitionStatesAndProbs(state, action):
                if elem[0] in pred:
                  pred[elem[0]].add(state)
                else:
                  pred[elem[0]] = {state}
            diff = abs(self.values[state] - temp)
            store.update(state, - diff)

        for itr in range(self.iterations):
          if store.isEmpty():
            break
          s = store.pop()
          if (len(self.mdp.getPossibleActions(state)) == 0):
              continue
          else:
            temp = -float('inf')
            for action in self.mdp.getPossibleActions(s):
              if self.computeQValueFromValues(s, action) > temp:
                temp = self.computeQValueFromValues(s, action)
            self.values[s] = temp

          for p in pred[s]:
            if (len(self.mdp.getPossibleActions(state)) == 0):
              continue
            else:
              temp = -float('inf')
              for action in self.mdp.getPossibleActions(p):
                if self.computeQValueFromValues(p, action) > temp:
                  temp = self.computeQValueFromValues(p, action)
              diff = abs(self.values[p] - temp)
              if (diff > self.theta):
                store.update(p, -diff)

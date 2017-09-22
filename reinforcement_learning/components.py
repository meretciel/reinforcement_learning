from collections import defaultdict
import math


POS_INFINITY = float( 'inf' )
NEG_INFINITY = -float( 'inf' )

class RLProcessor( object ):
    """
    RLProcessor class is responsible for the learning process.'

    Args:
        states: a list of states. This defines all the possible states in the problem.
        env:    An Environment object representing the external environment.
        initialValues: The initial value of each state. If not specified, the default value is 0.
    """

    def __init__( self, states, env, initialValues = None, maxValueGap=0.01 ):
        self._states     = states self._env        = env
        self._policy     = Policy( env )
        self.maxValueGap = maxValueGap

        # set up the initial values for sates if available
        self._values = defaultdict( lambda : 0 )
        if initialValues:
            for state, initialValue in initialValues.iteritems():
                self._values[ state.getHash() ] = initialValue


    def doPolicyEvaluation( self ):
        ''' perform policy evaluation '''
        gap = -1
        while gap < 0 or gap > self.maxValueGap:
            gap = 0.
            for state in self._states:
                hashCode   = state.getHash()
                oldValue   = self._values[ hashCode ]
                action     = self._policy.getAction( state )
                newValue   = self._updateValue( state, action )
                self._values[ hashCode ]   = newValue
                state.setValue( newValue )
                gap        = max( gap, abs( oldValue - newValue ) )

    def getStateValue( self, state ):
        ''' get the current value of a given state '''
        return self._values[ state.getHash() ]


    def getPolicy( self ):
        return self._policy


    def getValues( self ):
        return self._values


    def _updateValue( self, state, action ):
        ''' return the updated value of the a state '''
        distributionOfNextState = self._env.getDistributionOfNextState( state, action )
        value = 0
        gamma = self._gamma
        for state, prob in distributionOfNextState.getDistribution().iteritems():
            value += prob * ( reward + gamma * self.getStateValue( state ) )
        return value


    def getDistributionOfNextState( self, state, action ):
        raise NotImplementedError

    def _findGreedyAction( self,  state ):
        max_reward = NEG_INFINITY
        optimalAction = None
        for action in state.getActions():
            reward = 0
            for nextState, prob in self.getDistribution( state, action ):
                reward +=  prob * ( self._env.getReward( state, action, nextState ) + self._env.discountFactor * self._values[ nextState.getHash() ] )

            if reward > max_reward:
                max_reward = reward
                optimalAction = action

        return optimalAction




    def doPolicyImprovement( self ):
        ''' perform policy improvement '''
        isPolicyStable = True

        for state in self.states:
            oldAction = self._policy.getAction( state )
            greedyAction = self._findGreedyAction( state )
            self._policy.updateActionOfState( state,  greedyAction )
            if not oldAction.isEqualTo( greedyAction ):
                isPolicyStable = False

        return isPolicyStable


    def run( self ):
        isPolicyStable = False
        while not isPolicyStable:
            self.doPolicyEvaluation();
            isPolicyStable = self.doPolicyImprovement()





class FiniteDistribution( object ):


    def __init__( self ):
        self._probTable = {}

    def getStates( self ):
        ''' return states '''
        pass

    def getDistribution( self ):
        ''' 
        return probability distribution of different states
        in a form of dictionary.
        '''
        pass

    def normalize( self ):
        sum_prob = sum( self._probTable.values() )
        assert sum_prob != 0, "Toal probability cannot be zero."

        for key, value in self._probTable.iteritems():
            self._probTable[ key ] = value / sum_prob



        
class Policy( object ):
    def __init__( self, env=None ):
        self._greedyAction = dict()
        self._env = env

    def getAction( self, state ):
        stateHash = state.getHash()
        if stateHash in self._greedyAction:
            return self._greedyAction[ stateHash ]

        if self._env:   # if we have a modle of the environment
            env          = self._env
            discount     = env.getDiscountFactor()
            max_value    = NEG_INFINITY
            greedyAction = None

            for action in state.getActions():
                distributionOfNextState = env.getDistributionOfNextState( state, action )
                value = 0
                for nextState, prob in distributionOfNextState:
                    valueOfNextState = env.getValue( nextState )
                    value += prob * ( env.getReward( state, action, nextState ) + discount * valueOfNextState )

                if value > max_value:
                    max_value = value
                    greedyAction = action

            self._greedyAction[ stateHash ] = greedyAction

            return greedyAction
        
        
    def getActions( self, state ):
        raise NotImplementedError

    def updateActionOfState( self, state, action ):
        self._greedyAction[ state.getHash() ] = action








class State( object ):
    _value = None

    def getActions( self ):
        ''' return available actions for this state '''

    def getHash( self ):
        raise NotImplementedError

    def getDistributionOfNextState( self, state, action ):
        ''' 
        returns a distribution object that represents the probability
        distribution of states for the next step. 
        '''
        raise NotImplementedError

    def setValue( self, value ):
        self._value = value
        
    def getValue( self ):
        ''' get the curret estimate of value '''
        raise self._value


class Action( object ):
    def isEqualTo( self, other ):
        raise NotImplementedError

class Environment( object ):
    def getDistributionOfNextState( self, state, action ):
        ''' 
        returns a distribution object that represents the probability
        distribution of states for the next step. 
        '''
        raise NotImplementedError


    def getReward( self, state, action, nextState ):
        raise NotImplementedError


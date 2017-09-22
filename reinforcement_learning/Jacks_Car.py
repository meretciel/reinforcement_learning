import scipy.stats
from reinforcement_learning.components import State, Environment, Policy, Action
import itertools

class JacksAction( Action ):
    def __init__( self, rent_1, rent_2, numMoveFromFirstToSecond ):
        self._rent_1 = rent_1
        self._rent_2 = rent_2
        self._numMoveFromFirstToSecond = numMoveFromFirstToSecond

    def isEqualTo( self, other ):
        return self.getHash() == other.getHash()

    def getHash( self ):
        return ( self._rent_1, self._rent_2, self._numMoveFromFirstToSecond )

    def getNumMoveFromFirstToSecond( self ):
        return self._numMoveFromFirstToSecond


class CarBusinessState( State ):
    def __init__( self, requestFirstLoc, requestSecondLoc,  returnedFirstLoc, returnedSecondLoc, currentFirstLoc, currentSecondLoc ):
        self._rq_1   = requestFirstLoc
        self._rq_2   = requestSecondLoc
        self._ret_1  = returnedFirstLoc
        self._ret_2  = returnedSecondLoc
        self._curr_1 = currentFirstLoc
        self._curr_2 = currentSecondLoc

    def getHash( self ):
        ''' return a hashable object '''
        return ( self._rq_1, self._rq_2, self._ret_1, self._ret_2, self._curr_1, self._curr_2 )

    def getActions( self ):
        output = []
        for numMoveFromFirstToSecond in xrange( -5, 6 ):
            curr_1 = self._curr_1 + numMoveFromFirstToSecond
            curr_2 = self._curr_2 - numMoveFromFirstToSecond

            max_rent_1 = min( self._rq_1, curr_1 + self._ret_1 )
            max_rent_2 = min( self._rq_2, curr_2 + self._ret_2 )

            for rent_1, rent_2 in itertools.product( xrange( max_rent_1 + 1 ), xrange( max_rent_2 + 1 ) ):
                output.append( JacksAction( rent_1, rent_2, numMoveFromFirstToSecond ) )

        return output



class ModeledEnv( Environment ):
    maxCarInEachLocation = 20
    minProb              = 0.0002
    lambdas              = [ 4, 3, 3, 2 ] 
    model                = scipy.stats.poisson
    poissonDistributionTable    = None
    stateDistributionTable      = dict()
    

    def computeProb( self, _n, _lambda ):
        return ModelEnv.model.pmf( _lambda, _n )

    def getDistributionOfNextState( self, state, action ):
        assert action in state.getActions()

        rq_1, rq_2, ret_1, ret_2, curr_1, curr_2 = state.getHash()
        increment_1 = ret_1 - rq_1
        increment_2 = ret_2 - rq_2
        
        numMoveFromFirstToSecond = action.getNumMoveFromFirstToSecond()
        _curr_1 = min( curr_1 + increment_1 - numMoveFromFirstToSecond, self.maxCarInEachLocation )
        _curr_2 = min( curr_2 + increment_2 + numMoveFromFirstToSecond, self.maxCarInEachLocation )

        return self._constructDistribution( _curr_1, _curr_2 )


    def _constructDistribution( self, curr_1, curr_2 ):
        if ( curr_1, curr_2 ) in self.stateDistributionTable:
            return self.stateDistributionTable[ ( curr_1, curr_2 ) ]

        if not self.poissonDistributionTable is None:
            _distribution = { CarBusinessState( a,b,c,d,curr_1, curr_2 ) : prob for ( a,b,c,d ), prob in self.poissonDistributionTable.iteritems() }
            self.stateDistributionTable[ ( curr_1, curr_2 ) ] = _distribution
            return _distribution

        else:
            s = [ 0,0,0,0 ]
            distribution = dict()
            self._constructDistribution_aux( s, 0, distribution, 1 )

            sum_prob = sum( distribution.keys() )
            assert sum_prob > 0

            self.poissonDistributionTable = { state : prob / sum_pob for state, prob in distribution.iteritems() }

            return self._constructDistribution( curr_1, curr_2 )


    def _constructDistribution_aux( self, s, idx, distribution, currProb ):
        if idx == len( s ):
            rq_1, rq_2, ret_1, ret_2 = s
            state = CarBusinessState(  rq_1, rq_2, ret_1, ret_2, curr_1, curr_2 ) 
            distribution[ tuple( s ) ] = currProb
            return 

        n = 0
        param = self.lambdas[ idx ]
        prob = self.computeProb( n, param )
        overallProb = prob * currProb

        while overallProb > self.minProb:
            self._constructDistribution( s, idx + 1, curr_1, curr_2, distribution, overallProb )
            n += 1
            prob = self.computeProb( n, param )
            overallProb = prob * currProb
            s[ idx ] = n


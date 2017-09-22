

import scipy.stats
import pandas as pd 




def constructDistribution( s, idx, distribution, currProb ):
    if idx == len( s ):
        rq_1, rq_2, ret_1, ret_2 = s
        distribution[ tuple( s ) ] = currProb
        return 


def computeProb( n, param ):
   return scipy.stats.poisson.pmf( n, param ) 


def constructDistribution_aux( s, idx, distribution, currProb, lambdas, minProb ):
    identifier =  "=> idx: {}, s: {}".format( idx, s ) 
    print( identifier )

    if idx == len( s ):
        rq_1, rq_2, ret_1, ret_2 = s
        distribution[ tuple( s ) ] = currProb
        return 

    n = 0
    param = lambdas[ idx ]
    prob = computeProb( n, param )
    overallProb = prob * currProb

    print( "{} => currProb: {}, prob: {}, overallProb: {}".format( 
        identifier, currProb, prob, overallProb ) )

    while overallProb > minProb:
        constructDistribution_aux( s, idx + 1, distribution, overallProb, lambdas, minProb )
        n += 1
        prob = computeProb( n, param )
        overallProb = prob * currProb
        s[ idx ] = n



if __name__ =='__main__':

    lambdas = [ 4, 3, 3, 2 ] 
    minProb = 0.0005

    print( "construct distribution." )
    s = [ 0,0,0,0 ]
    distribution = dict()
    constructDistribution_aux( s, 0, distribution, 1, lambdas, minProb )

    print( "distribution is constructed." )

    print( "len( distribution ): {}".format( len( distribution ) ) )
    sum_prob = sum( distribution.values() )
    assert sum_prob > 0

    d_dist = { state : prob / sum_prob for state, prob in distribution.iteritems() }




    for state, prob in d_dist.iteritems():
        print state, prob

    data = [ list( state ) + [ prob ] for state, prob in d_dist.iteritems() ]


    columns = [ 'rq_1', 'rq_2', 'ret_1', 'ret_2', 'prob' ]
    df = pd.DataFrame( data, columns = columns )

    df = df.sort_values( 'prob', ascending=False )
    df.to_csv( "./data/state_distribution.csv" )

    print( df )




import reinforcement_learning.components as comp
from Jacks_Car import CarBusinessState, ModeledEnv
from os import path 
import pandas as pd 




if __name__ == '__main__':


    data_dir  = r"/home/ruikun/workspace/python/reinforcement_learning/data"
    df_dist   = pd.read_csv( path.join( data_dir, 'state_distribution.csv' ) )

    requestReturnStates = df_dist.drop( 'prob', axis=1 ).values


    print( "construct states." )
    states = [] 
    for ( rq_1, rq_2, ret_1, ret_2 ) in requestReturnStates:
        for i in xrange( 20 ):
            for j in xrange( 20 ):
                states.append( CarBusinessState( rq_1, rq_2, ret_1, ret_2, i, j ) )

    print ( "total {} states are created.".format( len( states ) ) )

    print( "construct processor." )

    env               = ModeledEnv()
    learningProcessor = comp.RLProcessor( states, env )

    print( "Processor will run shortly." )
    learningProcessor.run()


    print( "get value of states" )
    stateValues = learningProcessor.getValues()






    



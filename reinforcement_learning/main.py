
import reinforcement_learning.components as comp
from Jacks_Car import CarBusinessState


if __name__ == '__main__':

    print( "construct states." )
    states = [] 
    for i0 in xrange( 10 ):
        for i1 in xrange( 10 ):
            for i2 in xrange( 10 ):
                for i3 in xrange( 10 ):
                    for i4 in xrange( 20 ):
                        for i5 in xrange( 20 ):
                            states.append( CarBusinessState( i0, i1, i2, i3, i4, i5 ) )

    print( "states are ready." )                            

    print( "construct processor." )
    learningProcessor = comp.RLProcessor( states )

    print( "Processor will run shortly." )
    learningProcessor.run()


    print( "get value of states" )
    stateValues = learningProcessor.getValues()






    



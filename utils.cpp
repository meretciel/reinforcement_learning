
#include "utils.h"



int utils::random_int_uniform_distribution(int lower_bound, int upper_bound)
{
    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::uniform_int_distribution<int> distr(lower_bound, upper_bound);
    return distr(gen);
}



double utils::random_real_uniform_distribution_0_1()
{
    static std::default_random_engine generator;
    static std::uniform_real_distribution<double> distr(0.0, 1.0);
    return distr(generator);
}



    

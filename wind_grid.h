/*
 * wind_grid.h
 *
 *  Created on: Oct 6, 2017
 *      Author: ruikun
 */

#ifndef WIND_GRID_H_
#define WIND_GRID_H_


#include<utility>
#include<vector>
#include<map>
#include<string>

namespace wind_grid
{

    using Wind = std::vector<int>;
    using Action = std::pair<int,int>;
    using Position = std::pair<int,int>;
    using SA_Pair  = std::pair<Position, Action>;
    using Q_Function = std::map< SA_Pair, double>;
    using Result = std::vector<int>;

    const std::vector<Action> actions_4_directions{
        {0,1}, {0,-1}, {1,0}, {-1,0}
    };

    const std::vector<Action> actions_8_directions{
        {0,1}, {0,-1}, {1,0}, {-1,0},
        {1,1}, {1,-1}, {1,-1}, {-1,-1}
    };


    class Wind_Grid_Problem
    {
        private:
        int self_n_rows;
        int self_n_cols;
        double self_alpha;
        double self_gamma;
        double self_epsilon;
        int self_n_steps;
        int self_n_directions;
        Wind self_wind;
        const Position self_start_position;
        const Position self_target_position;
        Q_Function self_Q_function;
        Result self_result;
        std::vector<Action> self_all_actions;



        public:
        Wind_Grid_Problem( int n_rows, int n_cols, double alpha, double gamma, double epsilon, int n_steps, int n_directions, Wind wind, Position start_position, Position target_position);
        std::vector<Action> get_available_actions(const Position& current_position);
        Position get_next_position(const Position& current_position, const Action& action);

        Action get_greedy_action(const Position& current_position);
        Action get_action_by_epsilon_greedy(const Position& current_position);


        void run(const std::string& algo);
        void save_result(const std::string& filename);

        Position update_Q_function_SARSA(const Position& current_position);
        Position update_Q_function_Q_learning(const Position& current_position);
        Position update_Q_function_Expectation(const Position& current_positino);
    };


}




#endif /* WIND_GRID_H_ */

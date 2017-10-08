
#include <iostream>
#include <fstream>
#include "wind_grid.h"
#include "utils.h"
#include <tuple>
#include <algorithm>
#include <limits>
#include <utility>

using wind_grid::Position;
using wind_grid::Action;
using wind_grid::Result;
using wind_grid::Wind;
using wind_grid::Q_Function;
using wind_grid::SA_Pair;


wind_grid::Wind_Grid_Problem::Wind_Grid_Problem( 
        int n_rows, int n_cols, double alpha, double gamma, double epsilon, int n_steps, int n_directions, Wind wind,
        Position start_position, Position target_position
):
    self_n_rows(n_rows),
    self_n_cols(n_cols),
    self_alpha(alpha),
    self_gamma(gamma),
    self_epsilon(epsilon),
    self_n_steps(n_steps),
    self_n_directions(n_directions),
    self_wind(wind),
    self_start_position(start_position),
    self_target_position(target_position)
{

    if (self_n_directions == 4)
        self_all_actions = wind_grid::actions_4_directions;
    else if (self_n_directions == 8)
        self_all_actions = wind_grid::actions_8_directions;


}




std::vector<Action> wind_grid::Wind_Grid_Problem::get_available_actions(const Position& current_position)
{
    std::vector<Action> output{};

    for (auto& action : self_all_actions)
    {
        int dx, dy;
        std::tie(dx, dy) = action;

        int current_x, current_y;
        std::tie(current_x, current_y) = current_position;

        int new_x, new_y;
        new_x = current_x + dx;
        new_y = current_y + dy;

        if (new_x >= 0 && new_y >= 0 && new_x < self_n_cols && new_y < self_n_rows) output.emplace_back(action);

    }
    return output;

}




Position wind_grid::Wind_Grid_Problem::get_next_position(const Position& current_position, const Action& action)
{
    int current_x, current_y;
    std::tie(current_x, current_y) = current_position;

    int dx, dy;
    std::tie(dx, dy) = action;


    int new_x, new_y;
    new_x = current_x + dx;
    new_y = current_y + dy + self_wind[current_x];

    new_x = std::min(std::max(new_x, 0), self_n_cols - 1);
    new_y = std::min(std::max(new_y, 0), self_n_rows - 1);

    return std::make_pair(new_x, new_y);

}



Action wind_grid::Wind_Grid_Problem::get_greedy_action(const Position& current_position)
{
    double max_value = std::numeric_limits<double>::lowest();

    Action result{};

    auto actions = get_available_actions(current_position);

    for (auto& action : actions)
    {
        auto state_action = std::make_pair(current_position, action);

        auto value{0};
        if (self_Q_function.find(state_action) != self_Q_function.end()) value = self_Q_function.at(state_action);

        if (value > max_value)
        {
            max_value = value;
            result = action;
        }

    }

    return result;
}



Action wind_grid::Wind_Grid_Problem::get_action_by_epsilon_greedy(const Position& current_position)
{
    auto random_num = utils::random_real_uniform_distribution_0_1();

    if (random_num < self_epsilon)
    {
        auto actions = get_available_actions(current_position);
        auto index = utils::random_int_uniform_distribution(0, actions.size()-1);
        return actions[index];
    }
    else
    {
        auto greedy_action = get_greedy_action(current_position);
        return greedy_action;
    }

}


void wind_grid::Wind_Grid_Problem::run(const std::string& algo)
{

    auto current_position = self_start_position;

    using Update_Function = Position(wind_grid::Wind_Grid_Problem::*)(const Position&);
    Update_Function update_Q_function = nullptr;

    if (algo == "SARSA")
        update_Q_function = &wind_grid::Wind_Grid_Problem::update_Q_function_SARSA;
    else if (algo == "Q_learning")
        update_Q_function = &wind_grid::Wind_Grid_Problem::update_Q_function_Q_learning;
    else if (algo == "Expectation")
        update_Q_function = &wind_grid::Wind_Grid_Problem::update_Q_function_Expectation;


    int n_episodes{0};
    self_result.clear();
    self_Q_function.clear();
    for (int i = 0; i < self_n_steps; ++i)
    {
        current_position = (this->*update_Q_function)(current_position);

        if (current_position == self_target_position)
        {
            ++n_episodes;
            current_position = self_start_position;
        }

        self_result.push_back(n_episodes);
    }


}


void wind_grid::Wind_Grid_Problem::save_result(const std::string& filename)
{
    std::ofstream out_file(filename);
    // write header
    out_file << "step, episodes" << '\n';

    for (size_t i = 0; i < self_result.size(); ++i)
        out_file << i << "," << self_result[i] << '\n';
    out_file.close();
}




Position wind_grid::Wind_Grid_Problem::update_Q_function_SARSA(const Position& current_position)
{
    auto current_action = get_action_by_epsilon_greedy(current_position);
    auto next_position  = get_next_position(current_position, current_action);
    auto next_action    = get_action_by_epsilon_greedy(next_position);

    auto current_pair = std::make_pair(current_position, current_action);
    auto next_pair    = std::make_pair(next_position, next_action);

    if (self_Q_function.find(current_pair) == self_Q_function.end()) self_Q_function[current_pair] = 0.0;
    if (self_Q_function.find(next_pair) == self_Q_function.end()) self_Q_function[next_pair] = 0.0;

    auto current_value = self_Q_function[current_pair];
    auto next_value    = self_Q_function[next_pair];
    std:: cout << '(' << current_position.first << ',' << current_position.second << ") -> ("
        << next_position.first << ',' << next_position.second << "), action: (" 
        << current_action.first << ',' << current_action.second << ')' 
        << " value: " << current_value << ", " << next_value 
        << '\n';
    self_Q_function[current_pair] = current_value + self_alpha * ( -1 + self_gamma * next_value - current_value);

    return next_position;
}



Position wind_grid::Wind_Grid_Problem::update_Q_function_Q_learning(const Position& current_position)
{
    auto current_action = get_action_by_epsilon_greedy(current_position);
    auto next_position  = get_next_position(current_position, current_action);
    auto next_action    = get_greedy_action(next_position);

    auto current_pair = std::make_pair(current_position, current_action);
    auto next_pair    = std::make_pair(next_position, next_action);

    if (self_Q_function.find(current_pair) == self_Q_function.end()) self_Q_function[current_pair] = 0.0;
    if (self_Q_function.find(next_pair) == self_Q_function.end()) self_Q_function[next_pair] = 0.0;

    auto current_value = self_Q_function[current_pair];
    auto next_value    = self_Q_function[next_pair];

    std:: cout << '(' << current_position.first << ',' << current_position.second << ") -> ("
        << next_position.first << ',' << next_position.second << "), action: (" 
        << current_action.first << ',' << current_action.second << ')' 
        << " value: " << current_value << ", " << next_value 
        << '\n';


    self_Q_function[current_pair] = current_value + self_alpha * ( -1 + self_gamma * next_value - current_value);

    return next_position;

}



Position wind_grid::Wind_Grid_Problem::update_Q_function_Expectation(const Position& current_position)
{

    auto current_action      = get_action_by_epsilon_greedy(current_position);
    auto next_position       = get_next_position(current_position, current_action);
    auto next_greedy_action  = get_greedy_action(next_position);

    auto current_pair = std::make_pair(current_position, current_action);
    auto next_greedy_pair    = std::make_pair(next_position, next_greedy_action);

    if (self_Q_function.find(current_pair) == self_Q_function.end()) self_Q_function[current_pair] = 0.0;
    if (self_Q_function.find(next_greedy_pair) == self_Q_function.end()) self_Q_function[next_greedy_pair] = 0.0;
    auto current_value = self_Q_function[current_pair];
    auto next_greedy_value = self_Q_function[next_greedy_pair];

    // calculate the expectation of the next value
    double total_value{0};
    auto next_available_actions = get_available_actions(next_position);
    for (const auto& action : next_available_actions)
    {
        auto _pair = std::make_pair(next_position, action);
        if (self_Q_function.find(_pair) != self_Q_function.end())
            total_value += self_Q_function.at(_pair);
    }

    auto next_value = self_epsilon * total_value / next_available_actions.size() + (1 - self_epsilon) * next_greedy_value;

    std:: cout << '(' << current_position.first << ',' << current_position.second << ") -> ("
        << next_position.first << ',' << next_position.second << "), action: (" 
        << current_action.first << ',' << current_action.second << ')' 
        << " value: " << current_value << ", " << next_value 
        << '\n';

    self_Q_function[current_pair] = current_value + self_alpha * ( -1 + self_gamma * next_value - current_value);

    return next_position;

}

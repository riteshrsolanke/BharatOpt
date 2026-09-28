#pragma once
#include <string>
#include <vector>

namespace bharatopt {

enum class NodeStatus {
    OPEN,
    INFEASIBLE,
    INTEGER,
    PRUNED_BY_BOUND,
    COMPLETED
};

struct MILPNode {
    int id = 0;
    int depth = 0;
    int parent_id = -1;
    double bound = 0.0;
    double incumbent_at_creation = 0.0;
    int branching_variable = -1;
    double branching_value = 0.0;
    std::string branch_direction = ""; // "<=" or ">="
    NodeStatus status = NodeStatus::OPEN;
};

struct MILPStatistics {
    int nodes_explored = 0;
    int nodes_pruned = 0;
    int cuts_generated = 0;
    int cuts_accepted = 0;
    int lp_relaxations_solved = 0;
};

struct MILPResult {
    double best_bound = 0.0;
    double incumbent = 0.0;
    double mip_gap = 0.0;
    std::vector<double> solution;
    MILPStatistics stats;
    std::vector<MILPNode> tree_nodes;
};

} // namespace bharatopt
